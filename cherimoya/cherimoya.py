# cherimoya.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
An implementing of the Cherimoya deep learning model, which is a compact
architecture for predicting genomic modalities from sequence alone.
"""

import h5py
import time 
import numpy
import torch

import torch
import torch.nn as nn
import triton
import triton.language as tl
import itertools

from .losses import MNLLLoss
from .losses import log1pMSELoss
from .losses import _mixture_loss

from .performance import pearson_corr
from .performance import calculate_performance_measures

from tqdm import tqdm

from tangermeme.predict import predict
from bpnetlite.logging import Logger

torch.backends.cudnn.benchmark = True	
torch.set_float32_matmul_precision('high')


def autotune_configs():
    num_warps = [4, 8, 16]
    num_stages = [2, 3, 4, 5]
    BLOCK_Ls = [32, 64, 128, 256]
    
    configs = []
    for num_warp, num_stage, L in itertools.product(num_warps, num_stages, BLOCK_Ls):
        configs.append(triton.Config({
            'num_warps': num_warp,
            'num_stages': num_stage,
            'BLOCK_L': L
        }))
    return configs


@triton.autotune(
    configs = autotune_configs(),
    key=['C', 'L'],
)
@triton.jit
def fwd_conv_kernel(
    X_ptr, W_ptr, Y_ptr, Mean_ptr, Rstd_ptr,
    stride_xn, dilation, eps,
	L: tl.constexpr,
	C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr
):
	
	pid_n = tl.program_id(0)
	offs_c = tl.arange(0, BLOCK_C)[None, :]
	mask_c = offs_c < C
	
	w_idx = W_ptr + offs_c
	w0 = tl.load(w_idx,       mask=mask_c, other=0.0).to(tl.float32)
	w1 = tl.load(w_idx + C,   mask=mask_c, other=0.0).to(tl.float32)
	w2 = tl.load(w_idx + C*2, mask=mask_c, other=0.0).to(tl.float32)
	
	running_sum = 0.0
	running_sq_sum = 0.0
	for l_start in tl.range(0, L, BLOCK_L):
		offs = l_start + tl.arange(0, BLOCK_L)[:, None]
		offs_l = offs - dilation
		offs_r = offs + dilation
		
		mask = (offs < L) & mask_c
		mask_l = (offs_l >= 0) & mask
		mask_r = (offs_r < L) & mask
		
		x_idx = X_ptr + pid_n * stride_xn + offs_c
		x1 = tl.load(x_idx + offs*C,   mask=mask,   other=0.0).to(tl.float32)
		x0 = tl.load(x_idx + offs_l*C, mask=mask_l, other=0.0).to(tl.float32)
		x2 = tl.load(x_idx + offs_r*C, mask=mask_r, other=0.0).to(tl.float32)
		
		conv = x0*w0 + x1*w1 + x2*w2
		conv2 = conv * conv
	
		running_sum += tl.sum(conv)
		running_sq_sum += tl.sum(conv2)
	
	count = L * C
	mean = running_sum / count
	var = (running_sq_sum / count) - (mean * mean)
	rstd = 1.0 / tl.sqrt(var + eps)
	
	tl.store(Mean_ptr + pid_n, mean)
	tl.store(Rstd_ptr + pid_n, rstd)
	
	for l_start in tl.range(0, L, BLOCK_L):
		offs = l_start + tl.arange(0, BLOCK_L)[:, None]
		offs_l = offs - dilation
		offs_r = offs + dilation
		
		mask = (offs < L) & mask_c
		mask_l = (offs_l >= 0) & mask
		mask_r = (offs_r < L) & mask
		
		x_idx = X_ptr + pid_n * stride_xn + offs_c
		x1 = tl.load(x_idx + offs*C,   mask=mask, other=0.0).to(tl.float32)
		x0 = tl.load(x_idx + offs_l*C, mask=mask_l, other=0.0).to(tl.float32)
		x2 = tl.load(x_idx + offs_r*C, mask=mask_r, other=0.0).to(tl.float32)
	
		conv = x0*w0 + x1*w1 + x2*w2
		x_hat = (conv - mean) * rstd
	
		y_idx = Y_ptr + pid_n * stride_xn + offs * C + offs_c
		tl.store(y_idx, x_hat, mask=mask)


@triton.autotune(
    configs = autotune_configs(),
    key=['C', 'L']
)
@triton.jit
def bwd_conv_kernel(
	dY_ptr, X_ptr, W_ptr, Mean_ptr, Rstd_ptr,
    dX_ptr, dW_ptr, stride_xn, dilation,
	L: tl.constexpr, 
	C: tl.constexpr, 
	BLOCK_C: tl.constexpr, 
	BLOCK_L: tl.constexpr
):
	pid_n = tl.program_id(0)
	offs_c = tl.arange(0, BLOCK_C)
	mask_c = offs_c < C
	
	mean = tl.load(Mean_ptr + pid_n)
	rstd = tl.load(Rstd_ptr + pid_n)
	
	w_idx = W_ptr + offs_c
	w0 = tl.load(w_idx,       mask=mask_c)[None, :].to(tl.float32)
	w1 = tl.load(w_idx + C,   mask=mask_c)[None, :].to(tl.float32)
	w2 = tl.load(w_idx + C*2, mask=mask_c)[None, :].to(tl.float32)
	
	sum_dy = 0.0
	sum_dy_xhat = 0.0
	
	mask_c = mask_c[None, :]
	offs_c = offs_c[None, :]
	
	for l_start in tl.range(0, L, BLOCK_L):
		offs = l_start + tl.arange(0, BLOCK_L)[:, None]
		offs_l = offs - dilation
		offs_r = offs + dilation
		
		mask = (offs < L) & mask_c
		mask_l = (offs_l >= 0) & mask
		mask_r = (offs_r < L) & mask
		
		x_idx = X_ptr + pid_n * stride_xn + offs_c
		x1 = tl.load(x_idx + offs*C,   mask=mask, other=0.0).to(tl.float32)
		x0 = tl.load(x_idx + offs_l*C, mask=mask_l, other=0.0).to(tl.float32)
		x2 = tl.load(x_idx + offs_r*C, mask=mask_r, other=0.0).to(tl.float32)
		
		conv = x0*w0 + x1*w1 + x2*w2
		x_hat = (conv - mean) * rstd
		
		y_idx = dY_ptr + pid_n * stride_xn + offs * C + offs_c
		dy = tl.load(y_idx, mask=mask, other=0.0).to(tl.float32)
		
		sum_dy += tl.sum(dy)
		sum_dy_xhat += tl.sum(dy * x_hat)
	
	dw0 = tl.zeros((1, BLOCK_C), dtype=tl.float32)
	dw1 = tl.zeros((1, BLOCK_C), dtype=tl.float32)
	dw2 = tl.zeros((1, BLOCK_C), dtype=tl.float32)
	
	for l_start in tl.range(0, L, BLOCK_L):
		offs = l_start + tl.arange(0, BLOCK_L)[:, None]
		offs_l = offs - dilation
		offs_r = offs + dilation
		
		mask = (offs < L) & mask_c
		mask_l = (offs_l >= 0) & mask
		mask_r = (offs_r < L) & mask
		
		x_idx = X_ptr + pid_n * stride_xn + offs_c
		x1 = tl.load(x_idx + offs*C,   mask=mask, other=0.0).to(tl.float32)
		x0 = tl.load(x_idx + offs_l*C, mask=mask_l, other=0.0).to(tl.float32)
		x2 = tl.load(x_idx + offs_r*C, mask=mask_r, other=0.0).to(tl.float32)
		
		conv = x0*w0 + x1*w1 + x2*w2
		x_hat = (conv - mean) * rstd
	
		###
		
		dy_idx = dY_ptr + pid_n * stride_xn + offs * C + offs_c
		dy = tl.load(dy_idx, mask=mask, other=0.0).to(tl.float32)
	
		count = L * C
		d_conv = (rstd / count) * (count * dy - sum_dy - x_hat * sum_dy_xhat)
		
		dw0 += tl.sum(d_conv * x0, axis=0)[None, :]
		dw1 += tl.sum(d_conv * x1, axis=0)[None, :]
		dw2 += tl.sum(d_conv * x2, axis=0)[None, :]
	
		dx_idx0 = dX_ptr + pid_n * stride_xn + offs * C + offs_c
		
		dx1 = tl.load(dx_idx0, mask=mask, other=0.0)
		dx0 = tl.load(dx_idx0 - dilation*C, mask=mask_l, other=0.0)
		dx2 = tl.load(dx_idx0 + dilation*C, mask=mask_r, other=0.0)
	
		dx1 += d_conv * w1
		dx0 += d_conv * w0
		dx2 += d_conv * w2
	
		tl.store(dx_idx0,              dx1, mask=mask)
		tl.store(dx_idx0 - dilation*C, dx0, mask=mask_l)
		tl.store(dx_idx0 + dilation*C, dx2, mask=mask_r)
	
	
	dw_idx = dW_ptr + pid_n * (C * 3) + offs_c
	tl.store(dw_idx,       dw0, mask=mask_c)
	tl.store(dw_idx + C,   dw1, mask=mask_c)
	tl.store(dw_idx + C*2, dw2, mask=mask_c)


class FusedDilatedConvNormFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, w, dilation):
		N, L, C = x.shape
		BLOCK_C = triton.next_power_of_2(C)
		eps = 1e-3
		
		mean = torch.empty((N,), dtype=torch.float32, device=x.device)
		rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
		y = torch.empty_like(x)
		
		fwd_conv_kernel[(N,)](
			x, w, y, mean, rstd,
			x.stride(0), dilation, eps,
			L, C, BLOCK_C=BLOCK_C
		)
	
		ctx.save_for_backward(x, w, mean, rstd)
		ctx.dilation = dilation
		return y
	
	@staticmethod
	def backward(ctx, dy):
		x, w, mean, rstd = ctx.saved_tensors
		N, L, C = x.shape
		BLOCK_C = triton.next_power_of_2(C)
		
		dy = dy.contiguous()
		dx = torch.zeros_like(x, dtype=x.dtype)
		dw = torch.empty((N, 3, C), device=x.device, dtype=torch.float32)
		
		bwd_conv_kernel[(N,)](
			dy, x, w, mean, rstd, dx, dw, 
			x.stride(0), ctx.dilation,
			L, C, BLOCK_C, 
		)
	
		dw = dw.sum(dim=0)		
		return dx.to(x.dtype), dw.to(x.dtype), None


###

		
class CheriBlock(torch.nn.Module):
	def __init__(self, n_filters, dilation, eps=0.01):
		super().__init__()
		self.n_filters = n_filters
		self.dilation = dilation

		self.conv_weight = torch.nn.Parameter(torch.randn(3, n_filters))	
		self.linear1 = torch.nn.Linear(n_filters, 2*n_filters, bias=False)
		self.linear2 = torch.nn.Linear(2*n_filters, n_filters, bias=False)
		self.gamma = torch.nn.Parameter(torch.ones(1, n_filters) * eps)
		self.activation = torch.nn.GELU(approximate='tanh')

		torch.nn.init.trunc_normal_(self.conv_weight, std=0.02)
		torch.nn.init.trunc_normal_(self.linear1.weight, std=0.02)
		torch.nn.init.trunc_normal_(self.linear2.weight, std=0.02)
	
	def forward(self, X):
		X_conv = FusedDilatedConvNormFunc.apply(X, self.conv_weight, self.dilation)
		X_mlp = self.linear2(self.activation(self.linear1(X_conv)))
		return X + X_mlp * self.gamma
		

class CheriBlock2(torch.nn.Module):
	def __init__(self, n_filters, dilation, eps=0.01):
		super().__init__()
		self.n_filters = n_filters
		self.dilation = dilation

		self.conv = torch.nn.Conv1d(n_filters, n_filters, groups=n_filters, dilation=dilation, padding=dilation, kernel_size=3)
		#self.norm = torch.nn.LayerNorm((n_filters, 2114), elementwise_affine=False, bias=False, eps=1e-3)		
		self.linear1 = torch.nn.Conv1d(n_filters, 3*n_filters, kernel_size=1, bias=False)
		self.linear2 = torch.nn.Conv1d(3*n_filters, n_filters, kernel_size=1, bias=False)
		self.gamma = torch.nn.Parameter(torch.ones(n_filters, 1) * eps) 
		self.activation = torch.nn.GELU(approximate='tanh')
		
		torch.nn.init.trunc_normal_(self.conv.weight, std=0.02)
		torch.nn.init.trunc_normal_(self.linear1.weight, std=0.02)
		torch.nn.init.trunc_normal_(self.linear2.weight, std=0.02)
	
	def forward(self, X):
		X_conv = self.conv(X)
		X_conv = self.norm(X_conv)
		X_mlp = self.linear2(self.activation(self.linear1(X_conv)))
		X = torch.add(X, X_mlp * self.gamma)
		return X



class Cherimoya(torch.nn.Module):
	def __init__(self, n_filters=64, n_layers=9, n_outputs=1, 
		n_control_tracks=0, name=None, trimming=None, 
		single_count_output=True, verbose=True):
		super(Cherimoya, self).__init__()
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_outputs = n_outputs
		self.n_control_tracks = n_control_tracks

		self.name = name or "cherimoya.{}.{}".format(n_filters, n_layers)
		self.trimming = trimming or 46 + sum(2**i for i in range(n_layers))

		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=19, padding=9)
		self.igelu = torch.nn.GELU(approximate='tanh')

		self.blocks = torch.nn.ModuleList([
			CheriBlock(n_filters, 2**i) for i in range(self.n_layers)
		])
		
		self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs, 
			kernel_size=75, padding=37)

		self.lw0 = torch.nn.Parameter(torch.ones(1)) 
		self.lw1 = torch.nn.Parameter(torch.ones(1))

		n_count_control = 1 if n_control_tracks > 0 else 0
		n_count_outputs = 1 if single_count_output else n_outputs
		self.linear = torch.nn.Linear(n_filters+n_count_control, n_count_outputs)
		
		torch.nn.init.trunc_normal_(self.iconv.weight, std=0.02)
		torch.nn.init.trunc_normal_(self.fconv.weight, std=0.02)
		torch.nn.init.trunc_normal_(self.linear.weight, std=0.02)

		torch.nn.init.zeros_(self.iconv.bias)
		torch.nn.init.zeros_(self.fconv.bias)
		torch.nn.init.zeros_(self.linear.bias)

		self.logger = Logger(["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Training Count MSE", 
			"Validation MNLL", "Validation Profile Pearson", 
			"Validation Count Pearson", "Validation Count MSE", "Saved?"], 
			verbose=verbose)


	@torch.compile(mode='max-autotune')
	def forward(self, X, X_ctl=None):
		"""A forward pass of the model.

		This method takes in a nucleotide sequence X, a corresponding
		per-position value from a control track, and a per-locus value
		from the control track and makes predictions for the profile 
		and for the counts. This per-locus value is usually the
		log(sum(X_ctl_profile)+1) when the control is an experimental
		read track but can also be the output from another model.

		Parameters
		----------
		X: torch.tensor, shape=(batch_size, 4, length)
			The one-hot encoded batch of sequences.

		X_ctl: torch.tensor or None, shape=(batch_size, n_strands, length)
			A value representing the signal of the control at each position in 
			the sequence. If no controls, pass in None. Default is None.

		Returns
		-------
		y_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
			The output predictions for each strand trimmed to the output
			length.
		"""

		start, end = self.trimming, X.shape[2] - self.trimming
		
		X = self.igelu(self.iconv(X))
		X = X.transpose(1, 2).contiguous()
		for i in range(self.n_layers):
			X = self.blocks[i](X)

		X = X.transpose(1, 2).contiguous()
		if X_ctl is None:
			X_w_ctl = X
		else:
			X_w_ctl = torch.cat([X, X_ctl], dim=1)

		y_profile = self.fconv(X_w_ctl)[:, :, start:end]

		# counts prediction
		X = torch.mean(X[:, :, start-37:end+37].float(), dim=2)
		if X_ctl is not None:
			X_ctl = torch.sum(X_ctl[:, :, start-37:end+37].float(), dim=(1, 2))
			X_ctl = X_ctl.unsqueeze(-1)
			X = torch.cat([X, torch.log(X_ctl+1)], dim=-1)

		y_counts = self.linear(X)
		return y_profile, y_counts

		
	def fit(self, training_data, muon_optimizer, adam_optimizer, muon_scheduler,
		adam_scheduler, X_valid, X_ctl_valid, y_valid, max_epochs=100, batch_size=64, 
		dtype='float32', device='cuda', early_stopping=None):
		"""Fit the model to data and validate it periodically.

		This method controls the training of a BPNet model. It will fit the
		model to examples generated by the `training_data` DataLoader object
		and, if validation data is provided, will validate the model against 
		it at the end of each epoch and return those values.

		Two versions of the model will be saved: the best model found during
		training according to the validation measures, and the final model
		at the end of training. Additionally, a log will be saved of the
		training and validation statistics, e.g. time and performance.


		Parameters
		----------
		training_data: torch.utils.data.DataLoader
			A generator that produces examples to train on. If n_control_tracks
			is greater than 0, must product two inputs, otherwise must produce
			only one input.

		muon_optimizer: torch.optim.Optimizer
			A Muon optimizer to control the training of the 2D non-head/non-tail layers
			in the model. This is mostly the dense layers and depth-wise convolutions of
			the Cheri blocks.

		adam_optimizer: torch.optim.Optimizer
			An Adam/W optimizer to control the training of the other parametrers. This
			should be the head/tail layers, the bias terms, and any other parameters
			that are not 2D matrices.

		muon_scheduler: torch.optim.lr_scheduler
			The scheduler to use for the Muon optimizer. This should likely be a cosine
			decay with a warmup phase.

		adam_scheduler: torch.optim.lr_scheduler
			The scheduler to use for the Adam/W optimizer. This should likely be the
			same cosine decay with a warmup phase used for the Muon optimizer.

		X_valid: torch.tensor, shape=(n, 4, length)
			A block of sequences to validate on at the end of each epoch.

		X_ctl_valid: torch.tensor or None, shape=(n, n_control_tracks, length)
			A block of control sequences to use for making the validation set
			predictions at the end of each epoch. If n_control_tracks is None, pass in
			None. Default is None.

		y_valid: torch.tensor or None, shape=(n, n_outputs, output_length)
			A block of signals to validate against at the end of each epochs.

		max_epochs: int
			The maximum number of epochs to train for, as measured by the
			number of times that `training_data` is exhausted. Default is 100.

		batch_size: int, optional
			The number of examples to include in each batch. Default is 64.
		
		dtype: str or torch.dtype
			The torch.dtype to use when training. Usually, this will be torch.float32
			or torch.bfloat16. Default is torch.float32.
		
		device: str
			The device to use for training and inference. Typically, this will be
			'cuda' but can be anything supported by torch. Default is 'cuda'.

		early_stopping: int or None, optional
			Whether to stop training early. If None, continue training until
			max_epochs is reached. If an integer, continue training until that
			number of epochs has been hit without improvement in performance. 
			Default is None.
		"""
		
		if X_valid is not None:
			y_valid_counts = y_valid.sum(dim=2)

		if X_ctl_valid is not None:
			X_ctl_valid = (X_ctl_valid,)
			
		dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

		iteration = 0
		early_stop_count = 0
		best_loss = float("inf")
		self.logger.start()
		
		###

		for epoch in range(max_epochs):
			tic = time.time()
			
			for data in training_data:
				X, y, labels = data[0], data[-2], data[-1]
				X_ctl = data[1].to(device) if len(data) == 4 else None

				if X.shape[0] != batch_size:
					continue

				X = X.to(device).float()
				y = y.to(device)
				
				# Clear the optimizer and set the model to training mode
				muon_optimizer.zero_grad()
				adam_optimizer.zero_grad()
				self.train()

				# Make one training step
				with torch.autocast(device_type=device, dtype=dtype):
					y_hat_logits, y_hat_logcounts = self(X, X_ctl)

				profile_loss, count_loss = _mixture_loss(y, y_hat_logits.float(), 
					y_hat_logcounts.float(), )


				w0 = (1.0 / (2.0 * self.lw0 ** 2))
				w1 = (1.0 / (2.0 * self.lw1 ** 2))
				loss = w0*profile_loss + w1*count_loss
				
				if self.lw0.requires_grad == True:
					loss += torch.sum(torch.log(self.lw0) ** 2 + torch.log(self.lw1) ** 2)
				
				loss.backward()
				
				muon_optimizer.step()
				adam_optimizer.step()

				muon_scheduler.step()
				adam_scheduler.step()

				iteration += 1

			train_time = time.time() - tic 
			
			if self.lw0.requires_grad == True and torch.abs(self.lw0.grad).sum() < 1:
				self.lw0.requires_grad = False
				self.lw1.requires_grad = False
				
			# Validate the model at the end of the epoch
			with torch.no_grad():
				self.eval()
				tic = time.time()

				y_hat_logits, y_hat_logcounts = predict(self, X_valid, args=X_ctl_valid, 
					batch_size=batch_size, dtype=dtype, device=device)
				
				valid_profile_loss, valid_count_loss = _mixture_loss(y_valid,
					y_hat_logits, y_hat_logcounts)

				valid_loss = w0*valid_profile_loss + w1*valid_count_loss
				
				measures = calculate_performance_measures(y_hat_logits,
					y_valid, y_hat_logcounts, measures=['profile_pearson', 'count_pearson'])

				valid_profile_corr = numpy.nan_to_num(measures['profile_pearson'])
				valid_count_corr = numpy.nan_to_num(measures['count_pearson'])
				valid_time = time.time() - tic
				
				self.logger.add([epoch, 
					iteration, 
					train_time, 
					valid_time,
					profile_loss.item(), 
					count_loss.item(), 
					valid_profile_loss.item(), 
					valid_profile_corr.mean(),
					valid_count_corr.mean(), 
					valid_count_loss.item(),
					(valid_loss < best_loss).item()])

				self.logger.save("{}.log".format(self.name))
				
				if valid_loss < best_loss:
					torch.save(self, "{}.torch".format(self.name))
					best_loss = valid_loss
					early_stop_count = -1

			early_stop_count += 1
			if early_stopping is not None and early_stop_count >= early_stopping:
				break

		torch.save(self, "{}.final.torch".format(self.name))