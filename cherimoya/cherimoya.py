# cherimoya.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
An implementation of the Cherimoya deep learning model, a compact
architecture for predicting genomic modalities from sequence alone.
"""

import time
import numpy

import torch

from .cheri import CheriBlock
from .losses import _mixture_loss
from .performance import calculate_performance_measures

from tangermeme.predict import predict
from bpnetlite.logging import Logger


torch.set_float32_matmul_precision('high')


class EMA:
	"""Exponential moving average of a model's parameters.

	Maintains a shadow copy of every floating-point parameter that is
	updated as ``shadow = decay * shadow + (1 - decay) * parameter`` after
	each training step. The shadow weights are typically used at
	evaluation time, where they tend to produce smoother and more stable
	predictions than the raw running weights.

	Typical usage during training:

	1. Create an EMA wrapper after the model is constructed.
	2. Call :meth:`update` after every optimizer step.
	3. Call :meth:`apply_shadow` before evaluation to swap the shadow
	   weights into the model.
	4. Call :meth:`restore` after evaluation to put the training weights
	   back.

	Parameters
	----------
	model: torch.nn.Module
		The model whose parameters will be tracked.

	decay: float, optional
		The decay factor of the moving average. Larger values place more
		weight on the running shadow and less on each new update. Default
		is 0.999.
	"""

	def __init__(self, model, decay=0.999):
		self.decay = decay
		self.shadow = {}
		self._backup = {}
		for name, p in model.named_parameters():
			if p.requires_grad and p.is_floating_point():
				self.shadow[name] = p.detach().clone()

	@torch.no_grad()
	def update(self, model):
		"""Update the shadow weights using the current model parameters."""

		d = self.decay
		for name, p in model.named_parameters():
			if name in self.shadow:
				self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

	@torch.no_grad()
	def apply_shadow(self, model):
		"""Swap the model's parameters with the shadow weights.

		The original weights are kept in an internal backup so they can
		be restored after evaluation. Calling this method twice in a row
		without an intervening :meth:`restore` is an error.
		"""

		assert not self._backup
		for name, p in model.named_parameters():
			if name in self.shadow:
				self._backup[name] = p.detach().clone()
				p.data.copy_(self.shadow[name].data)

	@torch.no_grad()
	def restore(self, model):
		"""Put the original training weights back into the model."""

		for name, p in model.named_parameters():
			if name in self._backup:
				p.data.copy_(self._backup[name].data)
		self._backup = {}


class Cherimoya(torch.nn.Module):
	"""The Cherimoya sequence-to-function model.

	Parameters
	----------
	n_filters: int, optional
		Width of the convolutional backbone (the channel dimension).
		Default is 96.

	n_layers: int, optional
		Number of stacked Cheri Blocks. Block ``i`` uses dilation
		``2**i``. Default is 9.

	n_outputs: int, optional
		Number of output tracks for the profile head. Default is 1.

	n_control_tracks: int, optional
		Number of control input tracks. If 0, the model takes only the
		one-hot sequence as input. Default is 0.

	expansion: int, optional
		Channel-expansion factor for the MLP inside each Cheri Block. The
		inner projection maps ``n_filters -> expansion * n_filters`` and
		then back. Default is 2.

	residual_scale: float, optional
		Fixed scalar applied to the MLP output of each Cheri Block before
		it is added back to the residual stream. Default is 0.15.

	name: str or None, optional
		Display name used when saving model files. Defaults to
		``"cherimoya.{n_filters}.{n_layers}"``.

	trimming: int or None, optional
		Number of base pairs to trim from each side of the input when
		producing the output profile. If None, defaults to
		``46 + sum(2**i for i in range(n_layers))``.

	single_count_output: bool, optional
		If True, the count head returns a single scalar per example;
		otherwise it returns one count per output track. Default is True.

	verbose: bool, optional
		Whether the training-progress logger prints to stdout. Default
		is True.
	"""

	def __init__(self, n_filters=96, n_layers=9, n_outputs=1,
		n_control_tracks=0, expansion=2, residual_scale=0.15, name=None,
		trimming=None, single_count_output=True, verbose=True):
		super(Cherimoya, self).__init__()
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_outputs = n_outputs
		self.n_control_tracks = n_control_tracks
		self.expansion = expansion
		self.residual_scale = residual_scale
		self.single_count_output = single_count_output

		self.name = name or "cherimoya.{}.{}".format(n_filters, n_layers)
		self.trimming = trimming if trimming is not None else (
			46 + sum(2**i for i in range(n_layers)))

		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
		self.igelu = torch.nn.GELU(approximate='tanh')

		self.blocks = torch.nn.ModuleList([
			CheriBlock(n_filters, 2**i, expansion=expansion,
				residual_scale=residual_scale)
			for i in range(self.n_layers)
		])

		self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs,
			kernel_size=1, padding=0)

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

	def _init_kwargs(self):
		"""Return the kwargs needed to reconstruct this model."""
		return {
			'n_filters': self.n_filters,
			'n_layers': self.n_layers,
			'n_outputs': self.n_outputs,
			'n_control_tracks': self.n_control_tracks,
			'expansion': self.expansion,
			'residual_scale': self.residual_scale,
			'name': self.name,
			'trimming': self.trimming,
			'single_count_output': self.single_count_output,
			'verbose': False,
		}

	def save(self, path):
		"""Save the model to a file.

		The checkpoint stores the constructor arguments needed to rebuild
		the model along with its parameter state dict. This format can be
		loaded with ``weights_only=True`` and is robust to changes in
		source layout.

		Parameters
		----------
		path: str
			The destination file path.
		"""

		payload = {
			'config': self._init_kwargs(),
			'state_dict': self.state_dict(),
		}
		torch.save(payload, path)

	@classmethod
	def load(cls, path, device='cpu'):
		"""Load a model previously saved with :meth:`save`.

		Parameters
		----------
		path: str
			The checkpoint file path.

		device: str or torch.device, optional
			Device to map the parameters onto. Default is ``'cpu'``.

		Returns
		-------
		model: Cherimoya
			The reconstructed model, placed on ``device``.
		"""

		payload = torch.load(path, map_location=device, weights_only=True)
		model = cls(**payload['config'])
		model.load_state_dict(payload['state_dict'])
		return model.to(device)

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
		X = torch.mean(X[:, :, start:end].float(), dim=2)
		if X_ctl is not None:
			X_ctl = torch.sum(X_ctl[:, :, start:end].float(), dim=(1, 2))
			X_ctl = X_ctl.unsqueeze(-1)
			X = torch.cat([X, torch.log(X_ctl+1)], dim=-1)

		y_counts = self.linear(X)
		return y_profile, y_counts

	def fit(self, training_data, muon_optimizer, adam_optimizer, muon_scheduler,
		adam_scheduler, X_valid, X_ctl_valid, y_valid, max_epochs=50, batch_size=64,
		dtype='float32', device='cuda', early_stopping=None):
		"""Fit the model to data and validate it periodically.

		This method controls the training of a Cherimoya model. It will fit
		the model to examples generated by the `training_data` DataLoader
		object and, if validation data is provided, will validate the model
		against it at the end of each epoch and return those values.

		Two versions of the model will be saved using :meth:`save`: the best
		model found during training according to the validation measures, and
		the final model at the end of training. Additionally, a log will be
		saved of the training and validation statistics, e.g. time and
		performance.


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
			number of times that `training_data` is exhausted. Default is 50.

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
		best_corr = float("-inf")
		self.logger.start()

		ema = EMA(self, decay=0.999)

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
					y_hat_logcounts.float())


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

				ema.update(self)

				iteration += 1

			train_time = time.time() - tic

			if self.lw0.requires_grad == True and torch.abs(self.lw0.grad).sum() < 1:
				self.lw0.requires_grad = False
				self.lw1.requires_grad = False

			# Validate the model at the end of the epoch
			with torch.no_grad():
				self.eval()
				ema.apply_shadow(self)

				tic = time.time()

				y_hat_logits, y_hat_logcounts = predict(self, X_valid, args=X_ctl_valid,
					batch_size=batch_size, dtype=dtype, device=device)

				valid_profile_loss, valid_count_loss = _mixture_loss(y_valid,
					y_hat_logits, y_hat_logcounts)

				valid_loss = w0*valid_profile_loss + w1*valid_count_loss

				measures = calculate_performance_measures(y_hat_logits,
					y_valid, y_hat_logcounts, measures=['profile_pearson', 'count_pearson'])

				valid_profile_corr = numpy.nan_to_num(measures['profile_pearson'])
				valid_count_corr = numpy.nan_to_num(measures['count_pearson']).mean()
				valid_time = time.time() - tic

				self.logger.add([epoch,
					iteration,
					train_time,
					valid_time,
					profile_loss.item(),
					count_loss.item(),
					valid_profile_loss.item(),
					valid_profile_corr.mean(),
					valid_count_corr,
					valid_count_loss.item(),
					(valid_count_corr > best_corr).item()])

				self.logger.save("{}.log".format(self.name))

				if valid_count_corr > best_corr:
					self.save("{}.torch".format(self.name))
					best_corr = valid_count_corr
					early_stop_count = -1

				ema.restore(self)

			early_stop_count += 1
			if early_stopping is not None and early_stop_count >= early_stopping:
				break

		ema.apply_shadow(self)
		self.save("{}.final.torch".format(self.name))
		return best_corr
