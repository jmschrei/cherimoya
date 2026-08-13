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
from .io import _validate_signal_groups
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
		Default is 128.

	n_layers: int, optional
		Number of stacked Cheri Blocks. Block ``i`` uses dilation
		``2**i``. Default is 9.

	signal_groups: list of int, optional
		The number of channels in each signal group. A signal group is
		one biological modality whose channels share an orientation: a
		single-channel (unstranded) track is a group of size 1, a
		stranded ``(+, -)`` pair is a group of size 2. The profile head
		emits one channel per signal channel (total
		``sum(signal_groups)`` outputs); the count head emits one
		prediction per *group* (total ``len(signal_groups)``). Default
		is ``[1]`` — a single unstranded track.

	n_control_tracks: int, optional
		Number of control input tracks (the total channel count
		summed across all control groups, if any). If 0, the model
		takes only the one-hot sequence as input. Default is 0.

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

	verbose: bool, optional
		Whether the training-progress logger prints to stdout. Default
		is True.
	"""

	def __init__(self, n_filters=128, n_layers=9, signal_groups=None,
		n_control_tracks=0, expansion=2, residual_scale=0.15, name=None,
		trimming=None, verbose=True, compile=True,
		compile_mode='max-autotune'):
		super(Cherimoya, self).__init__()

		if signal_groups is None:
			signal_groups = [1]
		signal_groups = list(signal_groups)
		_validate_signal_groups(signal_groups)

		self.signal_groups = signal_groups
		self.n_outputs = sum(signal_groups)
		self.n_groups = len(signal_groups)

		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_control_tracks = n_control_tracks
		self.expansion = expansion
		self.residual_scale = residual_scale

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

		self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks,
			self.n_outputs, kernel_size=75, padding=37)

		n_count_control = 1 if n_control_tracks > 0 else 0
		self.linear = torch.nn.Linear(n_filters+n_count_control, self.n_groups)

		self.lw0 = torch.nn.Parameter(torch.ones(self.n_groups))
		self.lw1 = torch.nn.Parameter(torch.ones(self.n_groups))

		torch.nn.init.trunc_normal_(self.iconv.weight, std=0.02)
		torch.nn.init.trunc_normal_(self.fconv.weight, std=0.02)
		torch.nn.init.trunc_normal_(self.linear.weight, std=0.02)

		torch.nn.init.zeros_(self.iconv.bias)
		torch.nn.init.zeros_(self.fconv.bias)
		torch.nn.init.zeros_(self.linear.bias)

		summary_columns = ["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Training Count MSE",
			"Validation MNLL", "Validation Profile Pearson",
			"Validation Count Pearson", "Validation Count MSE", "Saved?"]
		self.logger = Logger(summary_columns, verbose=verbose)

		# Detail logger: same columns as the summary plus one
		# ProfilePearson_g{i} and one CountPearson_g{i} per signal
		# group, so multi-modal runs can be analyzed per-modality
		# offline. Stays out of stdout — at hundreds of groups the
		# detail rows would blow out the terminal — and lands at
		# `{name}.detailed.log` on disk in parallel with the summary
		# `{name}.log`.
		per_group_columns = []
		for i in range(self.n_groups):
			per_group_columns.append("ProfilePearson_g{}".format(i))
		for i in range(self.n_groups):
			per_group_columns.append("CountPearson_g{}".format(i))
		self.detail_logger = Logger(summary_columns + per_group_columns,
			verbose=False)

		# After load_state_dict completes (and the full recursion has
		# updated every nested CheriBlock's Linear weights), refresh
		# each block's eval-time bf16 weight cache if we're in eval
		# mode. This makes `model.eval(); model.load_state_dict(...)`
		# work as expected for the inference megakernel path.
		def _refresh_block_caches(module, _keys):
			for block in module.blocks:
				if not block.training:
					block.train(False)
		self.register_load_state_dict_post_hook(_refresh_block_caches)

		# Compile is opt-out via the `compile` kwarg, and the compile mode
		# is configurable via `compile_mode` (passed through to
		# `torch.compile(mode=...)`). Both are runtime knobs, not
		# architecture, so neither goes through `_init_kwargs` and they
		# are not persisted in checkpoints. `forward` (defined below) is
		# a thin trampoline that calls `self._forward_fn`, so the choice
		# picked here also governs subclasses that do
		# `super().forward(...)`.
		self._compile      = bool(compile)
		self._compile_mode = compile_mode
		self._forward_fn = (
			torch.compile(self._forward_impl, mode=self._compile_mode)
			if self._compile else self._forward_impl
		)

	def _init_kwargs(self):
		"""Return the kwargs needed to reconstruct this model."""
		return {
			'n_filters': self.n_filters,
			'n_layers': self.n_layers,
			'signal_groups': list(self.signal_groups),
			'n_control_tracks': self.n_control_tracks,
			'expansion': self.expansion,
			'residual_scale': self.residual_scale,
			'name': self.name,
			'trimming': self.trimming,
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
	def load(cls, path, device='cpu', compile=True,
		compile_mode='max-autotune'):
		"""Load a model previously saved with :meth:`save`.

		Parameters
		----------
		path: str
			The checkpoint file path.

		device: str or torch.device, optional
			Device to map the parameters onto. Default is ``'cpu'``.

		compile: bool, optional
			Whether the loaded model should wrap its forward in
			``torch.compile``. Default is ``True`` (matches pre-2026-05
			behavior). Pass ``False`` to get an eager forward — useful
			for scripts that hit the cudagraph cache-overwrite error or
			that need to debug / trace the model.

		compile_mode: str, optional
			The ``mode`` passed through to ``torch.compile`` when
			``compile=True``. Default is ``'max-autotune'``. Common
			alternatives:

			- ``'max-autotune-no-cudagraphs'`` — same kernel autotuning,
			  but disables CUDA graph capture. The safe choice if you
			  hit a cudagraph error but still want autotuned kernels.
			- ``'reduce-overhead'`` — lighter compile, smaller speedup,
			  no autotune sweep.

			Ignored when ``compile=False``.

		Returns
		-------
		model: Cherimoya
			The reconstructed model, placed on ``device``.
		"""

		payload = torch.load(path, map_location=device, weights_only=True)
		# The compile / compile_mode kwargs are runtime knobs that
		# `_init_kwargs` intentionally excludes from the saved config,
		# so they're always supplied here rather than read from the
		# checkpoint.
		model = cls(**payload['config'], compile=compile,
			compile_mode=compile_mode)
		model.load_state_dict(payload['state_dict'])
		return model.to(device)

	def forward(self, X, X_ctl=None):
		"""A forward pass of the model.

		Dispatches to ``self._forward_fn`` (which is either the compiled or
		eager forward, set in ``__init__`` according to the ``compile``
		kwarg). Kept as a class-level method so that subclasses overriding
		``forward`` can still call ``super().forward(...)``.
		"""

		return self._forward_fn(X, X_ctl)

	def _forward_impl(self, X, X_ctl=None):
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

		X_ctl: torch.tensor or None, shape=(batch_size, n_control_tracks, length)
			A value representing the signal of the control at each position in
			the sequence. If no controls, pass in None. Default is None.

		Returns
		-------
		y_profile: torch.tensor, shape=(batch_size, sum(signal_groups), out_length)
			Per-channel profile logits trimmed to the output length —
			one channel per signal channel across all groups.

		y_counts: torch.tensor, shape=(batch_size, len(signal_groups))
			Per-group log-count predictions — one prediction per signal
			group, so a stranded ``(+, -)`` pair contributes a single
			shared count.
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
			# Called inline rather than through an nn.Module. A module here
			# would give DeepLIFT a node to hook, but it would have nothing
			# to do: this input depends only on the control tracks, which
			# attribution holds fixed between a sequence and its references,
			# so the difference across the node is exactly zero and the
			# rescale rule has no multiplier to correct.
			X = torch.cat([X, torch.log(X_ctl+1)], dim=-1)

		y_counts = self.linear(X)
		return y_profile, y_counts

	def fit(self, training_data, muon_optimizer, adam_optimizer, lw_optimizer,
		muon_scheduler, adam_scheduler, lw_scheduler, X_valid, X_ctl_valid,
		y_valid, max_epochs=50, batch_size=64, dtype='float32', device='cuda',
		early_stopping=None):
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
			should be the head/tail layers, the bias terms, the per-block
			``conv_weight`` parameter, and any other parameters that are not 2D
			matrices routed to Muon.

		lw_optimizer: torch.optim.Optimizer
			An optimizer for the Kendall uncertainty weights (``lw0``,
			``lw1``). Typically SGD with momentum.

		muon_scheduler: torch.optim.lr_scheduler
			The scheduler to use for the Muon optimizer. This should likely be a cosine
			decay with a warmup phase.

		adam_scheduler: torch.optim.lr_scheduler
			The scheduler to use for the Adam/W optimizer. This should likely be the
			same cosine decay with a warmup phase used for the Muon optimizer.

		lw_scheduler: torch.optim.lr_scheduler
			The scheduler to use for the ``lw`` optimizer. Typically a
			linear warmup followed by a constant rate (no decay).

		X_valid: torch.tensor, shape=(n, 4, length)
			A block of sequences to validate on at the end of each epoch.

		X_ctl_valid: torch.tensor or None, shape=(n, n_control_tracks, length)
			A block of control sequences to use for making the validation set
			predictions at the end of each epoch. If n_control_tracks is None, pass in
			None. Default is None.

		y_valid: torch.tensor or None, shape=(n, sum(signal_groups), output_length)
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
		self.detail_logger.start()

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
				lw_optimizer.zero_grad()
				self.train()

				# Make one training step
				with torch.autocast(device_type=device, dtype=dtype):
					y_hat_logits, y_hat_logcounts = self(X, X_ctl)

				profile_loss, count_loss = _mixture_loss(y,
					y_hat_logits.float(), y_hat_logcounts.float(),
					signal_groups=self.signal_groups)


				w0 = (1.0 / (2.0 * self.lw0 ** 2))
				w1 = (1.0 / (2.0 * self.lw1 ** 2))
				loss = (w0 * profile_loss).sum() + (w1 * count_loss).sum()

				if self.lw0.requires_grad == True:
					loss += (torch.log(self.lw0) ** 2).sum()
					loss += (torch.log(self.lw1) ** 2).sum()

				loss.backward()

				muon_optimizer.step()
				adam_optimizer.step()
				lw_optimizer.step()

				muon_scheduler.step()
				adam_scheduler.step()
				lw_scheduler.step()

				ema.update(self)

				iteration += 1

			train_time = time.time() - tic

			if self.lw0.requires_grad == True and torch.abs(self.lw0.grad).mean() < 1:
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
					y_hat_logits, y_hat_logcounts,
					signal_groups=self.signal_groups)

				measures = calculate_performance_measures(y_hat_logits,
					y_valid, y_hat_logcounts,
					measures=['profile_pearson', 'count_pearson'],
					signal_groups=self.signal_groups)

				valid_profile_corr = numpy.nan_to_num(measures['profile_pearson'])
				valid_count_per_group = numpy.nan_to_num(measures['count_pearson'])
				valid_count_corr = valid_count_per_group.mean()
				valid_time = time.time() - tic

				# Per-group profile Pearson. The raw
				# `measures['profile_pearson']` is shape
				# (n_loci, sum(signal_groups)); for each group, average
				# over its channels and the locus dim so each modality
				# contributes one number -- a per-group summary like the
				# count Pearson. The Pearson metric stays per-channel and
				# is scale-invariant, so this channel-average is unaffected
				# by the loss's joint per-group normalization.
				per_group_profile_corr = []
				offset = 0
				for g in self.signal_groups:
					chunk = valid_profile_corr[:, offset:offset+g]
					per_group_profile_corr.append(float(chunk.mean()))
					offset += g
				valid_profile_corr_mean = float(numpy.mean(
					per_group_profile_corr))

				summary_row = [epoch,
					iteration,
					train_time,
					valid_time,
					profile_loss.mean().item(),
					count_loss.mean().item(),
					valid_profile_loss.mean().item(),
					valid_profile_corr_mean,
					valid_count_corr,
					valid_count_loss.mean().item(),
					(valid_count_corr > best_corr).item()]
				self.logger.add(summary_row)
				self.detail_logger.add(summary_row
					+ per_group_profile_corr
					+ [float(v) for v in valid_count_per_group.tolist()])

				self.logger.save("{}.log".format(self.name))
				self.detail_logger.save("{}.detailed.log".format(self.name))

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
