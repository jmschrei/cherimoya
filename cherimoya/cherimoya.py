# cherimoya.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
An implementation of the Cherimoya deep learning model, a compact
architecture for predicting genomic modalities from sequence alone.
"""

import torch

from .cheri import CheriBlock
from .io import _validate_signal_groups


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
				# Not `p.data.copy_`: a `.data` write is the one that
				# does not advance the version counter, which is how
				# `CheriBlock` spots a stale eval cache.
				p.copy_(self.shadow[name])

	@torch.no_grad()
	def restore(self, model):
		"""Put the original training weights back into the model."""

		for name, p in model.named_parameters():
			if name in self._backup:
				p.copy_(self._backup[name])
		self._backup = {}


def _group_depths(y, signal_groups):
	"""Batch-mean observed counts for each signal group.

	The profile MNLL is a sum of per-read log-likelihoods, so it scales with
	read depth; dividing each group's term by its own depth makes the
	weighted term depth-invariant. The channel axis of ``y`` is ordered by
	group, so a group's depth is a sum over its own channels only.

	Summing over every group at once instead would rescale all groups by the
	same number and leave their weights relative to each other untouched,
	which is not a normalization across groups at all.


	Parameters
	----------
	y: torch.tensor, shape=(batch_size, n_channels, length)
		The observed counts.

	signal_groups: list of int
		Channels belonging to each group, in order.


	Returns
	-------
	depths: torch.tensor, shape=(n_groups,)
		Batch-mean counts per group, floored at 1 so the division is safe on
		a batch where a group has no reads.
	"""

	depths, lo = [], 0
	for width in signal_groups:
		depths.append(y[:, lo:lo + width].sum(dim=(1, 2)).float().mean())
		lo += width

	return torch.stack(depths).clamp(min=1.0)


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
		Unused. It was the switch for the training log's stdout output,
		which moved to :func:`cherimoya.training.fit`, and is kept because
		every saved checkpoint's config passes it. Default is True.

	random_state: int or None, optional
		Seed for the weight initialization. Every parameter in the model
		is either overwritten by one of the ``trunc_normal_`` calls
		below, zeroed, or set to one, so a given seed fully determines
		the starting weights regardless of the global RNG state. The
		seed is *not* stored in the checkpoint: it describes how a model
		was initialized, not its architecture, and a loaded model takes
		its weights from the state dict rather than from init. If None,
		the global RNG is used and the initialization is not
		reproducible. Default is None.
	"""

	def __init__(self, n_filters=128, n_layers=9, signal_groups=None,
		n_control_tracks=0, expansion=2, residual_scale=0.15, name=None,
		trimming=None, verbose=True, compile=True,
		compile_mode='max-autotune', random_state=None):
		super(Cherimoya, self).__init__()

		# One generator feeds every weight draw in the model, including
		# those inside the blocks. A local generator rather than
		# `torch.manual_seed` so that asking for a reproducible model
		# does not reseed the caller's global RNG stream. Construction
		# still advances that stream -- Conv1d/Linear run their own
		# default init first -- but every value it produces is then
		# overwritten below.
		generator = (None if random_state is None
			else torch.Generator().manual_seed(random_state))

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
				residual_scale=residual_scale, generator=generator)
			for i in range(self.n_layers)
		])

		self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks,
			self.n_outputs, kernel_size=75, padding=37)

		n_count_control = 1 if n_control_tracks > 0 else 0
		self.linear = torch.nn.Linear(n_filters+n_count_control, self.n_groups)

		self.lw0 = torch.nn.Parameter(torch.ones(self.n_groups))
		self.lw1 = torch.nn.Parameter(torch.ones(self.n_groups))

		torch.nn.init.trunc_normal_(self.iconv.weight, std=0.02,
			generator=generator)
		torch.nn.init.trunc_normal_(self.fconv.weight, std=0.02,
			generator=generator)
		torch.nn.init.trunc_normal_(self.linear.weight, std=0.02,
			generator=generator)

		torch.nn.init.zeros_(self.iconv.bias)
		torch.nn.init.zeros_(self.fconv.bias)
		torch.nn.init.zeros_(self.linear.bias)

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

		torch.save(self._checkpoint(), path)

	def _checkpoint(self):
		"""The object :meth:`save` writes: the config and the state dict.

		Training writes its checkpoints from this too, so that every file
		holds the same thing however it was produced.
		"""

		return {
			'config': self._init_kwargs(),
			'state_dict': self.state_dict(),
		}

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

		Before dispatching, any block whose eval-time weight cache is
		older than its weights (e.g. after an EMA swap in eval mode) has
		the cache rebuilt. The check runs here, outside the compiled
		region, because ``CheriBlock`` skips it under ``torch.compile``.
		"""

		if not torch.compiler.is_compiling():
			for block in self.blocks:
				if (not block.training and block._eval_cache_version
					!= block._weight_versions()):
					block.train(False)

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
