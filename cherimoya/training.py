# training.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Training a Cherimoya model with PyTorch Lightning.

:class:`CherimoyaModule` holds one training step, the validation pass and
the checkpointing hooks, and :func:`fit` builds the `Trainer` around it.
On one device the step does what the pre-Lightning ``Cherimoya.fit`` loop
did, in the same order; on several devices the global batch is split
across them by :class:`~cherimoya.io.ShardedEpochSampler` and DDP averages
the gradients, so every step sees the examples one device would have.
"""

import copy
import os
import time
import warnings

import numpy
import torch

import lightning
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.io import TorchCheckpointIO

from torch.optim import Muon
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from .cherimoya import EMA
from .cherimoya import _group_depths
from .io import ShardedEpochSampler
from .losses import _mixture_loss
from .performance import calculate_performance_measures


_PRECISION = {
	'float32': '32-true',
	'bfloat16': 'bf16-mixed',
	'float16': '16-mixed',
}

_INPUT_DTYPE = {
	'bf16-mixed': torch.bfloat16,
	'16-mixed': torch.float16,
}

# Warnings Lightning raises about a layout that is deliberate here: the
# checkpoint and the metrics file are written into a directory that
# usually holds other files, the step count is small next to Lightning's
# logging interval, and the worker count is the one the user chose.
_QUIET = [
	"Checkpoint directory .* exists and is not empty",
	"Experiment logs directory .* exists and is not empty",
	"The number of training batches .* is smaller than the logging interval",
	".*does not have many workers.*",
]


def _split_parameters(model):
	"""Route each trainable parameter to one of the three optimizers.

	Muon takes 2D projection weights inside Cheri Blocks
	(linear1/linear2.weight). ``conv_weight`` is 2D but lives on the
	depth-wise dilated path, not a projection matmul, and is routed to
	AdamW -- note the test is a substring, so it matches whether the
	parameter sits directly on the block or on its ``conv`` submodule.
	``lw0`` / ``lw1`` are the Kendall loss-balancing weights and are
	routed by exact name to SGD. Everything else goes to AdamW.

	Every parameter lands in exactly one bucket, so the three lists
	partition ``model.named_parameters()``.

	Parameters
	----------
	model: torch.nn.Module
			The model whose parameters are being routed.

	Returns
	-------
	muon_params: list
			2D projection weights, for the Muon optimizer.

	adam_params: list
			Everything not claimed by the other two, for AdamW.

	lw_params: list
			The Kendall loss-balancing weights, for SGD.
	"""

	muon_params = []
	adam_params = []
	lw_params = []
	for name, p in model.named_parameters():
		if name in ("lw0", "lw1"):
			lw_params.append(p)
		elif (
			p.ndim == 2
			and "weight" in name
			and name != "linear.weight"
			and "conv_weight" not in name
		):
			muon_params.append(p)
		else:
			adam_params.append(p)

	return muon_params, adam_params, lw_params


def _shard_bounds(n, rank, world_size):
	"""The `[start, stop)` rows of `n` that `rank` validates on."""

	sizes = [n // world_size + (r < n % world_size) for r in range(world_size)]
	start = sum(sizes[:rank])
	return start, start + sizes[rank]


class _ContiguousShardSampler(torch.utils.data.Sampler):
	"""This rank's contiguous block of the validation set, with no padding.

	Lightning's own `DistributedSampler` pads the shards to equal length by
	repeating examples, which would count them twice in the metrics.
	"""

	def __init__(self, n, rank, world_size):
		self.start, self.stop = _shard_bounds(n, rank, world_size)

	def __len__(self):
		return self.stop - self.start

	def __iter__(self):
		return iter(range(self.start, self.stop))


class _CherimoyaCheckpointIO(TorchCheckpointIO):
	"""Write checkpoints in the `Cherimoya.save` format.

	`CherimoyaModule.on_save_checkpoint` puts the config and the EMA state
	dict under the `cherimoya` key; only that is written, so the file
	loads with `Cherimoya.load` and nothing else in Lightning's checkpoint
	reaches disk.
	"""

	def save_checkpoint(self, checkpoint, path, storage_options=None):
		torch.save(checkpoint['cherimoya'], path)


class _MetricsLogger(CSVLogger):
	"""A `CSVLogger` that writes one flat file, `{name}.metrics.csv`.

	`CSVLogger` fixes the file name to `metrics.csv` inside a versioned
	directory; this keeps its writer and points it at the given path, next
	to the checkpoint. The file is rewritten after every epoch.
	"""

	def __init__(self, path):
		super().__init__(save_dir=os.path.dirname(path) or ".", name="",
			version="", flush_logs_every_n_steps=1)
		self._metrics_path = path

	@property
	def experiment(self):
		experiment = super().experiment
		if self._experiment is not None:
			self._experiment.metrics_file_path = self._metrics_path
		return experiment


class CherimoyaModule(lightning.LightningModule):
	"""A Lightning module that trains a Cherimoya model.

	Each training step zeroes the three optimizers, runs the model, weights
	the per-group profile and count losses, and steps the optimizers, the
	schedulers and the EMA, in that order. Optimization is manual because
	the parameters are split across three optimizers (see
	:func:`_split_parameters`).

	At the end of every epoch the EMA shadow weights are swapped in and
	validated. Each device validates a contiguous block of the validation
	set, and the per-locus statistics are gathered so that the metrics are
	computed over the whole set, exactly as on one device.

	Saved checkpoints hold the EMA shadow in the :meth:`Cherimoya.save`
	format.

	Parameters
	----------
	model: Cherimoya
		The model to train.

	training_data: PeakNegativeSampler
		The training dataset. It must accept `(epoch, index)` pairs, which
		is how :class:`~cherimoya.io.ShardedEpochSampler` addresses it.

	X_valid: torch.tensor, shape=(n, 4, length)
		Validation sequences.

	y_valid: torch.tensor, shape=(n, sum(signal_groups), output_length)
		Validation signal.

	X_ctl_valid: torch.tensor or None, shape=(n, n_control_tracks, length)
		Validation controls, or None if the model takes none. Default is
		None.

	batch_size: int, optional
		The global batch size, split evenly across devices. Also the batch
		size each device validates with. Default is 64.

	num_workers: int, optional
		Data-loading workers per device. Default is 1.

	n_warmup_steps: int, optional
		Steps of linear learning-rate warmup, from 1% of each rate. Default
		is 0.

	n_decay_steps: int, optional
		Steps of cosine decay for the Muon and AdamW rates after warmup.
		The `lw` rate is held constant instead. Default is 1.

	muon_lr, muon_wd: float, optional
		Muon learning rate and weight decay. Defaults are 0.025 and 0.03.

	adam_lr, adam_wd: float, optional
		AdamW learning rate and weight decay. Defaults are 0.001 and 0.

	lw_lr, lw_wd, lw_momentum: float, optional
		SGD learning rate, weight decay and momentum for `lw0` and `lw1`.
		Defaults are 0.001, 0 and 0.9.

	loss_weights: tuple or None, optional
		Fixed `(profile, count)` weights replacing the learned Kendall
		weights. The profile term is first divided by each group's
		batch-mean read depth, and `lw0` / `lw1` stop receiving gradient.
		If None, the Kendall weights are learned until the mean magnitude
		of `lw0`'s gradient at the end of an epoch drops below 1, and are
		held fixed after that. Default is None.

	ema_decay: float, optional
		Decay of the EMA of the weights. Default is 0.999.
	"""

	def __init__(self, model, training_data, X_valid, y_valid,
		X_ctl_valid=None, batch_size=64, num_workers=1, n_warmup_steps=0,
		n_decay_steps=1, muon_lr=0.025, muon_wd=0.03, adam_lr=0.001,
		adam_wd=0.0, lw_lr=0.001, lw_wd=0.0, lw_momentum=0.9,
		loss_weights=None, ema_decay=0.999):
		super().__init__()
		self.automatic_optimization = False

		self.model = model
		self.training_data = training_data
		self.X_valid = X_valid
		self.y_valid = y_valid
		self.X_ctl_valid = X_ctl_valid

		self.batch_size = batch_size
		self.num_workers = num_workers
		self.n_warmup_steps = n_warmup_steps
		self.n_decay_steps = n_decay_steps
		self.muon_lr, self.muon_wd = muon_lr, muon_wd
		self.adam_lr, self.adam_wd = adam_lr, adam_wd
		self.lw_lr, self.lw_wd, self.lw_momentum = lw_lr, lw_wd, lw_momentum
		self.loss_weights = loss_weights
		self.ema_decay = ema_decay

		# Fixed weights take `lw0`/`lw1` out of the loss entirely. Turning
		# off their gradient here, before DDP wraps the model, is what lets
		# DDP leave them out of its gradient reduction.
		if loss_weights is not None:
			self.model.lw0.requires_grad = False
			self.model.lw1.requires_grad = False
		self._lw_frozen = loss_weights is not None

		self.ema = None
		self._lw0_grad = None
		self._iteration = 0
		self._best_valid = float("-inf")
		self._valid_outputs = []
		self._epoch_row = {}

	def train_dataloader(self):
		sampler = ShardedEpochSampler(len(self.training_data), self.batch_size,
			rank=self.global_rank, world_size=self.trainer.world_size)

		return torch.utils.data.DataLoader(self.training_data, sampler=sampler,
			batch_size=sampler.local_batch_size, num_workers=self.num_workers,
			pin_memory=self.device.type != "cpu",
			persistent_workers=self.num_workers > 0)

	def val_dataloader(self):
		tensors = [self.X_valid]
		if self.X_ctl_valid is not None:
			tensors.append(self.X_ctl_valid)

		dataset = torch.utils.data.TensorDataset(*tensors)
		sampler = _ContiguousShardSampler(len(dataset), self.global_rank,
			self.trainer.world_size)
		return torch.utils.data.DataLoader(dataset, sampler=sampler,
			batch_size=self.batch_size)

	def configure_optimizers(self):
		muon_params, adam_params, lw_params = _split_parameters(self.model)

		muon = Muon(muon_params, lr=self.muon_lr, weight_decay=self.muon_wd)
		adam = torch.optim.AdamW(adam_params, lr=self.adam_lr,
			weight_decay=self.adam_wd)
		lw = torch.optim.SGD(lw_params, lr=self.lw_lr, weight_decay=self.lw_wd,
			momentum=self.lw_momentum)

		def schedule(optimizer, after):
			warmup = LinearLR(optimizer, start_factor=0.01,
				total_iters=self.n_warmup_steps)
			return SequentialLR(optimizer, schedulers=[warmup, after(optimizer)],
				milestones=[self.n_warmup_steps])

		def cosine(optimizer):
			return CosineAnnealingLR(optimizer, T_max=self.n_decay_steps,
				eta_min=1e-5)

		# The Kendall weights are warmed up but not decayed.
		def constant(optimizer):
			return ConstantLR(optimizer, factor=1.0, total_iters=1)

		schedulers = [schedule(muon, cosine), schedule(adam, cosine),
			schedule(lw, constant)]

		return [muon, adam, lw], schedulers

	def on_fit_start(self):
		self.ema = EMA(self.model, decay=self.ema_decay)

	def _loss(self, y, profile_loss, count_loss):
		"""The scalar training loss from the per-group loss terms."""

		if self.loss_weights is not None:
			w0, w1 = self.loss_weights

			# The depths divide the loss, so a device's share of the batch
			# must use the depths of the whole global batch or the averaged
			# gradient would not be the one-device gradient. Every device
			# holds the same number of examples, so the mean of their means
			# is the global mean.
			reduce = None
			if self.trainer.world_size > 1:
				reduce = lambda d: self.all_gather(d).mean(dim=0)

			depths = _group_depths(y, self.model.signal_groups, reduce=reduce)
			return (w0 * profile_loss / depths).sum() + (w1 * count_loss).sum()

		lw0, lw1 = self.model.lw0, self.model.lw1
		if self._lw_frozen:
			lw0, lw1 = lw0.detach(), lw1.detach()

		loss = ((1.0 / (2.0 * lw0 ** 2)) * profile_loss).sum()
		loss = loss + ((1.0 / (2.0 * lw1 ** 2)) * count_loss).sum()

		if self._lw_frozen:
			# DDP expects a gradient for every parameter it tracks, so
			# frozen weights stay in the graph with a gradient of zero.
			return loss + 0.0 * (self.model.lw0.sum() + self.model.lw1.sum())

		return loss + (torch.log(lw0) ** 2).sum() + (torch.log(lw1) ** 2).sum()

	def training_step(self, batch, batch_idx):
		X, y = batch[0].float(), batch[-2]
		X_ctl = batch[1] if len(batch) == 4 else None

		muon, adam, lw = self.optimizers()
		for optimizer in (muon, adam, lw):
			optimizer.zero_grad()

		y_hat_logits, y_hat_logcounts = self.model(X, X_ctl)
		profile_loss, count_loss = _mixture_loss(y, y_hat_logits.float(),
			y_hat_logcounts.float(), signal_groups=self.model.signal_groups)

		self.manual_backward(self._loss(y, profile_loss, count_loss))

		# Lightning clears the gradients before validating, so the epoch-end
		# check on the Kendall weights reads the last step's value from here.
		if not self._lw_frozen:
			self._lw0_grad = self.model.lw0.grad.abs().mean().detach()

		# Lightning runs the whole step under the precision plugin's
		# autocast; the optimizers keep the precision they choose
		# themselves, as they did outside it.
		with torch.autocast(device_type=self.device.type, enabled=False):
			muon.step()
			adam.step()
			if not self._lw_frozen:
				lw.step()

		for scheduler in self.lr_schedulers():
			scheduler.step()

		self.ema.update(self.model)
		self._iteration += 1

		self.log("train_profile_mnll", profile_loss.mean().detach(),
			on_step=False, on_epoch=True, sync_dist=True, batch_size=len(X))
		self.log("train_count_mse", count_loss.mean().detach(),
			on_step=False, on_epoch=True, sync_dist=True, batch_size=len(X))

	def on_train_epoch_start(self):
		self._epoch_tic = time.time()

	def on_validation_epoch_start(self):
		self._epoch_row['train_time'] = time.time() - self._epoch_tic
		self._valid_tic = time.time()
		self._valid_outputs = []
		self.ema.apply_shadow(self.model)

	def validation_step(self, batch, batch_idx):
		# Validation inputs, controls included, are cast to the training
		# dtype, as `tangermeme.predict` did in the loop this replaces.
		dtype = _INPUT_DTYPE.get(self.trainer.precision, torch.float32)
		X = batch[0].to(dtype)
		X_ctl = batch[1].to(dtype) if len(batch) == 2 else None

		y_hat_logits, y_hat_logcounts = self.model(X, X_ctl)
		self._valid_outputs.append((y_hat_logits.cpu(), y_hat_logcounts.cpu()))

	def _gather_rows(self, x):
		"""Every rank's rows of `x`, concatenated in rank order.

		`all_gather` needs equal shapes, so each rank pads its block to the
		largest block's length and the padding is dropped afterwards.
		"""

		world_size = self.trainer.world_size
		if world_size == 1:
			return x

		n = len(self.X_valid)
		bounds = [_shard_bounds(n, r, world_size) for r in range(world_size)]
		longest = max(stop - start for start, stop in bounds)

		padded = torch.zeros(longest, *x.shape[1:], dtype=x.dtype)
		padded[:len(x)] = x
		gathered = self.all_gather(padded.to(self.device)).cpu()

		return torch.cat([gathered[r, :stop - start]
			for r, (start, stop) in enumerate(bounds)])

	def on_validation_epoch_end(self):
		signal_groups = self.model.signal_groups
		start, stop = _shard_bounds(len(self.X_valid), self.global_rank,
			self.trainer.world_size)

		y = self.y_valid[start:stop]
		y_hat_logits = torch.cat([logits for logits, _ in self._valid_outputs])
		y_hat_logcounts = torch.cat([counts for _, counts in self._valid_outputs])
		self._valid_outputs = []

		profile_loss, count_loss = _mixture_loss(y, y_hat_logits,
			y_hat_logcounts, signal_groups=signal_groups)
		profile_pearson = calculate_performance_measures(y_hat_logits, y,
			y_hat_logcounts, measures=['profile_pearson'],
			signal_groups=signal_groups)['profile_pearson']

		# The losses are means over this rank's rows, so the mean over the
		# whole set weights each rank by its row count.
		if self.trainer.world_size > 1:
			losses = torch.stack([profile_loss, count_loss]).to(self.device)
			losses = self.all_gather(losses * (stop - start)).sum(dim=0)
			profile_loss, count_loss = (losses / len(self.X_valid)).cpu()

		profile_pearson = self._gather_rows(profile_pearson)
		y_hat_logcounts = self._gather_rows(y_hat_logcounts)
		observed = self._gather_rows(y.sum(dim=-1))

		# The count Pearson needs only per-channel totals, so pass those as
		# a profile of length one.
		count_pearson = calculate_performance_measures(
			torch.zeros(*observed.shape, 1), observed.unsqueeze(-1),
			y_hat_logcounts, measures=['count_pearson'],
			signal_groups=signal_groups)['count_pearson']

		profile_pearson = numpy.nan_to_num(profile_pearson)
		count_pearson = numpy.nan_to_num(count_pearson)

		# Each group's profile Pearson averages over its channels and loci.
		per_group_profile, lo = [], 0
		for width in signal_groups:
			per_group_profile.append(float(profile_pearson[:, lo:lo+width].mean()))
			lo += width

		valid_count_pearson = count_pearson.mean()

		row = self._epoch_row
		row['valid_profile_mnll'] = profile_loss.mean().item()
		row['valid_count_mse'] = count_loss.mean().item()
		row['valid_profile_pearson'] = float(numpy.mean(per_group_profile))
		row['valid_count_pearson'] = torch.tensor(valid_count_pearson)
		row['saved'] = float(valid_count_pearson > self._best_valid)
		for i, value in enumerate(per_group_profile):
			row['valid_profile_pearson_g{}'.format(i)] = value
		for i, value in enumerate(count_pearson):
			row['valid_count_pearson_g{}'.format(i)] = float(value)

		self._best_valid = max(self._best_valid, valid_count_pearson)
		row['valid_time'] = time.time() - self._valid_tic

		self.ema.restore(self.model)

	def on_train_epoch_end(self):
		# Validation has already run for this epoch; logging its results
		# here, with the training averages, puts one row per epoch in the
		# metrics file, and is where the checkpoint and early-stopping
		# callbacks read `valid_count_pearson`.
		self._epoch_row['iteration'] = float(self._iteration)
		for key, value in self._epoch_row.items():
			self.log(key, value, on_step=False, on_epoch=True)
		self._epoch_row = {}

		# The gradient has been averaged across devices, so every rank
		# makes the same call.
		if not self._lw_frozen and self._lw0_grad < 1:
			self._lw_frozen = True

	def on_train_end(self):
		self.ema.apply_shadow(self.model)

	def on_save_checkpoint(self, checkpoint):
		# The EMA weights are what gets saved, in exactly the object
		# `Cherimoya.save` writes. The copy is deep because the swap back
		# overwrites the parameters in place, and a deep copy of the state
		# dict keeps its `_metadata`, which `load_state_dict` reads.
		swap = self.ema is not None and not self.ema._backup
		if swap:
			self.ema.apply_shadow(self.model)

		checkpoint['cherimoya'] = copy.deepcopy(self.model._checkpoint())

		if swap:
			self.ema.restore(self.model)


def fit(model, training_data, X_valid, y_valid, X_ctl_valid=None,
	max_epochs=50, early_stopping=None, dtype='float32', accelerator='auto',
	devices=1, verbose=False, **kwargs):
	"""Train a Cherimoya model and write its checkpoints and metrics.

	Three files are written next to ``model.name``: ``{name}.torch``, the
	EMA weights from the epoch with the highest mean validation count
	Pearson; ``{name}.final.torch``, the EMA weights at the end of
	training; and ``{name}.metrics.csv``, one row per epoch of training
	and validation measures, including one profile and one count Pearson
	column per signal group.

	Parameters
	----------
	model: Cherimoya
		The model to train.

	training_data: PeakNegativeSampler
		The training dataset.

	X_valid, y_valid, X_ctl_valid: torch.tensor
		The validation set; see :class:`CherimoyaModule`.

	max_epochs: int, optional
		The number of passes over the training data. Default is 50.

	early_stopping: int or None, optional
		Stop after this many epochs without an improvement in the
		validation count Pearson. If None, train for `max_epochs`. Default
		is None.

	dtype: str, optional
		``'float32'``, ``'bfloat16'`` or ``'float16'``. The two half
		precisions run the forward pass under autocast, and ``'float16'``
		also scales the loss. Default is ``'float32'``.

	accelerator: str, optional
		The Lightning accelerator, e.g. ``'gpu'``, ``'cpu'`` or
		``'auto'``. Default is ``'auto'``.

	devices: int, optional
		The number of devices. More than one trains with DDP, with the
		global batch split evenly across them. -1 uses every visible
		device. Default is 1.

	verbose: bool, optional
		Whether to show Lightning's progress bar and messages. Default is
		False.

	**kwargs
		Passed to :class:`CherimoyaModule`.

	Returns
	-------
	trainer: lightning.Trainer
		The trainer after fitting. ``trainer.lightning_module.model`` holds
		the EMA weights, and ``trainer.checkpoint_callback.best_model_score``
		the best validation count Pearson.
	"""

	if dtype not in _PRECISION:
		raise ValueError("dtype must be one of {}, got {!r}".format(
			list(_PRECISION), dtype))

	batch_size = kwargs.get('batch_size', 64)
	if len(training_data) < batch_size:
		raise ValueError("The training set has {} examples, fewer than one "
			"batch of {}, so no training step could be taken.".format(
				len(training_data), batch_size))

	module = CherimoyaModule(model, training_data, X_valid, y_valid,
		X_ctl_valid=X_ctl_valid, **kwargs)

	name = model.name
	directory = os.path.dirname(name) or os.getcwd()

	checkpoint = ModelCheckpoint(dirpath=directory,
		filename=os.path.basename(name), monitor='valid_count_pearson',
		mode='max', save_top_k=1, save_weights_only=True,
		save_on_train_epoch_end=True, enable_version_counter=False,
		auto_insert_metric_name=False)
	checkpoint.FILE_EXTENSION = ".torch"

	callbacks = [checkpoint]
	if early_stopping is not None:
		callbacks.append(EarlyStopping(monitor='valid_count_pearson',
			mode='max', patience=early_stopping,
			check_on_train_epoch_end=True))

	trainer = lightning.Trainer(accelerator=accelerator, devices=devices,
		strategy='ddp' if devices != 1 else 'auto',
		precision=_PRECISION[dtype], max_epochs=max_epochs,
		logger=_MetricsLogger("{}.metrics.csv".format(name)),
		callbacks=callbacks, plugins=[_CherimoyaCheckpointIO()],
		benchmark=True, inference_mode=False, num_sanity_val_steps=0,
		use_distributed_sampler=False, enable_progress_bar=verbose,
		enable_model_summary=verbose, default_root_dir=directory)

	with warnings.catch_warnings():
		for message in _QUIET:
			warnings.filterwarnings("ignore", message=message)

		trainer.fit(module)
		trainer.save_checkpoint("{}.final.torch".format(name),
			weights_only=True)

	return trainer
