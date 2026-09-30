# test_training.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import warnings

import numpy
import pandas
import pytest
import torch

from torch.optim import Muon
from torch.optim.lr_scheduler import (LinearLR, CosineAnnealingLR, ConstantLR,
	SequentialLR)

from tangermeme.predict import predict

from cherimoya import Cherimoya, EMA
from cherimoya.cherimoya import _group_depths
from cherimoya.io import PeakNegativeSampler, channel_permutation_from_groups
from cherimoya.losses import _mixture_loss
from cherimoya.performance import calculate_performance_measures
from cherimoya.training import fit, _split_parameters, _shard_bounds
from cherimoya.training import CherimoyaModule


torch.manual_seed(0)


BATCH_SIZE = 4

# The training log's column for each measure the reference loop records.
LOG_NAMES = {'train_profile_mnll': 'Training MNLL',
	'train_count_mse': 'Training Count MSE',
	'valid_profile_mnll': 'Validation MNLL',
	'valid_count_mse': 'Validation Count MSE',
	'valid_profile_pearson': 'Validation Profile Pearson',
	'valid_count_pearson': 'Validation Count Pearson'}
LOG_COLUMNS = ["Epoch", "Iteration", "Training Time", "Validation Time",
	"Training MNLL", "Training Count MSE", "Validation MNLL",
	"Validation Profile Pearson", "Validation Count Pearson",
	"Validation Count MSE", "Saved?"]
N_WARMUP_STEPS = 3
N_DECAY_STEPS = 5
LR = dict(muon_lr=0.025, muon_wd=0.03, adam_lr=1e-3, adam_wd=0.0,
	lw_lr=1e-3, lw_wd=0.0, lw_momentum=0.9)


def _model(signal_groups, n_controls=0):
	return Cherimoya(n_filters=8, n_layers=2, signal_groups=signal_groups,
		n_control_tracks=n_controls, verbose=False, compile=False,
		random_state=0)


def _data(signal_groups, counts, n_controls=0):
	"""A training sampler and a validation set for a tiny model.

	12 peaks at negative ratio 0.5 give 18 examples per epoch: four full
	batches of 4 and a partial batch of 2, which both paths must drop.
	`counts` is the largest count in the signal; small counts give a small
	MNLL and hence a small `lw0` gradient, which freezes the Kendall
	weights after the first epoch. Control tracks, when asked for, hold
	values bf16 cannot represent exactly, so that casting them shows.
	"""

	model = _model(signal_groups)
	n_out = sum(signal_groups)
	L = 46 + 2 * model.trimming + 8
	out_L = L - 2 * model.trimming

	g = torch.Generator().manual_seed(0)

	def one_hot(n):
		idx = torch.randint(0, 4, (n, L), generator=g)
		return torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float()

	def signal(n):
		return torch.randint(0, counts + 1, (n, n_out, out_L),
			generator=g).float()

	peak_sequences, peak_signals = one_hot(12), signal(12)
	negative_sequences, negative_signals = one_hot(6), signal(6)
	X_valid = one_hot(7)[:, :, 2:-2]
	y_valid = signal(7)[:, :, 2:-2]

	def controls(n):
		return (torch.rand(n, n_controls, L, generator=g) * 3 if n_controls
			else None)

	peak_controls, negative_controls = controls(12), controls(6)
	X_ctl_valid = controls(7)
	if X_ctl_valid is not None:
		X_ctl_valid = X_ctl_valid[:, :, 2:-2]

	sampler = PeakNegativeSampler(peak_sequences=peak_sequences,
		peak_signals=peak_signals, negative_sequences=negative_sequences,
		negative_signals=negative_signals, peak_controls=peak_controls,
		negative_controls=negative_controls, in_window=L - 4,
		out_window=out_L - 4, max_jitter=2, negative_ratio=0.5,
		reverse_complement=True, random_state=0,
		signal_perm=channel_permutation_from_groups(signal_groups))

	return sampler, X_valid, y_valid, X_ctl_valid


def _reference_fit(model, training_data, X_valid, y_valid, X_ctl_valid,
	max_epochs, loss_weights=None, dtype='float32'):
	"""The training and validation loop `Cherimoya.fit` ran before the move
	to Lightning, kept here as the thing the Lightning module must match.
	Checkpointing and log files are left out; everything that touches a
	weight or a logged number is as it was."""

	if loss_weights is not None:
		model.lw0.requires_grad = False
		model.lw1.requires_grad = False

	muon_params, adam_params, lw_params = _split_parameters(model)

	muon_optimizer = Muon(muon_params, lr=LR['muon_lr'],
		weight_decay=LR['muon_wd'])
	muon_scheduler = SequentialLR(muon_optimizer, schedulers=[
		LinearLR(muon_optimizer, start_factor=0.01,
			total_iters=N_WARMUP_STEPS),
		CosineAnnealingLR(muon_optimizer, T_max=N_DECAY_STEPS, eta_min=1e-5)],
		milestones=[N_WARMUP_STEPS])

	adam_optimizer = torch.optim.AdamW(adam_params, lr=LR['adam_lr'],
		weight_decay=LR['adam_wd'])
	adam_scheduler = SequentialLR(adam_optimizer, schedulers=[
		LinearLR(adam_optimizer, start_factor=0.01,
			total_iters=N_WARMUP_STEPS),
		CosineAnnealingLR(adam_optimizer, T_max=N_DECAY_STEPS, eta_min=1e-5)],
		milestones=[N_WARMUP_STEPS])

	lw_optimizer = torch.optim.SGD(lw_params, lr=LR['lw_lr'],
		weight_decay=LR['lw_wd'], momentum=LR['lw_momentum'])
	lw_scheduler = SequentialLR(lw_optimizer, schedulers=[
		LinearLR(lw_optimizer, start_factor=0.01, total_iters=N_WARMUP_STEPS),
		ConstantLR(lw_optimizer, factor=1.0, total_iters=1)],
		milestones=[N_WARMUP_STEPS])

	loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
	ema = EMA(model, decay=0.999)
	dtype = getattr(torch, dtype)
	rows, frozen_after = [], None

	for epoch in range(max_epochs):
		profile_loss_sum, count_loss_sum, n_batches = 0.0, 0.0, 0

		for data in loader:
			X, y = data[0], data[-2]
			X_ctl = data[1] if len(data) == 4 else None
			if X.shape[0] != BATCH_SIZE:
				continue

			X = X.float()
			muon_optimizer.zero_grad()
			adam_optimizer.zero_grad()
			lw_optimizer.zero_grad()
			model.train()

			with torch.autocast(device_type='cpu', dtype=dtype):
				y_hat_logits, y_hat_logcounts = model(X, X_ctl)
			profile_loss, count_loss = _mixture_loss(y, y_hat_logits.float(),
				y_hat_logcounts.float(), signal_groups=model.signal_groups)

			if loss_weights is None:
				w0 = (1.0 / (2.0 * model.lw0 ** 2))
				w1 = (1.0 / (2.0 * model.lw1 ** 2))
				loss = ((w0 * profile_loss).sum() + (w1 * count_loss).sum())

				if model.lw0.requires_grad == True:
					loss += (torch.log(model.lw0) ** 2).sum()
					loss += (torch.log(model.lw1) ** 2).sum()
			else:
				w0, w1 = loss_weights
				depths = _group_depths(y, model.signal_groups)
				loss = ((w0 * profile_loss / depths).sum()
					+ (w1 * count_loss).sum())

			loss.backward()

			muon_optimizer.step()
			adam_optimizer.step()
			lw_optimizer.step()

			muon_scheduler.step()
			adam_scheduler.step()
			lw_scheduler.step()

			ema.update(model)

			profile_loss_sum = profile_loss_sum + profile_loss.mean().detach()
			count_loss_sum = count_loss_sum + count_loss.mean().detach()
			n_batches += 1

		if (loss_weights is None and model.lw0.requires_grad == True
			and model.lw0.grad is not None
			and torch.abs(model.lw0.grad).mean() < 1):
			model.lw0.requires_grad = False
			model.lw1.requires_grad = False
			frozen_after = epoch

		with torch.no_grad():
			model.eval()
			ema.apply_shadow(model)

			y_hat_logits, y_hat_logcounts = predict(model, X_valid,
				args=None if X_ctl_valid is None else (X_ctl_valid,),
				batch_size=BATCH_SIZE, dtype=dtype, device='cpu')
			valid_profile_loss, valid_count_loss = _mixture_loss(y_valid,
				y_hat_logits, y_hat_logcounts,
				signal_groups=model.signal_groups)
			measures = calculate_performance_measures(y_hat_logits, y_valid,
				y_hat_logcounts, measures=['profile_pearson', 'count_pearson'],
				signal_groups=model.signal_groups)

			valid_profile_corr = numpy.nan_to_num(measures['profile_pearson'])
			valid_count_per_group = numpy.nan_to_num(measures['count_pearson'])

			per_group_profile_corr, offset = [], 0
			for g in model.signal_groups:
				chunk = valid_profile_corr[:, offset:offset+g]
				per_group_profile_corr.append(float(chunk.mean()))
				offset += g

			rows.append({
				'train_profile_mnll': (profile_loss_sum / n_batches).item(),
				'train_count_mse': (count_loss_sum / n_batches).item(),
				'valid_profile_mnll': valid_profile_loss.mean().item(),
				'valid_count_mse': valid_count_loss.mean().item(),
				'valid_profile_pearson': float(numpy.mean(
					per_group_profile_corr)),
				'valid_count_pearson': float(valid_count_per_group.mean()),
			})

			ema.restore(model)

	return ema, rows, frozen_after


def _lightning_fit(tmp_path, model, training_data, X_valid, y_valid,
	max_epochs, loss_weights=None, dtype='float32', X_ctl_valid=None):
	model.name = str(tmp_path / "lit")
	trainer = fit(model, training_data, X_valid, y_valid,
		X_ctl_valid=X_ctl_valid,
		max_epochs=max_epochs, dtype=dtype, accelerator='cpu', devices=1,
		batch_size=BATCH_SIZE, num_workers=0, n_warmup_steps=N_WARMUP_STEPS,
		n_decay_steps=N_DECAY_STEPS, loss_weights=loss_weights, **LR)
	return trainer.lightning_module


# These train two tiny models for three epochs, once through Lightning and
# once through the reference loop, and each takes several seconds. They are
# the check that the move to Lightning left training unchanged, which no
# smaller test can show.
@pytest.mark.parametrize(
	"signal_groups,counts,loss_weights,freezes,dtype,n_controls", [
	([1], 4, None, False, 'float32', 0),
	([1, 2], 4, None, False, 'float32', 0),
	([1], 0, None, True, 'float32', 0),
	([1, 2], 4, (1.333, 0.274), False, 'float32', 0),
	([1, 2], 4, (1.333, 0.274), False, 'float32', 2),
	([1, 2], 4, None, False, 'bfloat16', 0),
	([1, 2], 4, (1.333, 0.274), False, 'bfloat16', 2),
])
def test_lightning_matches_the_reference_loop_bitwise_on_cpu(tmp_path,
	signal_groups, counts, loss_weights, freezes, dtype, n_controls):
	max_epochs = 3

	ref_model = _model(signal_groups, n_controls)
	lit_model = _model(signal_groups, n_controls)
	for (name, p), q in zip(ref_model.named_parameters(),
			lit_model.parameters()):
		assert torch.equal(p, q), name

	ref_data, X_valid, y_valid, X_ctl_valid = _data(signal_groups, counts,
		n_controls)
	lit_data, _, _, _ = _data(signal_groups, counts, n_controls)

	ema, rows, frozen_after = _reference_fit(ref_model, ref_data, X_valid,
		y_valid, X_ctl_valid, max_epochs, loss_weights=loss_weights,
		dtype=dtype)
	module = _lightning_fit(tmp_path, lit_model, lit_data, X_valid, y_valid,
		max_epochs, loss_weights=loss_weights, dtype=dtype,
		X_ctl_valid=X_ctl_valid)

	assert (frozen_after is not None) == freezes
	assert module._lw_frozen == (freezes or loss_weights is not None)
	if loss_weights is not None:
		assert torch.equal(module.model.lw0, torch.ones(len(signal_groups)))
		assert torch.equal(module.model.lw1, torch.ones(len(signal_groups)))

	# After training the module holds the EMA weights and keeps the live
	# ones in the EMA's backup.
	assert module.ema.shadow.keys() == ema.shadow.keys()
	for name, value in ema.shadow.items():
		assert torch.equal(module.ema.shadow[name], value), name
	for name, p in ref_model.named_parameters():
		live = module.ema._backup.get(name, dict(
			module.model.named_parameters())[name])
		assert torch.equal(live, p), name

	# The training log, read back with every digit it wrote.
	log = pandas.read_csv(tmp_path / "lit.log", sep="\t",
		float_precision='round_trip')
	assert len(log) == max_epochs
	for epoch, row in enumerate(rows):
		for key, value in row.items():
			logged = numpy.float32(log[LOG_NAMES[key]].iloc[epoch])
			assert logged == numpy.float32(value), (epoch, key)
		assert log['Iteration'].iloc[epoch] == 4 * (epoch + 1)
		assert log['Epoch'].iloc[epoch] == epoch


@pytest.mark.parametrize("dtype,expected", [('float32', torch.float32),
	('bfloat16', torch.bfloat16)])
def test_validation_casts_inputs_to_the_training_dtype(tmp_path, dtype,
	expected):
	"""The loop this replaces validated through `tangermeme.predict`, which
	casts the sequences and the controls to `dtype`. Under autocast the
	cast rarely changes an output, so the bitwise test above cannot pin
	it; this checks what reaches the model."""

	training_data, X_valid, y_valid, X_ctl_valid = _data([1], 4, n_controls=2)
	model = _model([1], n_controls=2)

	seen = []
	forward = model.forward

	def spy(X, X_ctl=None):
		if not torch.is_grad_enabled():
			seen.append((X.dtype, X_ctl.dtype))
		return forward(X, X_ctl)

	model.forward = spy
	_lightning_fit(tmp_path, model, training_data, X_valid, y_valid,
		max_epochs=1, dtype=dtype, X_ctl_valid=X_ctl_valid)

	assert seen and set(seen) == {(expected, expected)}


def _assert_same_payload(a, b):
	assert a.keys() == b.keys() == {'config', 'state_dict'}
	assert a['config'] == b['config']
	assert type(a['state_dict']) is type(b['state_dict'])
	assert list(a['state_dict']) == list(b['state_dict'])
	for key in a['state_dict']:
		assert torch.equal(a['state_dict'][key], b['state_dict'][key]), key
	assert a['state_dict']._metadata == b['state_dict']._metadata


@pytest.mark.parametrize("n_controls", [0, 2])
def test_fit_writes_the_best_and_final_ema_checkpoints(tmp_path, monkeypatch,
	n_controls):
	"""Training writes its checkpoints through Lightning, and the files must
	be exactly what `Cherimoya.save` writes for the EMA weights -- the same
	config, keys, tensors and state-dict metadata -- so that nothing about
	saving or loading changes for anyone reading them."""

	# The EMA shadow at the end of every epoch, when the best checkpoint is
	# written.
	shadows = []
	epoch_end = CherimoyaModule.on_train_epoch_end

	def record(self):
		shadows.append({k: v.clone() for k, v in self.ema.shadow.items()})
		epoch_end(self)

	monkeypatch.setattr(CherimoyaModule, "on_train_epoch_end", record)

	signal_groups = [1, 2]
	training_data, X_valid, y_valid, X_ctl_valid = _data(signal_groups, 4,
		n_controls)
	module = _lightning_fit(tmp_path, _model(signal_groups, n_controls),
		training_data, X_valid, y_valid, max_epochs=3,
		X_ctl_valid=X_ctl_valid)

	# After training the model holds the EMA weights, which is what the
	# final checkpoint, and the best one if the last epoch was saved, were
	# written from.
	module.model.save(str(tmp_path / "reference.torch"))
	reference = torch.load(tmp_path / "reference.torch", weights_only=True)

	final = torch.load(tmp_path / "lit.final.torch", weights_only=True)
	_assert_same_payload(final, reference)

	log = pandas.read_csv(tmp_path / "lit.log", sep="\t")
	assert bool(log['Saved?'].iloc[0])

	best = torch.load(tmp_path / "lit.torch", weights_only=True)
	if bool(log['Saved?'].iloc[-1]):
		_assert_same_payload(best, reference)
	else:
		assert best['config'] == reference['config']
		assert best['state_dict']._metadata == reference['state_dict']._metadata

	# The best checkpoint holds the EMA weights of the last epoch that
	# improved, whichever epoch that was.
	saved = numpy.flatnonzero(log['Saved?'].astype(bool))[-1]
	best_model = Cherimoya.load(str(tmp_path / "lit.torch"), compile=False)
	parameters = dict(best_model.named_parameters())
	for name, value in shadows[saved].items():
		assert torch.equal(parameters[name], value), name

	for filename in ("lit.torch", "lit.final.torch"):
		model = Cherimoya.load(str(tmp_path / filename), compile=False)
		assert model.signal_groups == signal_groups
		assert model.n_control_tracks == n_controls


def test_fit_writes_the_training_logs_in_their_format(tmp_path):
	"""`{name}.log` and `{name}.detailed.log` keep the columns and layout
	they have always had: tab-separated, the summary columns in order, and
	the detailed log adding one ProfilePearson and one CountPearson column
	per signal group. Nothing else is written beside the checkpoints."""

	signal_groups = [1, 2]
	training_data, X_valid, y_valid, _ = _data(signal_groups, 4)
	_lightning_fit(tmp_path, _model(signal_groups), training_data, X_valid,
		y_valid, max_epochs=2)

	summary = (tmp_path / "lit.log").read_text().splitlines()
	detailed = (tmp_path / "lit.detailed.log").read_text().splitlines()
	assert summary[0].split("\t") == LOG_COLUMNS
	assert detailed[0].split("\t") == LOG_COLUMNS + ["ProfilePearson_g0",
		"ProfilePearson_g1", "CountPearson_g0", "CountPearson_g1"]
	assert len(summary) == len(detailed) == 3
	assert summary[1].split("\t")[-1] == "True"
	for line_s, line_d in zip(summary[1:], detailed[1:]):
		assert line_d.split("\t")[:len(LOG_COLUMNS)] == line_s.split("\t")

	assert sorted(p.name for p in tmp_path.iterdir()) == ["lit.detailed.log",
		"lit.final.torch", "lit.log", "lit.torch"]


def test_early_stopping_ends_the_run(tmp_path):
	training_data, X_valid, y_valid, _ = _data([1], 4)
	model = _model([1])
	model.name = str(tmp_path / "early")

	# An all-zero validation target has an undefined count Pearson, which
	# is logged as 0 every epoch, so nothing after the first epoch counts
	# as an improvement and two epochs of patience end the run at three.
	trainer = fit(model, training_data, X_valid, torch.zeros_like(y_valid),
		max_epochs=10, early_stopping=2, accelerator='cpu',
		batch_size=BATCH_SIZE, num_workers=0)

	log = pandas.read_csv(tmp_path / "early.log", sep="\t")
	assert len(log) == 3
	assert list(log['Saved?']) == [True, False, False]
	assert trainer.current_epoch == 3


@pytest.mark.parametrize("loss_weights", [None, (1.333, 0.274)])
def test_fit_does_not_warn_about_the_scheduler_order(tmp_path, loss_weights):
	"""With fixed `loss_weights` the `lw` optimizer never steps, so stepping
	its schedule made PyTorch warn that a scheduler stepped before its
	optimizer."""

	training_data, X_valid, y_valid, _ = _data([1, 2], 4)
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		_lightning_fit(tmp_path, _model([1, 2]), training_data, X_valid,
			y_valid, max_epochs=1, loss_weights=loss_weights)

	messages = [str(w.message) for w in caught]
	assert not [m for m in messages if "lr_scheduler.step()" in m]


def test_fit_refuses_a_training_set_smaller_than_one_batch(tmp_path):
	training_data, X_valid, y_valid, _ = _data([1], 4)
	model = _model([1])
	model.name = str(tmp_path / "small")

	with pytest.raises(ValueError, match="fewer than one batch"):
		fit(model, training_data, X_valid, y_valid, accelerator='cpu',
			batch_size=64, num_workers=0)


def test_fit_rejects_an_unknown_dtype(tmp_path):
	training_data, X_valid, y_valid, _ = _data([1], 4)
	with pytest.raises(ValueError, match="dtype"):
		fit(_model([1]), training_data, X_valid, y_valid, dtype='float64',
			accelerator='cpu', batch_size=BATCH_SIZE)


@pytest.mark.parametrize("n,world_size", [(7, 1), (7, 2), (7, 3), (8, 4),
	(2, 4)])
def test_validation_shards_cover_every_row_once(n, world_size):
	rows = []
	for rank in range(world_size):
		start, stop = _shard_bounds(n, rank, world_size)
		rows.extend(range(start, stop))
	assert rows == list(range(n))


@pytest.mark.parametrize("world_size,expected", [(1, True), (2, False)])
def test_setup_turns_off_dynamo_ddp_graph_splitting_under_ddp(monkeypatch,
	world_size, expected):
	"""Under DDP, torch.compile splits the graph at DDP's gradient buckets
	(`torch._dynamo.config.optimize_ddp`). With the model's CUDA-graph
	compile mode, that path failed when one rank recompiled for a
	validation batch of its own size, and the other ranks waited on it until
	the NCCL timeout. Training on several devices turns it off; one device
	leaves the setting alone."""

	import types
	import torch._dynamo

	monkeypatch.setattr(torch._dynamo.config, "optimize_ddp", True)

	training_data, X_valid, y_valid, _ = _data([1], 4)
	module = CherimoyaModule(_model([1]), training_data, X_valid, y_valid)
	module._trainer = types.SimpleNamespace(world_size=world_size)
	module.setup("fit")

	assert torch._dynamo.config.optimize_ddp is expected


@pytest.mark.cuda
def test_checkpoints_record_the_training_device(tmp_path):
	"""The loop this replaces saved both checkpoints from the model on the
	GPU, so the files' tensors were stored as CUDA tensors, which is what
	`torch.load` without a `map_location` hands back. Lightning moves the
	model to the CPU when fitting ends, so the final checkpoint has to be
	written before that for the file to be the same."""

	# 16 filters rather than 8: the Triton kernels need at least 16 channels.
	training_data, X_valid, y_valid, _ = _data([1], 4)
	model = Cherimoya(n_filters=16, n_layers=2, signal_groups=[1],
		verbose=False, compile=False, random_state=0)
	model.name = str(tmp_path / "gpu")
	fit(model, training_data, X_valid, y_valid, max_epochs=1,
		accelerator='gpu', devices=1, batch_size=BATCH_SIZE, num_workers=0)

	for filename in ("gpu.torch", "gpu.final.torch"):
		locations = set()

		def record(storage, location):
			locations.add(location)
			return storage

		torch.load(tmp_path / filename, map_location=record, weights_only=True)
		assert locations == {"cuda:0"}, filename


def test_verbose_prints_the_epoch_table(tmp_path, capsys):
	"""`verbose` prints the table the training log always printed: its header,
	then one row per epoch, the rows `{name}.log` records."""

	training_data, X_valid, y_valid, _ = _data([1, 2], 4)
	model = _model([1, 2])
	model.name = str(tmp_path / "table")
	fit(model, training_data, X_valid, y_valid, max_epochs=2, accelerator='cpu',
		batch_size=BATCH_SIZE, num_workers=0, verbose=True, progress_bar=False)

	lines = capsys.readouterr().out.splitlines()
	start = lines.index("\t".join(LOG_COLUMNS))
	rows = [line.split("\t") for line in lines[start + 1:]
		if line.count("\t") == 10]
	assert len(rows) == 2

	# The printed row is the log's row, rounded as the log has always
	# printed it.
	log = pandas.read_csv(tmp_path / "table.log", sep="\t", dtype=str)
	for epoch, row in enumerate(rows):
		assert row[0] == log['Epoch'][epoch] == str(epoch)
		assert row[1] == log['Iteration'][epoch]
		assert float(row[4]) == pytest.approx(float(log['Training MNLL'][epoch]),
			abs=1e-4)
		assert row[8] == log['Validation Count Pearson'][epoch]
		assert row[10] == log['Saved?'][epoch]


def test_quiet_without_verbose(tmp_path, capsys):
	training_data, X_valid, y_valid, _ = _data([1], 4)
	model = _model([1])
	model.name = str(tmp_path / "quiet")
	fit(model, training_data, X_valid, y_valid, max_epochs=1, accelerator='cpu',
		batch_size=BATCH_SIZE, num_workers=0)

	assert "Validation Count Pearson" not in capsys.readouterr().out


@pytest.mark.parametrize("progress_bar,expected", [(True, True),
	(False, False)])
def test_progress_bar_can_be_forced(tmp_path, progress_bar, expected):
	training_data, X_valid, y_valid, _ = _data([1], 4)
	model = _model([1])
	model.name = str(tmp_path / "bar")
	trainer = fit(model, training_data, X_valid, y_valid, max_epochs=1,
		accelerator='cpu', batch_size=BATCH_SIZE, num_workers=0, verbose=True,
		progress_bar=progress_bar)

	assert (trainer.progress_bar_callback is not None) is expected


def test_progress_bar_by_default_only_for_a_terminal_or_a_notebook(
	monkeypatch):
	"""A bar redirected to a file writes a redraw per step, so by default it
	is drawn only when someone is watching: a terminal or a Jupyter kernel.
	Under pytest stdout is captured, which is neither."""

	import sys
	from cherimoya.training import _show_progress_bar

	monkeypatch.delitem(sys.modules, 'ipykernel', raising=False)
	assert _show_progress_bar(None) is False

	monkeypatch.setitem(sys.modules, 'ipykernel', object())
	assert _show_progress_bar(None) is True

