"""Tests for the Cherimoya model construction and forward pass."""

import os

import pytest
import torch

from cherimoya import Cherimoya


@pytest.fixture
def small_model_kwargs():
	# A tiny but valid configuration that runs quickly on CPU.
	return dict(n_filters=8, n_layers=3, signal_groups=[1], n_control_tracks=0,
		verbose=False)


def _input_window_for(model):
	"""Compute a valid input window length for `model`."""
	# Output window must be > 0; trimming bytes are removed from each side.
	return 2 * model.trimming + 64


# --------- Construction ----------------------------------------------------

def test_default_construction():
	model = Cherimoya(verbose=False)
	assert model.n_filters == 128
	assert model.n_layers == 9
	assert model.n_outputs == 1
	assert model.n_control_tracks == 0
	assert model.expansion == 2
	assert model.residual_scale == 0.15
	# Default trimming is 46 + sum_{i<n_layers} 2**i = 46 + 511 = 557.
	assert model.trimming == 46 + sum(2**i for i in range(9))


def test_residual_scale_propagates_to_blocks():
	model = Cherimoya(n_filters=8, n_layers=3, residual_scale=0.07,
		verbose=False)
	assert model.residual_scale == 0.07
	for block in model.blocks:
		assert block.residual_scale == 0.07


def test_residual_scale_round_trips_through_save_load(tmp_path):
	model = Cherimoya(n_filters=8, n_layers=2, residual_scale=0.42,
		verbose=False)
	path = tmp_path / "m.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False)
	assert loaded.residual_scale == 0.42
	for block in loaded.blocks:
		assert block.residual_scale == 0.42


def test_expansion_propagates_to_blocks():
	model = Cherimoya(n_filters=8, n_layers=2, expansion=3, verbose=False)
	for block in model.blocks:
		assert block.expansion == 3
		assert block.linear1.out_features == 24
		assert block.linear2.in_features == 24


def test_expansion_round_trips_through_save_load(tmp_path):
	model = Cherimoya(n_filters=8, n_layers=2, expansion=4, verbose=False)
	path = tmp_path / "m.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False)
	assert loaded.expansion == 4
	for block in loaded.blocks:
		assert block.linear1.out_features == 32


def test_custom_construction(small_model_kwargs):
	model = Cherimoya(**small_model_kwargs)
	assert model.n_filters == 8
	assert model.n_layers == 3
	assert len(model.blocks) == 3


def test_profile_head_and_control_tracks_shape():
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 1],
		n_control_tracks=2, verbose=False)
	assert model.fconv.out_channels == 2
	assert model.fconv.in_channels == 8 + 2  # n_filters + n_control_tracks


# --------- signal_groups construction --------------------------------------

def test_signal_groups_default_is_single_unstranded():
	model = Cherimoya(n_filters=8, n_layers=2, verbose=False)
	assert model.signal_groups == [1]
	assert model.n_outputs == 1
	assert model.n_groups == 1
	assert model.lw0.shape == (1,)
	assert model.lw1.shape == (1,)


def test_signal_groups_grouped_pair():
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		verbose=False)
	assert model.signal_groups == [1, 2]
	assert model.n_outputs == 3   # 1 + 2 channels total
	assert model.n_groups == 2    # 2 groups
	assert model.fconv.out_channels == 3
	# Count head is per-group; lw0 and lw1 are *both* per-group so each
	# modality contributes equally to the Kendall-Gal loss.
	assert model.linear.out_features == 2
	assert model.lw0.shape == (2,)
	assert model.lw1.shape == (2,)


def test_signal_groups_stranded_pair_shares_one_count_prediction():
	"""A stranded ``(+, -)`` pair is one group, so the count head emits
	a single per-group prediction tying the two strands together."""

	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[2],
		verbose=False)
	assert model.n_outputs == 2   # two profile channels
	assert model.n_groups == 1    # one group
	assert model.linear.out_features == 1
	assert model.lw1.shape == (1,)


def test_signal_groups_rejects_bad_values():
	with pytest.raises(ValueError, match="positive ints"):
		Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 0],
			verbose=False)
	with pytest.raises(ValueError, match="positive ints"):
		Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, -1],
			verbose=False)


def test_signal_groups_rejects_empty_list():
	"""An empty signal_groups would construct a model with zero output
	channels — degenerate. The constructor must catch this at the
	boundary rather than letting Conv1d / Linear error 20 lines later
	with a less helpful message."""

	with pytest.raises(ValueError, match="non-empty"):
		Cherimoya(n_filters=8, n_layers=2, signal_groups=[], verbose=False)


def test_grouped_model_forward_loss_measures_integrate():
	"""End-to-end shape contract for a co-trained grouped model: the
	model's forward, the mixture loss, and the performance measures
	must all line up on the same ``signal_groups=[1, 2]`` config. This
	is what would catch a mismatch where, say, the count head was
	sized off n_outputs and the loss off n_groups."""

	from cherimoya.losses import _mixture_loss
	from cherimoya.performance import calculate_performance_measures

	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		verbose=False, compile=False).eval()
	L = _input_window_for(model)
	out_L = L - 2 * model.trimming

	X = torch.randn(4, 4, L)
	y = torch.randint(0, 5, (4, 3, out_L)).float()

	with torch.no_grad():
		y_hat_logits, y_hat_logcounts = model(X)

	# Profile head emits sum(signal_groups) channels; count head emits
	# len(signal_groups) predictions.
	assert y_hat_logits.shape == (4, 3, out_L)
	assert y_hat_logcounts.shape == (4, 2)

	profile_loss, count_loss = _mixture_loss(
		y, y_hat_logits, y_hat_logcounts, signal_groups=[1, 2])
	# Per-group on both sides now: one loss term per modality.
	assert profile_loss.shape == (2,)
	assert count_loss.shape == (2,)
	assert torch.isfinite(profile_loss).all()
	assert torch.isfinite(count_loss).all()

	measures = calculate_performance_measures(
		y_hat_logits, y, y_hat_logcounts,
		measures=['count_pearson', 'count_mse'],
		signal_groups=[1, 2])
	# One Pearson correlation per group.
	assert measures['count_pearson'].shape == (2,)
	assert torch.isfinite(measures['count_pearson']).all()


def test_grouped_model_fit_smoke(tmp_path):
	"""Run ``Cherimoya.fit`` for a few iterations on a grouped model
	(``signal_groups=[1, 2]``) with synthetic data. Confirms the
	training loop, validation pass, optimizer routing, save callback,
	and ``_mixture_loss`` / ``calculate_performance_measures``
	plumbing all agree on shapes end-to-end. CPU-only and ~seconds."""

	import torch
	import os
	from torch.optim import Muon
	from torch.optim.lr_scheduler import LinearLR
	from cherimoya.io import (PeakNegativeSampler,
		channel_permutation_from_groups)

	signal_groups = [1, 2]
	n_outputs = sum(signal_groups)

	model = Cherimoya(n_filters=8, n_layers=2,
		signal_groups=signal_groups, verbose=False, compile=False)
	# .save() writes next to model.name; redirect into tmp_path so the
	# test doesn't pollute the working directory.
	model.name = str(tmp_path / "smoke")

	L = _input_window_for(model)
	out_L = L - 2 * model.trimming

	# Synthetic peaks + negatives. Use deterministic seeds so the test
	# can't flake from random initialization quirks.
	g = torch.Generator().manual_seed(0)
	n_peaks, n_negs = 16, 8
	peak_sequences = torch.zeros(n_peaks, 4, L)
	peak_sequences[:, 0, :] = 1.0  # one-hot at A
	peak_signals = torch.randint(0, 5, (n_peaks, n_outputs, out_L),
		generator=g).float()
	neg_sequences = torch.zeros(n_negs, 4, L)
	neg_sequences[:, 0, :] = 1.0
	neg_signals = torch.zeros(n_negs, n_outputs, out_L)

	sampler = PeakNegativeSampler(
		peak_sequences=peak_sequences, peak_signals=peak_signals,
		negative_sequences=neg_sequences, negative_signals=neg_signals,
		in_window=L, out_window=out_L, max_jitter=0,
		negative_ratio=0, random_state=0, reverse_complement=True,
		signal_perm=channel_permutation_from_groups(signal_groups))
	training_data = torch.utils.data.DataLoader(sampler, batch_size=4,
		num_workers=0)

	# Optimizers — route params using the same helper `cherimoya fit`
	# uses, rather than restating the rule, so this smoke test exercises
	# the real routing instead of a copy that could drift from it.
	from cherimoya_cli.commands.fit import _split_parameters

	muon_params, adam_params, lw_params = _split_parameters(model)
	muon_opt = Muon(muon_params, lr=1e-3, weight_decay=0.0)
	adam_opt = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.0)
	lw_opt = torch.optim.SGD(lw_params, lr=1e-3, weight_decay=0.0,
		momentum=0.9)
	muon_sched = LinearLR(muon_opt, start_factor=1.0, total_iters=1)
	adam_sched = LinearLR(adam_opt, start_factor=1.0, total_iters=1)
	lw_sched = LinearLR(lw_opt, start_factor=1.0, total_iters=1)

	# Validation tensors with the same shape contract.
	X_valid = torch.zeros(4, 4, L)
	X_valid[:, 0, :] = 1.0
	y_valid = torch.randint(0, 5, (4, n_outputs, out_L),
		generator=g).float()

	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		best = model.fit(training_data, muon_opt, adam_opt, lw_opt,
			muon_sched, adam_sched, lw_sched,
			X_valid=X_valid, X_ctl_valid=None, y_valid=y_valid,
			max_epochs=2, batch_size=4, dtype='float32',
			device='cpu', early_stopping=None)
	finally:
		os.chdir(cwd)

	# `best` is the validation count-Pearson at the best epoch (a
	# numpy/torch scalar after `nan_to_num`). It just needs to come
	# back as a finite real number — NaN would indicate a degenerate
	# count target shape.
	import math
	assert math.isfinite(float(best))

	# Both log files exist. The summary log has the original column
	# set; the detail log appends one ProfilePearson/CountPearson
	# column per signal group (here [1, 2] -> two groups -> four
	# extra columns).
	summary_log = tmp_path / "smoke.log"
	detail_log = tmp_path / "smoke.detailed.log"
	assert summary_log.exists(), "summary log not written"
	assert detail_log.exists(), "detail log not written"

	summary_header = summary_log.read_text().splitlines()[0].split("\t")
	detail_header = detail_log.read_text().splitlines()[0].split("\t")

	# Detail header strictly extends the summary header.
	assert detail_header[:len(summary_header)] == summary_header
	extra = detail_header[len(summary_header):]
	assert extra == [
		"ProfilePearson_g0", "ProfilePearson_g1",
		"CountPearson_g0", "CountPearson_g1",
	], "unexpected detail columns: {}".format(extra)


def _tiny_fit_setup(tmp_path, name, n_examples, batch_size):
	"""Build everything ``Cherimoya.fit`` needs for a one-epoch CPU run.

	Returns the model and the keyword arguments for ``fit``, with the
	training data coming from a plain ``TensorDataset`` so the number of
	batches per epoch is exactly known.
	"""

	from torch.optim import Muon
	from torch.optim.lr_scheduler import LinearLR
	from cherimoya_cli.commands.fit import _split_parameters

	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1],
		verbose=False, compile=False, random_state=0)
	# .save()/.log land next to model.name, so point it into tmp_path.
	model.name = str(tmp_path / name)

	L = _input_window_for(model)
	out_L = L - 2 * model.trimming

	g = torch.Generator().manual_seed(0)
	X = torch.zeros(n_examples, 4, L)
	X[:, 0, :] = 1.0  # one-hot at A
	y = torch.randint(0, 5, (n_examples, 1, out_L), generator=g).float()
	labels = torch.ones(n_examples)

	dataset = torch.utils.data.TensorDataset(X, y, labels)
	training_data = torch.utils.data.DataLoader(dataset,
		batch_size=batch_size, num_workers=0)

	muon_params, adam_params, lw_params = _split_parameters(model)
	muon_opt = Muon(muon_params, lr=1e-3, weight_decay=0.0)
	adam_opt = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.0)
	lw_opt = torch.optim.SGD(lw_params, lr=1e-3, weight_decay=0.0,
		momentum=0.9)

	X_valid = torch.zeros(2, 4, L)
	X_valid[:, 0, :] = 1.0
	y_valid = torch.randint(0, 5, (2, 1, out_L), generator=g).float()

	kwargs = dict(training_data=training_data,
		muon_optimizer=muon_opt, adam_optimizer=adam_opt,
		lw_optimizer=lw_opt,
		muon_scheduler=LinearLR(muon_opt, start_factor=1.0, total_iters=1),
		adam_scheduler=LinearLR(adam_opt, start_factor=1.0, total_iters=1),
		lw_scheduler=LinearLR(lw_opt, start_factor=1.0, total_iters=1),
		X_valid=X_valid, X_ctl_valid=None, y_valid=y_valid,
		max_epochs=1, batch_size=batch_size, dtype='float32',
		device='cpu', early_stopping=None)

	return model, kwargs


def _read_log_row(path, epoch=0):
	"""Read one row of a training log, parsed the way a reader would.

	The logger writes the table with ``pandas.DataFrame.to_csv``, so a
	nan lands in the file as an empty cell; reading it back with pandas
	turns it into a nan again.
	"""

	import pandas
	return pandas.read_csv(path, sep="\t").iloc[epoch]


def test_training_losses_are_epoch_averages(tmp_path, monkeypatch):
	"""The ``Training MNLL`` and ``Training Count MSE`` columns report
	the average over the epoch's batches, not the last batch alone
	(issue #19). Spies on ``_mixture_loss`` to capture what each
	training batch actually produced, then checks the logged value
	against their mean -- and against the last batch, which is what the
	column used to hold."""

	import cherimoya.cherimoya as cherimoya_module

	batch_size, n_batches = 2, 4
	model, kwargs = _tiny_fit_setup(tmp_path, "avg",
		n_examples=batch_size * n_batches, batch_size=batch_size)

	# Record the per-batch losses. The validation pass calls
	# `_mixture_loss` too, but under `torch.no_grad()`, so the
	# grad-enabled calls are exactly the training batches.
	real_mixture_loss = cherimoya_module._mixture_loss
	per_batch = []

	def spy(*args, **kwargs_):
		profile_loss, count_loss = real_mixture_loss(*args, **kwargs_)
		if torch.is_grad_enabled():
			per_batch.append((profile_loss.mean().item(),
				count_loss.mean().item()))
		return profile_loss, count_loss

	monkeypatch.setattr(cherimoya_module, "_mixture_loss", spy)

	model.fit(**kwargs)

	assert len(per_batch) == n_batches, (
		"expected one grad-enabled loss call per batch, got {}"
		.format(len(per_batch)))

	row = _read_log_row(tmp_path / "avg.log")
	logged_profile = row["Training MNLL"]
	logged_count = row["Training Count MSE"]

	expected_profile = sum(p for p, _ in per_batch) / n_batches
	expected_count = sum(c for _, c in per_batch) / n_batches

	assert logged_profile == pytest.approx(expected_profile, rel=1e-6)
	assert logged_count == pytest.approx(expected_count, rel=1e-6)

	# The old behavior logged the final batch. Guard against a test that
	# would pass either way by requiring the average to differ from it.
	last_profile, last_count = per_batch[-1]
	assert logged_profile != pytest.approx(last_profile, rel=1e-6)
	assert logged_count != pytest.approx(last_count, rel=1e-6)


def test_epoch_with_no_full_batch_logs_nan_training_losses(tmp_path):
	"""An epoch in which every batch is smaller than ``batch_size`` --
	so the training loop skips them all and no step is taken -- finishes
	and logs nan for the two training-loss columns instead of raising."""

	import math

	# Three examples with batch_size=4: one short batch, which the fit
	# loop skips, leaving the epoch with no training step at all.
	model, kwargs = _tiny_fit_setup(tmp_path, "empty", n_examples=3,
		batch_size=4)

	best = model.fit(**kwargs)
	assert math.isfinite(float(best))

	row = _read_log_row(tmp_path / "empty.log")
	assert math.isnan(row["Training MNLL"])
	assert math.isnan(row["Training Count MSE"])
	# The validation half of the row is still real.
	assert math.isfinite(row["Validation MNLL"])
	assert int(row["Iteration"]) == 0


def test_signal_groups_round_trips_through_save_load(tmp_path):
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		verbose=False)
	path = tmp_path / "m.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False)
	assert loaded.signal_groups == [1, 2]
	assert loaded.n_outputs == 3
	assert loaded.n_groups == 2


@pytest.mark.parametrize("signal_groups,expected_lw0,expected_lw1", [
	([1],       (1,), (1,)),  # single unstranded (the default)
	([1, 1, 1], (3,), (3,)),  # three unstranded groups — one weight per group
	([1, 2],    (2,), (2,)),  # mixed: one profile loss term per *group*
	([2],       (1,), (1,)),  # one stranded pair: 2 profile channels share one loss
])
def test_lw0_lw1_shapes(signal_groups, expected_lw0, expected_lw1):
	"""Both `lw0` and `lw1` size with `len(signal_groups)` — every signal
	group contributes one profile-loss term and one count-loss term
	regardless of channel count, so the uncertainty weights are
	per-group on both sides of the Kendall-Gal combination."""

	model = Cherimoya(n_filters=8, n_layers=2,
		signal_groups=signal_groups, verbose=False)
	assert model.lw0.shape == expected_lw0
	assert model.lw1.shape == expected_lw1
	# Both initialize to ones; the fit loop relies on this.
	assert torch.allclose(model.lw0, torch.ones(*expected_lw0))
	assert torch.allclose(model.lw1, torch.ones(*expected_lw1))


@pytest.mark.parametrize("signal_groups", [
	[1],          # single unstranded (the default)
	[1, 1, 1],    # all unstranded
	[1, 2],       # mixed
])
def test_lw0_lw1_save_load_round_trip(tmp_path, signal_groups):
	"""Save/load preserves lw0/lw1 shapes and values across all
	grouping shapes."""

	model = Cherimoya(n_filters=8, n_layers=2,
		signal_groups=signal_groups, verbose=False)
	# Perturb the weights so the round-trip comparison is non-trivial.
	with torch.no_grad():
		model.lw0.add_(torch.randn_like(model.lw0) * 0.1)
		model.lw1.add_(torch.randn_like(model.lw1) * 0.1)

	path = tmp_path / "m.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False)

	assert loaded.lw0.shape == model.lw0.shape
	assert loaded.lw1.shape == model.lw1.shape
	assert torch.allclose(loaded.lw0, model.lw0)
	assert torch.allclose(loaded.lw1, model.lw1)


def test_default_name_includes_filters_and_layers():
	model = Cherimoya(n_filters=12, n_layers=4, verbose=False)
	assert model.name == "cherimoya.12.4"


# --------- Forward pass ----------------------------------------------------

def test_forward_shape_no_controls(small_model_kwargs):
	model = Cherimoya(**small_model_kwargs).eval()
	L = _input_window_for(model)
	X = torch.randn(2, 4, L)
	y_profile, y_counts = model(X)
	assert y_profile.shape == (2, 1, L - 2 * model.trimming)
	assert y_counts.shape == (2, 1)


def test_forward_with_controls():
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1],
		n_control_tracks=2, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)
	X_ctl = torch.randn(1, 2, L)
	y_profile, y_counts = model(X, X_ctl)
	assert y_profile.shape == (1, 1, L - 2 * model.trimming)
	assert y_counts.shape == (1, 1)


def test_forward_multi_output_per_track_counts():
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 1, 1],
		n_control_tracks=0, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)
	y_profile, y_counts = model(X)
	assert y_profile.shape == (1, 3, L - 2 * model.trimming)
	assert y_counts.shape == (1, 3)


def test_forward_runs_on_default_device(device, small_model_kwargs):
	model = Cherimoya(**small_model_kwargs).to(device).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L, device=device)
	y_profile, y_counts = model(X)
	assert y_profile.device.type == device
	assert y_counts.device.type == device


# --------- Save / load round-trip -----------------------------------------

def test_save_load_roundtrip_preserves_predictions(tmp_path, small_model_kwargs):
	model = Cherimoya(**small_model_kwargs).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)
	expected_profile, expected_counts = model(X)

	path = tmp_path / "model.torch"
	model.save(str(path))

	loaded = Cherimoya.load(str(path), compile=False).eval()
	assert loaded.n_filters == model.n_filters
	assert loaded.n_layers == model.n_layers
	got_profile, got_counts = loaded(X)
	assert torch.allclose(expected_profile, got_profile, atol=1e-6)
	assert torch.allclose(expected_counts, got_counts, atol=1e-6)


def test_save_payload_format(tmp_path, small_model_kwargs):
	model = Cherimoya(**small_model_kwargs)
	path = tmp_path / "model.torch"
	model.save(str(path))
	# Must be loadable in weights_only mode — the security-safe path.
	payload = torch.load(str(path), weights_only=True, map_location='cpu')
	assert isinstance(payload, dict)
	assert set(payload.keys()) == {'config', 'state_dict'}
	assert payload['config']['n_filters'] == small_model_kwargs['n_filters']


def test_saved_checkpoint_uses_legacy_conv_weight_keys(tmp_path,
	small_model_kwargs):
	"""The on-disk key set is a compatibility surface shared with every
	previously trained checkpoint and with installed versions of the
	package, so it is frozen even though the parameter has moved onto
	the ``conv`` submodule. `test_cheri.py` pins this for a lone block;
	this pins it for the artifact `save` actually writes."""

	model = Cherimoya(**small_model_kwargs)
	path = tmp_path / "model.torch"
	model.save(str(path))

	state_dict = torch.load(str(path), weights_only=True,
		map_location='cpu')['state_dict']

	conv_keys = [k for k in state_dict if 'conv_weight' in k]
	assert conv_keys == ['blocks.{}.conv_weight'.format(i)
		for i in range(model.n_layers)]

	# The live parameter is the one that moved; the two spellings must
	# not both appear, or an older install would see an unexpected key.
	assert [n for n, _ in model.named_parameters() if 'conv_weight' in n] == [
		'blocks.{}.conv.conv_weight'.format(i) for i in range(model.n_layers)]


def test_saved_checkpoint_reloads_without_mutating_the_payload(tmp_path,
	small_model_kwargs):
	"""`CheriBlock._load_from_state_dict` re-keys entries as it loads.
	It must do that to a copy: a caller holding the payload — to load it
	into a second model, or to inspect it afterwards — must not find its
	dict rewritten underneath it."""

	model = Cherimoya(**small_model_kwargs)
	path = tmp_path / "model.torch"
	model.save(str(path))
	payload = torch.load(str(path), weights_only=True, map_location='cpu')

	keys_before = list(payload['state_dict'].keys())

	fresh = Cherimoya(**small_model_kwargs)
	fresh.load_state_dict(payload['state_dict'])
	assert list(payload['state_dict'].keys()) == keys_before

	# The second load is the point: it must see the same legacy keys.
	again = Cherimoya(**small_model_kwargs)
	again.load_state_dict(payload['state_dict'])
	assert torch.equal(again.blocks[0].conv.conv_weight,
		model.blocks[0].conv.conv_weight)


def test_load_to_specified_device(tmp_path, small_model_kwargs, device):
	model = Cherimoya(**small_model_kwargs)
	path = tmp_path / "model.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), device=device, compile=False)
	# Check at least one parameter ended up on the requested device.
	param = next(loaded.parameters())
	assert param.device.type == device


# --------- Inference fast-path invariants ---------------------------------
#
# These tests guard against silent regressions when a separate inference
# kernel is dispatched under `torch.no_grad()`. Existing trained-model
# checkpoints must continue to produce the same predictions, so the
# tolerance budget here is tight on fp32.

def test_model_no_grad_matches_grad_cpu(small_model_kwargs):
	"""On CPU the model takes the same code path regardless of grad
	state — verify that explicitly so the merge of an inference-only
	kernel doesn't accidentally divert CPU through a different
	branch."""

	model = Cherimoya(**small_model_kwargs).eval()
	L = _input_window_for(model)
	X = torch.randn(2, 4, L)

	y_profile_grad, y_counts_grad = model(X)
	with torch.no_grad():
		y_profile_ng, y_counts_ng = model(X)

	assert torch.allclose(y_profile_grad.detach(), y_profile_ng,
		atol=1e-6, rtol=1e-5)
	assert torch.allclose(y_counts_grad.detach(), y_counts_ng,
		atol=1e-6, rtol=1e-5)


def test_model_save_load_predictions_match_under_no_grad(tmp_path,
	small_model_kwargs):
	"""The inference-time entry point is: load a saved model, set
	`.eval()`, run forward under `torch.no_grad()`. This is exactly the
	combination a separate inference kernel would dispatch under, so the
	round-trip must produce the same predictions as a grad-enabled
	forward on the unloaded model. Tight tolerance is required because
	users rely on saved checkpoints being numerically stable."""

	model = Cherimoya(**small_model_kwargs).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)

	expected_profile, expected_counts = model(X)

	path = tmp_path / "model.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), compile=False).eval()

	with torch.no_grad():
		got_profile, got_counts = loaded(X)

	assert torch.allclose(expected_profile, got_profile, atol=1e-6)
	assert torch.allclose(expected_counts, got_counts, atol=1e-6)


def test_model_no_grad_matches_grad_with_controls():
	"""Same parity check, exercising the control-tracks branch of the
	forward where the inference path's residual layout could in
	principle differ."""

	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1],
		n_control_tracks=2, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)
	# Counts head takes log(sum(X_ctl)+1); use non-negative controls so
	# that the comparison values are finite.
	X_ctl = torch.rand(1, 2, L)

	y_profile_grad, y_counts_grad = model(X, X_ctl)
	with torch.no_grad():
		y_profile_ng, y_counts_ng = model(X, X_ctl)

	assert torch.allclose(y_profile_grad.detach(), y_profile_ng,
		atol=1e-6, rtol=1e-5)
	assert torch.allclose(y_counts_grad.detach(), y_counts_ng,
		atol=1e-6, rtol=1e-5)


def test_model_small_n_filters_works_under_no_grad():
	"""With n_filters=8 and expansion=1, the per-block hidden width is 8
	— below the multiple-of-16 constraint that a fused inference kernel
	may impose. Such configurations must transparently fall back."""

	model = Cherimoya(n_filters=8, n_layers=2, expansion=1, signal_groups=[1],
		n_control_tracks=0, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)

	with torch.no_grad():
		y_profile, y_counts = model(X)

	assert y_profile.shape == (1, 1, L - 2 * model.trimming)
	assert y_counts.shape == (1, 1)
	assert torch.isfinite(y_profile).all()
	assert torch.isfinite(y_counts).all()


@pytest.mark.cuda
@pytest.mark.triton
def test_model_no_grad_matches_grad_cuda():
	"""GPU parity at the model level. The inference path of an
	individual CheriBlock is correctness-checked in test_cheri.py; this
	test validates that stacking multiple blocks plus the head layers
	does not compound errors past the 1e-4 budget for fp32 inputs."""

	model = Cherimoya(n_filters=32, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False).cuda().eval()
	L = _input_window_for(model)
	X = torch.randn(2, 4, L, device='cuda')

	y_profile_grad, y_counts_grad = model(X)
	with torch.no_grad():
		y_profile_ng, y_counts_ng = model(X)

	prof_diff = (y_profile_grad - y_profile_ng).abs().max().item()
	count_diff = (y_counts_grad - y_counts_ng).abs().max().item()
	assert prof_diff <= 1e-4, f"profile diverged: max-abs-diff={prof_diff}"
	assert count_diff <= 1e-4, f"counts diverged: max-abs-diff={count_diff}"


@pytest.mark.cuda
@pytest.mark.triton
def test_model_save_load_no_grad_matches_grad_cuda(tmp_path):
	"""The full inference workflow on GPU: build, save, reload, eval,
	predict under no_grad. Must match the grad-enabled forward of the
	original (unloaded) model within tight tolerance — this is the
	contract trained checkpoints depend on."""

	model = Cherimoya(n_filters=32, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False).cuda().eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L, device='cuda')

	expected_profile, expected_counts = model(X)

	path = tmp_path / "model.torch"
	model.save(str(path))
	loaded = Cherimoya.load(str(path), device='cuda', compile=False).eval()

	with torch.no_grad():
		got_profile, got_counts = loaded(X)

	prof_diff = (expected_profile - got_profile).abs().max().item()
	count_diff = (expected_counts - got_counts).abs().max().item()
	assert prof_diff <= 1e-4, f"profile diverged after save/load: {prof_diff}"
	assert count_diff <= 1e-4, f"counts diverged after save/load: {count_diff}"


@pytest.mark.cuda
@pytest.mark.triton
def test_cherimoya_backward_matches_cpu_autograd():
	"""End-to-end gradient parity for the full Cherimoya model. CPU
	autograd uses pure PyTorch ops; CUDA autograd uses the Triton
	fwd+bwd kernels in every CheriBlock plus cuDNN/cuBLAS for the
	surrounding layers. The two must agree on every parameter grad and
	the input grad — this is the broadest regression guard for the
	training kernels.

	Tolerance budget: 3 stacked blocks, each with Triton conv+norm bwd
	feeding into cuBLAS linear bwd (TF32). Errors compound; allow
	~5e-3 absolute or relative. This is the realistic precision floor
	of fp32-with-TF32 training on GPU vs a fp32 CPU reference.

	The first GPU call to each block shape triggers Triton autotune,
	whose benchmark trials can contaminate the user-visible output via
	atomic_add residue in the bwd. We warm up to lock the configs
	before measuring."""

	torch.manual_seed(0)
	cpu_model = Cherimoya(n_filters=16, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False)
	gpu_model = Cherimoya(n_filters=16, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False).cuda()
	gpu_model.load_state_dict(cpu_model.state_dict())

	L = _input_window_for(cpu_model)

	# Warmup pass on GPU: triggers autotune for every block's fwd+bwd
	# kernels at the shapes this model uses. Each block has a different
	# dilation so each has its own autotune entry.
	with torch.enable_grad():
		xw = torch.randn(1, 4, L, device='cuda', requires_grad=True)
		yp, yc = gpu_model(xw)
		(yp.sum() + yc.sum()).backward()
		gpu_model.zero_grad()

	x_cpu = torch.randn(1, 4, L, requires_grad=True)
	x_gpu = x_cpu.detach().cuda().requires_grad_(True)

	y_prof_cpu, y_count_cpu = cpu_model(x_cpu)
	y_prof_gpu, y_count_gpu = gpu_model(x_gpu)

	(y_prof_cpu.sum() + y_count_cpu.sum()).backward()
	(y_prof_gpu.sum() + y_count_gpu.sum()).backward()

	# Forward outputs must agree first — otherwise grad comparison
	# isn't meaningful.
	assert torch.allclose(y_prof_cpu, y_prof_gpu.cpu(),
		atol=5e-3, rtol=5e-3), "forward profile diverged"
	assert torch.allclose(y_count_cpu, y_count_gpu.cpu(),
		atol=5e-3, rtol=5e-3), "forward counts diverged"

	# Per-parameter grad parity. Some params (lw0/lw1) aren't used in
	# forward — grad is None on both sides; skip them.
	cpu_params = dict(cpu_model.named_parameters())
	for name, gpu_p in gpu_model.named_parameters():
		cpu_p = cpu_params[name]
		if cpu_p.grad is None:
			assert gpu_p.grad is None, \
				f"{name}: CPU grad None but GPU grad is not"
			continue
		diff = (cpu_p.grad - gpu_p.grad.cpu()).abs().max().item()
		scale = max(cpu_p.grad.abs().max().item(), 1e-6)
		rel = diff / scale
		assert diff < 5e-3 or rel < 5e-3, \
			f"{name}: max-abs-diff={diff:.3e}  max-rel={rel:.3e}"

	# Input grad parity.
	in_diff = (x_cpu.grad - x_gpu.grad.cpu()).abs().max().item()
	assert in_diff < 5e-3, f"input grad max-abs-diff={in_diff:.3e}"


@pytest.mark.cuda
@pytest.mark.triton
def test_cherimoya_inference_megakernel_matches_cpu():
	"""End-to-end forward parity between the pure-PyTorch CPU model and
	the CUDA model under no_grad (which routes every CheriBlock through
	the new megakernel). bf16-dot precision accumulates across the
	stack of blocks, so the tolerance budget is looser than the
	single-block test but still pins a hard upper bound."""

	torch.manual_seed(0)
	cpu_model = Cherimoya(n_filters=16, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False).eval()
	gpu_model = Cherimoya(n_filters=16, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False).cuda().eval()
	gpu_model.load_state_dict(cpu_model.state_dict())

	L = _input_window_for(cpu_model)
	x = torch.randn(1, 4, L)

	with torch.no_grad():
		y_prof_cpu, y_count_cpu = cpu_model(x)
		y_prof_gpu, y_count_gpu = gpu_model(x.cuda())

	prof_diff = (y_prof_cpu - y_prof_gpu.cpu()).abs().max().item()
	count_diff = (y_count_cpu - y_count_gpu.cpu()).abs().max().item()
	assert prof_diff <= 5e-2, \
		f"CPU vs CUDA-megakernel profile max-abs-diff={prof_diff:.3e}"
	assert count_diff <= 5e-2, \
		f"CPU vs CUDA-megakernel counts max-abs-diff={count_diff:.3e}"


@pytest.mark.cuda
@pytest.mark.triton
def test_model_no_grad_stable_across_repeated_calls_cuda():
	"""A weight cache keyed only on Parameter identity could silently
	stale-hit if the model is reused across many inference passes
	(e.g., in saturation mutagenesis loops). Verify deterministic output
	across repeated no_grad forwards on the same input."""

	model = Cherimoya(n_filters=32, n_layers=2, verbose=False).cuda().eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L, device='cuda')

	with torch.no_grad():
		p1, c1 = model(X)
		p2, c2 = model(X)
		p3, c3 = model(X.clone())

	assert torch.equal(p1, p2)
	assert torch.equal(c1, c2)
	assert torch.equal(p1, p3)
	assert torch.equal(c1, c3)


# --------- control-track term in the count head ---------------------------

def test_counts_head_control_term_is_log_of_summed_controls():
	"""The count head takes ``log(sum(controls) + 1)`` as one extra input.

	Asserted on the arithmetic rather than by hooking a module, so it holds
	however that log happens to be spelled. The count path takes control
	tracks *only* through this term -- the trunk features come from the
	sequence alone -- so holding the sequence fixed and changing only the
	controls isolates it: the whole change in the prediction has to be the
	last column of the head's weight times the change in the log term."""

	torch.manual_seed(0)
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1],
		n_control_tracks=2, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)

	ctl_a = torch.rand(1, 2, L)
	ctl_b = torch.rand(1, 2, L) * 7.0     # a different total, same shape

	with torch.no_grad():
		_, counts_a = model(X, ctl_a)
		_, counts_b = model(X, ctl_b)

	# Controls are summed over the trimmed window only, so the untrimmed
	# flanks must not contribute.
	start, end = model.trimming, L - model.trimming
	sums = [c[:, :, start:end].float().sum() for c in (ctl_a, ctl_b)]
	delta_term = torch.log(sums[1] + 1) - torch.log(sums[0] + 1)

	expected = model.linear.weight[:, -1] * delta_term
	observed = (counts_b - counts_a)[0]

	assert torch.allclose(observed, expected, atol=1e-5), (
		"count head's control term is not log(sum + 1): "
		"expected {}, got {}".format(expected.tolist(), observed.tolist()))


def test_counts_head_ignores_controls_outside_the_trimmed_window():
	"""Signal in the untrimmed flanks must not reach the count head, which
	is what makes the sum above a sum over ``[trimming, L - trimming)``."""

	torch.manual_seed(0)
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1],
		n_control_tracks=2, verbose=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)

	ctl = torch.zeros(1, 2, L)
	ctl[:, :, model.trimming:L - model.trimming] = 0.5

	flanked = ctl.clone()
	flanked[:, :, :model.trimming] = 99.0      # only outside the window

	with torch.no_grad():
		_, counts = model(X, ctl)
		_, counts_flanked = model(X, flanked)

	assert torch.equal(counts, counts_flanked)


# --------- random_state initialization -------------------------------------

def _init_state(**kwargs):
	"""Build a small model and return its state dict."""

	params = dict(n_filters=8, n_layers=2, verbose=False)
	params.update(kwargs)
	return Cherimoya(**params).state_dict()


def test_random_state_reproduces_the_initialization():
	a = _init_state(random_state=0)
	b = _init_state(random_state=0)

	assert a.keys() == b.keys()
	for key in a:
		assert torch.equal(a[key], b[key]), (
			"parameter {} differs between two models built with the same "
			"random_state".format(key))


def test_random_state_reproduces_the_initialization_with_controls():
	"""The control path widens fconv and adds a column to the count head,
	so it is initialized by the same calls but at different shapes."""

	a = _init_state(random_state=0, n_control_tracks=2)
	b = _init_state(random_state=0, n_control_tracks=2)

	for key in a:
		assert torch.equal(a[key], b[key])


def test_random_state_ignores_the_global_rng():
	"""A seeded model must not depend on whatever the caller last seeded
	the global RNG with — that is the whole point of the local generator."""

	torch.manual_seed(999)
	a = _init_state(random_state=0)

	torch.manual_seed(111)
	b = _init_state(random_state=0)

	for key in a:
		assert torch.equal(a[key], b[key])


def test_different_random_state_changes_the_initialization():
	a = _init_state(random_state=0)
	b = _init_state(random_state=1)

	assert not torch.equal(a['iconv.weight'], b['iconv.weight'])


def test_no_random_state_leaves_the_initialization_unseeded():
	"""The default stays non-deterministic, so nothing that relied on
	fresh weights per construction changes."""

	a = _init_state()
	b = _init_state()

	assert not torch.equal(a['iconv.weight'], b['iconv.weight'])


def test_random_state_is_not_part_of_the_checkpoint_config():
	"""random_state describes how a model was initialized, not its
	architecture. Persisting it would put a key in the saved config that
	older versions would reject as an unexpected kwarg."""

	model = Cherimoya(n_filters=8, n_layers=2, verbose=False, random_state=0)
	assert 'random_state' not in model._init_kwargs()


def test_random_state_survives_a_save_load_round_trip(tmp_path):
	"""Loading restores weights from the state dict, so the loaded model
	matches the seeded one even though the seed itself is not stored."""

	model = Cherimoya(n_filters=8, n_layers=2, verbose=False, random_state=0)

	path = str(tmp_path / "seeded.torch")
	model.save(path)
	loaded = Cherimoya.load(path)

	a, b = model.state_dict(), loaded.state_dict()
	for key in a:
		assert torch.equal(a[key], b[key])


# --------- fixed loss weights ----------------------------------------------
#
# `lw0` and `lw1` are Parameters of shape (n_groups,), so the Kendall
# mechanism learns one weight per signal group. `loss_weights` replaces them
# with constants, and the per-group depth division is what stands in for the
# per-group adaptation they provided. These import `_group_depths` from the
# module rather than restating the rule.

def test_group_depths_sums_within_each_group():
	"""Each group's depth is a sum over its own channels only. With
	signal_groups=[1, 2] the first group is channel 0 and the second is
	channels 1-2, so a pooled sum would give both the same number."""

	from cherimoya.cherimoya import _group_depths

	y = torch.zeros(2, 3, 10)
	y[:, 0, :] = 1.0      # group 0: 10 counts per example
	y[:, 1, :] = 2.0      # group 1: (2 + 3) * 10 = 50 per example
	y[:, 2, :] = 3.0

	depths = _group_depths(y, [1, 2])
	assert depths.shape == (2,)
	assert torch.allclose(depths, torch.tensor([10.0, 50.0]))


def test_group_depths_averages_over_the_batch():
	"""The divisor is a batch mean, so two examples of different depth
	give their average rather than either one."""

	from cherimoya.cherimoya import _group_depths

	y = torch.zeros(2, 1, 4)
	y[0] = 1.0            # 4 counts
	y[1] = 3.0            # 12 counts

	assert torch.allclose(_group_depths(y, [1]), torch.tensor([8.0]))


def test_group_depths_is_floored_at_one():
	"""A group with no reads in a batch would otherwise divide by zero."""

	from cherimoya.cherimoya import _group_depths

	y = torch.zeros(2, 2, 5)
	y[:, 1, :] = 4.0

	depths = _group_depths(y, [1, 1])
	assert torch.allclose(depths, torch.tensor([1.0, 20.0]))


def test_group_depths_differs_from_a_pooled_mean():
	"""The point of the per-group form: a pooled divisor rescales every
	group by the same number and so leaves their weights relative to each
	other untouched."""

	from cherimoya.cherimoya import _group_depths

	y = torch.zeros(1, 2, 10)
	y[:, 0, :] = 1.0
	y[:, 1, :] = 9.0

	depths = _group_depths(y, [1, 1])
	pooled = y.sum(dim=(1, 2)).float().mean()

	assert not torch.allclose(depths, pooled.expand(2))
	assert torch.allclose(depths.sum(), pooled)


def test_fit_with_loss_weights_freezes_the_kendall_weights(tmp_path):
	"""Passing `loss_weights` must stop `lw0`/`lw1` receiving gradient, so
	the SGD group is inert and the weights hold the values they were
	initialized to."""

	import math
	import os
	from torch.optim import Muon
	from torch.optim.lr_scheduler import LinearLR
	from cherimoya.io import (PeakNegativeSampler,
		channel_permutation_from_groups)
	from cherimoya_cli.commands.fit import _split_parameters

	signal_groups = [1, 2]
	n_outputs = sum(signal_groups)

	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=signal_groups,
		verbose=False, compile=False)
	model.name = str(tmp_path / "fixed")

	L = _input_window_for(model)
	out_L = L - 2 * model.trimming

	g = torch.Generator().manual_seed(0)
	peak_sequences = torch.zeros(16, 4, L)
	peak_sequences[:, 0, :] = 1.0
	peak_signals = torch.randint(0, 5, (16, n_outputs, out_L),
		generator=g).float()
	neg_sequences = torch.zeros(8, 4, L)
	neg_sequences[:, 0, :] = 1.0
	neg_signals = torch.zeros(8, n_outputs, out_L)

	sampler = PeakNegativeSampler(
		peak_sequences=peak_sequences, peak_signals=peak_signals,
		negative_sequences=neg_sequences, negative_signals=neg_signals,
		in_window=L, out_window=out_L, max_jitter=0, negative_ratio=0,
		random_state=0, reverse_complement=True,
		signal_perm=channel_permutation_from_groups(signal_groups))
	training_data = torch.utils.data.DataLoader(sampler, batch_size=4,
		num_workers=0)

	muon_params, adam_params, lw_params = _split_parameters(model)
	muon_opt = Muon(muon_params, lr=1e-3, weight_decay=0.0)
	adam_opt = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.0)
	lw_opt = torch.optim.SGD(lw_params, lr=1e-1, weight_decay=0.0,
		momentum=0.9)
	scheds = [LinearLR(o, start_factor=1.0, total_iters=1)
		for o in (muon_opt, adam_opt, lw_opt)]

	X_valid = torch.zeros(4, 4, L)
	X_valid[:, 0, :] = 1.0
	y_valid = torch.randint(0, 5, (4, n_outputs, out_L),
		generator=g).float()

	lw0_before = model.lw0.detach().clone()
	lw1_before = model.lw1.detach().clone()

	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		best = model.fit(training_data, muon_opt, adam_opt, lw_opt,
			scheds[0], scheds[1], scheds[2],
			X_valid=X_valid, X_ctl_valid=None, y_valid=y_valid,
			max_epochs=2, batch_size=4, dtype='float32', device='cpu',
			early_stopping=None, loss_weights=(1.333, 0.274))
	finally:
		os.chdir(cwd)

	assert math.isfinite(float(best))
	assert model.lw0.requires_grad is False
	assert model.lw1.requires_grad is False
	assert torch.allclose(model.lw0.detach(), lw0_before)
	assert torch.allclose(model.lw1.detach(), lw1_before)


##
# Guards against a run that silently does nothing.
##

def _fit_optimizers(model):
	"""Build the three optimizers and schedulers `fit` takes, routed
	the way `cherimoya fit` routes them."""

	from torch.optim import Muon
	from torch.optim.lr_scheduler import LinearLR

	from cherimoya_cli.commands.fit import _split_parameters

	muon_params, adam_params, lw_params = _split_parameters(model)
	muon_opt = Muon(muon_params, lr=1e-3, weight_decay=0.0)
	adam_opt = torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.0)
	lw_opt = torch.optim.SGD(lw_params, lr=1e-3, momentum=0.9)
	scheds = [LinearLR(o, start_factor=1.0, total_iters=1)
		for o in (muon_opt, adam_opt, lw_opt)]
	return (muon_opt, adam_opt, lw_opt), scheds


def _fit_inputs(model, n=6, loader_batch=3):
	"""A trivial loader plus validation tensors shaped for `model`."""

	L = _input_window_for(model)
	out_L = L - 2 * model.trimming

	g = torch.Generator().manual_seed(0)
	X = torch.zeros(n, 4, L)
	X[:, 0, :] = 1.0
	y = torch.randint(0, 4, (n, 1, out_L), generator=g).float()
	labels = torch.ones(n)

	loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(X, y, labels),
		batch_size=loader_batch)

	X_valid = torch.zeros(2, 4, L)
	X_valid[:, 0, :] = 1.0
	y_valid = torch.randint(0, 4, (2, 1, out_L), generator=g).float()
	return loader, X_valid, y_valid


def test_fit_raises_when_no_batch_matches_batch_size(tmp_path, monkeypatch):
	"""`fit` skips any batch whose size is not exactly `batch_size`. If
	the loader's batch size disagrees with the argument, *every* batch
	is skipped and the run completes having taken no optimizer step,
	logging nan for the full `max_epochs` and saving the initial
	weights as though it had trained.

	A single empty epoch is a supported outcome -- a training set
	smaller than one batch produces it, which
	`test_epoch_with_no_full_batch_logs_nan_training_losses` pins -- so
	this warns rather than raising. The warning is what turns "the log
	has nan in it" into something the user sees without reading the log.
	"""

	torch.manual_seed(0)
	model = Cherimoya(n_filters=4, n_layers=2, signal_groups=[1],
		verbose=False, compile=False)
	loader, X_valid, y_valid = _fit_inputs(model, loader_batch=3)
	opts, scheds = _fit_optimizers(model)
	monkeypatch.chdir(tmp_path)

	with pytest.warns(RuntimeWarning, match="batch_size"):
		model.fit(loader, *opts, *scheds, X_valid=X_valid,
			X_ctl_valid=None, y_valid=y_valid, max_epochs=1,
			batch_size=64, device='cpu')


def test_fit_runs_when_the_batch_size_matches(tmp_path, monkeypatch):
	"""The control: the same setup with matching batch sizes trains and
	warns about nothing."""

	import warnings as _warnings

	torch.manual_seed(0)
	model = Cherimoya(n_filters=4, n_layers=2, signal_groups=[1],
		verbose=False, compile=False)
	loader, X_valid, y_valid = _fit_inputs(model, loader_batch=3)
	opts, scheds = _fit_optimizers(model)
	monkeypatch.chdir(tmp_path)

	with _warnings.catch_warnings(record=True) as caught:
		_warnings.simplefilter("always")
		model.fit(loader, *opts, *scheds, X_valid=X_valid,
			X_ctl_valid=None, y_valid=y_valid, max_epochs=1,
			batch_size=3, device='cpu')

	assert not [w for w in caught
		if issubclass(w.category, RuntimeWarning)
		and "batch_size" in str(w.message)]
