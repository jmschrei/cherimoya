"""Tests for PeakNegativeSampler that don't require BigWig / FASTA files."""

from unittest import mock

import numpy
import pytest
import torch

from cherimoya.io import (PeakGenerator, PeakNegativeSampler,
	normalize_signal_groups, channel_permutation_from_groups,
	_validate_signal_groups)


def _make_sampler(n_peaks=8, n_negs=8, in_window=20, out_window=10,
		max_jitter=4, controls=False, **kwargs):
	# Allocate enough length for jitter on both sides.
	in_len = in_window + 2 * max_jitter
	out_len = out_window + 2 * max_jitter

	peak_seqs = torch.zeros(n_peaks, 4, in_len)
	# Mark each peak with a unique flag in channel 0 for traceability.
	for i in range(n_peaks):
		peak_seqs[i, 0, :] = float(i + 1)

	# Deterministic signal data so repeat calls to _make_sampler produce
	# bit-identical fixtures (otherwise determinism tests below would
	# fail because the signal tensors differ run-to-run).
	g = torch.Generator().manual_seed(12345)
	peak_signals = torch.randint(0, 5, (n_peaks, 1, out_len),
		generator=g).float()

	neg_seqs = torch.zeros(n_negs, 4, in_len)
	for i in range(n_negs):
		neg_seqs[i, 0, :] = -(float(i + 1))
	neg_signals = torch.randint(0, 5, (n_negs, 1, out_len),
		generator=g).float()

	peak_ctl = torch.zeros(n_peaks, 1, in_len) if controls else None
	neg_ctl = torch.zeros(n_negs, 1, in_len) if controls else None

	# Default the seed so tests are deterministic unless overridden.
	kwargs.setdefault('random_state', 0)

	sampler = PeakNegativeSampler(
		peak_sequences=peak_seqs,
		peak_signals=peak_signals,
		negative_sequences=neg_seqs,
		negative_signals=neg_signals,
		peak_controls=peak_ctl,
		negative_controls=neg_ctl,
		in_window=in_window,
		out_window=out_window,
		max_jitter=max_jitter,
		**kwargs,
	)
	return sampler


# --------- Basic shape / length contract ----------------------------------

def test_len_combines_peaks_and_negative_ratio():
	sampler = _make_sampler(n_peaks=10, n_negs=20, negative_ratio=0.1)
	# 10 peaks + 0.1 * 10 = 1 negatives => length 11.
	assert len(sampler) == 11


def test_getitem_returns_correct_shapes_no_controls():
	sampler = _make_sampler(in_window=20, out_window=10, max_jitter=4)
	X, y, label = sampler[0]
	assert X.shape == (4, 20)
	assert y.shape == (1, 10)
	assert label in (0, 1)


def test_getitem_returns_controls_tuple_when_provided():
	sampler = _make_sampler(controls=True, in_window=20, out_window=10,
		max_jitter=4)
	out = sampler[0]
	assert len(out) == 4
	X, X_ctl, y, label = out
	assert X.shape == (4, 20)
	assert X_ctl.shape == (1, 20)
	assert y.shape == (1, 10)


def test_peak_ordering_stable_within_an_epoch():
	"""peak_ordering must be set up in __init__ (so forked workers that
	never see idx==0 still have a valid ordering) and must stay constant
	for every monotonically-increasing index sequence within one epoch,
	including the very first call at idx==0."""

	sampler = _make_sampler(n_peaks=16, n_negs=4, negative_ratio=0,
		max_jitter=1)
	assert sampler.peak_ordering is not None
	assert sorted(sampler.peak_ordering.tolist()) == list(range(16))

	ordering_initial = sampler.peak_ordering.copy()
	for idx in range(6):  # includes idx==0
		sampler[idx]
		assert numpy.array_equal(sampler.peak_ordering, ordering_initial)


def test_getitem_works_without_idx_zero():
	"""With multiple DataLoader workers, worker N>=1 may never receive
	idx==0. __getitem__ must still produce a valid sample."""

	sampler = _make_sampler(n_peaks=8, n_negs=4, negative_ratio=0,
		max_jitter=2)
	X, y, label = sampler[5]
	assert label == 1
	assert X.shape[0] == 4


# --------- Per-epoch reshuffle (wrap-around detection) -------------------

def test_wrap_around_triggers_reshuffle():
	"""When idx jumps backward (an epoch boundary in any worker's index
	stream) the peak_ordering must be reshuffled."""

	sampler = _make_sampler(n_peaks=16, n_negs=4, negative_ratio=0,
		max_jitter=1)
	for idx in [3, 4, 5, 9]:
		sampler[idx]
	ordering_before_wrap = sampler.peak_ordering.copy()

	sampler[2]   # wrap-around
	assert not numpy.array_equal(sampler.peak_ordering, ordering_before_wrap)


def test_no_reshuffle_when_shuffle_false():
	"""shuffle=False disables peak-ordering shuffling at every epoch."""

	sampler = _make_sampler(n_peaks=16, n_negs=4, negative_ratio=0,
		max_jitter=1, shuffle=False)
	# peak_ordering should be the identity in every epoch.
	expected = numpy.arange(16)
	for idx in [3, 4, 5, 0, 1, 2]:
		sampler[idx]
		assert numpy.array_equal(sampler.peak_ordering, expected)


# --------- Determinism guarantee -----------------------------------------

def _draw_epoch(sampler, n=None):
	"""Return a list of (label, X[0,0], y_first) tuples for one epoch."""
	n = n if n is not None else len(sampler)
	out = []
	for i in range(n):
		X, y, label = sampler[i]
		# Use a small fingerprint so we don't compare large tensors.
		out.append((int(label), float(X[0, 0]), float(y[0, 0])))
	return out


def test_same_seed_produces_same_sequence():
	"""Two samplers built with the same random_state must produce
	identical __getitem__ outputs for every idx."""

	a = _make_sampler(n_peaks=20, n_negs=10, negative_ratio=0.5,
		max_jitter=2, random_state=42)
	b = _make_sampler(n_peaks=20, n_negs=10, negative_ratio=0.5,
		max_jitter=2, random_state=42)
	assert _draw_epoch(a) == _draw_epoch(b)


def test_different_seeds_produce_different_sequences():
	a = _make_sampler(n_peaks=20, n_negs=10, negative_ratio=0.5,
		max_jitter=2, random_state=1)
	b = _make_sampler(n_peaks=20, n_negs=10, negative_ratio=0.5,
		max_jitter=2, random_state=2)
	assert _draw_epoch(a) != _draw_epoch(b)


def test_getitem_is_pure_within_an_epoch():
	"""__getitem__(idx) within a single epoch is independent of call
	history — multiple calls with the same idx return the same data."""

	sampler = _make_sampler(n_peaks=16, n_negs=8, negative_ratio=0.5,
		max_jitter=2, random_state=5)
	first = sampler[3]
	# Visit some other indices, all monotonically increasing so we don't
	# wrap into a new epoch.
	sampler[5]; sampler[7]
	# Now go back. This will wrap and prepare the next epoch — the value
	# at idx=3 in the *next* epoch may differ. Compare within the same
	# epoch instead by re-indexing forward.
	second = sampler[8]
	again_at_3 = sampler[3]   # wraps to a new epoch
	# In the same epoch (epoch 0), the idx=3 result must reproduce when
	# we rebuild a fresh sampler.
	fresh = _make_sampler(n_peaks=16, n_negs=8, negative_ratio=0.5,
		max_jitter=2, random_state=5)
	again_in_fresh_epoch_0 = fresh[3]
	assert torch.equal(first[0], again_in_fresh_epoch_0[0])
	assert torch.equal(first[1], again_in_fresh_epoch_0[1])
	assert first[2] == again_in_fresh_epoch_0[2]
	# And the post-wrap reindex sees epoch 1 content, which differs from
	# epoch 0 with high probability.
	assert (
		not torch.equal(again_at_3[0], first[0])
		or not torch.equal(again_at_3[1], first[1])
	)


def test_each_peak_drawn_exactly_once_per_epoch():
	"""With deterministic positions, every peak appears exactly once
	per epoch and every negative draw comes from the negative tensor —
	this catches any mismatch between _labels and _source_idx."""

	sampler = _make_sampler(n_peaks=32, n_negs=16, negative_ratio=0.5,
		max_jitter=0, random_state=0)
	peaks_drawn, negatives_drawn = [], []
	for i in range(len(sampler)):
		X, _, label = sampler[i]
		marker = int(X[0, 0].item())
		# _make_sampler bakes a positive marker into peak rows and a
		# negative marker into negative rows.
		if label == 1:
			peaks_drawn.append(marker)
		else:
			negatives_drawn.append(marker)
	assert sorted(peaks_drawn) == list(range(1, 33))
	# 16 negative draws, every marker in [-16, -1].
	assert len(negatives_drawn) == 16
	assert all(-16 <= m <= -1 for m in negatives_drawn)


def test_label_proportions_are_exact():
	"""Exactly n_peaks slots are peak draws; the rest are negative."""

	sampler = _make_sampler(n_peaks=64, n_negs=64, negative_ratio=1.0,
		max_jitter=2)
	labels = [sampler[i][-1] for i in range(len(sampler))]
	assert labels.count(1) == 64
	assert labels.count(0) == 64


# --------- Equivalence between num_workers=1 and num_workers>1 -----------

def _build_minimal_dataloader(num_workers, n_peaks=16, batch_size=2,
		seed=0):
	sampler = _make_sampler(n_peaks=n_peaks, n_negs=8, max_jitter=1,
		negative_ratio=0.5, random_state=seed)
	loader = torch.utils.data.DataLoader(
		sampler, batch_size=batch_size, num_workers=num_workers,
		persistent_workers=num_workers > 0,
	)
	return loader


def test_dataloader_two_epochs_two_workers():
	"""Multi-worker loader yields the full dataset, every epoch."""
	loader = _build_minimal_dataloader(num_workers=2)
	for _ in range(2):
		count = sum(batch[0].shape[0] for batch in loader)
		assert count == len(loader.dataset)


def _flatten_batches(loader, n_epochs=1):
	"""Concatenate every batch from `loader` across `n_epochs` into one
	tensor stack so two loaders' outputs can be compared elementwise."""
	rows = []
	for _ in range(n_epochs):
		for batch in loader:
			X, y, label = batch
			for i in range(X.shape[0]):
				rows.append((float(X[i, 0, 0]), float(y[i, 0, 0]),
					int(label[i])))
	return rows


def test_num_workers_does_not_change_data_sequence():
	"""The headline guarantee: num_workers=1 and num_workers=4 must
	produce identical batch sequences for the same seed."""

	loader_a = _build_minimal_dataloader(num_workers=1, seed=99,
		n_peaks=24, batch_size=4)
	loader_b = _build_minimal_dataloader(num_workers=4, seed=99,
		n_peaks=24, batch_size=4)
	a_rows = _flatten_batches(loader_a, n_epochs=2)
	b_rows = _flatten_batches(loader_b, n_epochs=2)
	assert a_rows == b_rows


def test_num_workers_zero_matches_one_worker():
	"""num_workers=0 (in-process) must also match num_workers>=1."""

	loader_a = _build_minimal_dataloader(num_workers=0, seed=7,
		n_peaks=20, batch_size=4)
	loader_b = _build_minimal_dataloader(num_workers=2, seed=7,
		n_peaks=20, batch_size=4)
	assert _flatten_batches(loader_a) == _flatten_batches(loader_b)


# --------- Edge cases: negative_ratio=0 and max_jitter=0 -----------------

def test_negative_ratio_zero_excludes_negatives():
	"""With negative_ratio=0, length excludes negatives and every draw is
	a peak."""
	sampler = _make_sampler(n_peaks=10, n_negs=20, negative_ratio=0)
	assert len(sampler) == 10
	for i in range(len(sampler)):
		_, _, label = sampler[i]
		assert label == 1


def test_negative_ratio_negative_raises():
	with pytest.raises(ValueError, match="negative_ratio"):
		_make_sampler(negative_ratio=-0.1)


def test_max_jitter_zero_returns_full_slice():
	"""max_jitter=0 must not crash the RNG and must return the unjittered
	(offset 0) window."""
	sampler = _make_sampler(n_peaks=2, n_negs=2, max_jitter=0,
		in_window=12, out_window=6, negative_ratio=0)
	for i in range(len(sampler)):
		X, y, label = sampler[i]
		assert label == 1
		assert X.shape == (4, 12)
		assert y.shape == (1, 6)
		# The whole peak row was filled with the same marker, so the
		# offset-0 slice should be uniformly that value.
		marker = X[0, 0].item()
		assert (X[0] == marker).all()


def test_max_jitter_negative_raises():
	with pytest.raises(ValueError, match="max_jitter"):
		_make_sampler(max_jitter=-1)


def test_reverse_complement_flips_when_enabled():
	"""With reverse_complement=True roughly half the peak draws should
	have been flipped along the channel/length axes."""

	sampler = _make_sampler(n_peaks=200, n_negs=20, in_window=8,
		out_window=4, max_jitter=1, reverse_complement=True,
		negative_ratio=0)
	flipped = 0
	peak_draws = 0
	for i in range(len(sampler)):
		X, _, label = sampler[i]
		if label != 1:
			continue
		peak_draws += 1
		# Each unflipped peak has a positive marker in channel 0; flipping
		# along (0, 1) puts a zero there.
		if X[0, 0].item() == 0.0:
			flipped += 1
	assert peak_draws == 200
	assert 0.4 * peak_draws < flipped < 0.6 * peak_draws


# --------- normalize_signal_groups / channel_permutation_from_groups -------

def test_normalize_signal_groups_none():
	flat, groups = normalize_signal_groups(None)
	assert flat is None
	assert groups == []


def test_normalize_signal_groups_flat_list_is_all_unstranded():
	"""Breaking-change semantics: a flat list of N files is N independent
	unstranded groups (NOT a single stranded pair)."""

	flat, groups = normalize_signal_groups(["a.bw", "b.bw", "c.bw"])
	assert flat == ["a.bw", "b.bw", "c.bw"]
	assert groups == [1, 1, 1]


def test_normalize_signal_groups_nested_pair():
	flat, groups = normalize_signal_groups([["plus.bw", "minus.bw"]])
	assert flat == ["plus.bw", "minus.bw"]
	assert groups == [2]


def test_normalize_signal_groups_mixed():
	flat, groups = normalize_signal_groups(
		["atac.bw", ["ctcf.+.bw", "ctcf.-.bw"], "h3k27ac.bw"])
	assert flat == ["atac.bw", "ctcf.+.bw", "ctcf.-.bw", "h3k27ac.bw"]
	assert groups == [1, 2, 1]


def test_normalize_signal_groups_rejects_empty_group():
	with pytest.raises(ValueError, match="empty"):
		normalize_signal_groups([["a.bw"], []])


def test_normalize_signal_groups_rejects_bad_types():
	with pytest.raises(TypeError):
		normalize_signal_groups("not-a-list")
	with pytest.raises(TypeError):
		normalize_signal_groups([1, 2, 3])
	with pytest.raises(TypeError):
		normalize_signal_groups([["a.bw", 5]])


# --- Contract with the `cherimoya pipeline` command's bam2bw rewrite ----
#
# `cherimoya_cli/commands/pipeline.py` rewrites `signals` and `controls`
# to two specific literal forms after `bam2bw` converts BAM/SAM/etc to
# bigWigs. Both forms MUST resolve to a single group whose size matches
# the strandedness:
#
#   unstranded=True  -> ``[name + ".bw"]``                    -> groups=[1]
#   unstranded=False -> ``[[name + ".+.bw", name + ".-.bw"]]`` -> groups=[2]
#
# If either side of this contract changes (the pipeline drops the inner
# list wrapper, or `normalize_signal_groups` re-interprets a singleton
# list of strings), BAM-input pipeline runs would silently regress to
# the original RC scrambling bug. These tests pin both forms in place.

def test_pipeline_bam2bw_unstranded_form_is_one_unstranded_group():
	flat, groups = normalize_signal_groups(["my_experiment.bw"])
	assert flat == ["my_experiment.bw"]
	assert groups == [1]


def test_pipeline_bam2bw_stranded_form_is_one_stranded_pair():
	flat, groups = normalize_signal_groups(
		[["my_experiment.+.bw", "my_experiment.-.bw"]])
	assert flat == ["my_experiment.+.bw", "my_experiment.-.bw"]
	assert groups == [2]


def test_pipeline_bam2bw_control_unstranded_form_is_one_unstranded_group():
	flat, groups = normalize_signal_groups(["my_experiment.control.bw"])
	assert flat == ["my_experiment.control.bw"]
	assert groups == [1]


def test_pipeline_bam2bw_control_stranded_form_is_one_stranded_pair():
	flat, groups = normalize_signal_groups(
		[["my_experiment.control.+.bw", "my_experiment.control.-.bw"]])
	assert flat == [
		"my_experiment.control.+.bw", "my_experiment.control.-.bw"]
	assert groups == [2]


# --------- _validate_signal_groups ----------------------------------------

@pytest.mark.parametrize("groups", [
	[1],
	[1, 1, 1],
	[2],
	[1, 2],
	[2, 1, 3],
])
def test_validate_signal_groups_accepts_valid(groups):
	_validate_signal_groups(groups)   # no raise


def test_validate_signal_groups_rejects_empty():
	with pytest.raises(ValueError, match="non-empty"):
		_validate_signal_groups([])


def test_validate_signal_groups_rejects_non_list():
	with pytest.raises(ValueError, match="list of positive ints"):
		_validate_signal_groups(2)
	with pytest.raises(ValueError, match="list of positive ints"):
		_validate_signal_groups(None)


@pytest.mark.parametrize("bad", [
	[0],          # zero-size group
	[1, 0],       # one of several is zero
	[-1, 2],      # negative
	[1, 1.5],     # float
	[1, "2"],     # string
	[1, True],    # bool — sneaky, isinstance(True, int) is True in plain Python
])
def test_validate_signal_groups_rejects_bad_values(bad):
	with pytest.raises(ValueError, match="positive ints"):
		_validate_signal_groups(bad)


def test_validate_signal_groups_label_appears_in_message():
	"""The `label` kwarg should surface in the error so a caller can
	distinguish between a bad signal_groups vs a bad control_groups."""

	with pytest.raises(ValueError, match="control_groups"):
		_validate_signal_groups([], label='control_groups')


# --------- PeakGenerator (with mocked extract_loci) -----------------------

def _fake_extract_loci_factory(n=16, outlier_idx=None,
		signal_count_template=None):
	"""Build a stand-in for tangermeme's `extract_loci` that produces
	deterministic tensors shaped to whatever signals/controls the
	caller asked for. Stays decoupled from on-disk bigWigs.

	Parameters
	----------
	n : int
		Number of loci returned. Identical for peaks and negatives.
	outlier_idx : int or None
		If not None, the index whose signal counts are 10000x the rest.
		Used to drive the outlier-filter tests.
	signal_count_template : callable(channel_index) -> float or None
		If given, channel `c` is filled with this constant; lets the
		caller stage per-channel magnitudes.
	"""

	def fake(loci, sequences, signals, in_signals, chroms, in_window,
			out_window, max_jitter, min_counts, max_counts, summits,
			exclusion_lists, ignore, return_mask, verbose):
		assert return_mask is True, "PeakGenerator always asks for masks"

		n_signal_ch = len(signals) if signals is not None else 0
		n_ctl_ch = len(in_signals) if in_signals is not None else 0
		in_len = in_window + 2 * max_jitter
		out_len = out_window + 2 * max_jitter

		X = torch.zeros(n, 4, in_len)
		y = torch.ones(n, n_signal_ch, out_len)
		if signal_count_template is not None:
			for c in range(n_signal_ch):
				y[:, c, :] = signal_count_template(c)
		if outlier_idx is not None and outlier_idx < n:
			y[outlier_idx] *= 10_000
		mask = torch.ones(n, dtype=torch.bool)

		if in_signals is not None:
			X_ctl = torch.zeros(n, n_ctl_ch, in_len)
			return X, y, X_ctl, mask
		return X, y, mask

	return fake


def _patch_extract_loci(fake):
	# tangermeme.io.extract_loci is re-exported as
	# cherimoya.io.extract_loci by the `from ... import extract_loci`
	# at the top of io.py, so patch the *bound* name in cherimoya.io.
	return mock.patch('cherimoya.io.extract_loci', side_effect=fake)


def _minimal_peakgen_kwargs(**overrides):
	"""Smallest set of kwargs PeakGenerator needs once extract_loci is
	mocked. Anything not relevant to the assertion is filled with a
	zero/None placeholder."""
	base = dict(
		peaks="ignored.bed", negatives="ignored.bed",
		sequences="ignored.fa", signals=None, controls=None,
		chroms=None, in_window=16, out_window=8, max_jitter=0,
		negative_ratio=0, reverse_complement=False, shuffle=False,
		num_workers=0, batch_size=2, verbose=False,
	)
	base.update(overrides)
	return base


def test_peak_generator_threads_structured_signals_into_sampler():
	"""A grouped signals spec should produce a sampler whose
	signal_perm matches `channel_permutation_from_groups` for those
	groups."""

	with _patch_extract_loci(_fake_extract_loci_factory()):
		loader = PeakGenerator(**_minimal_peakgen_kwargs(
			signals=["atac.bw", ["ctcf.+.bw", "ctcf.-.bw"]]))

	expected = channel_permutation_from_groups([1, 2])
	assert torch.equal(loader.dataset.signal_perm, expected)


def test_peak_generator_signal_groups_disagrees_raises():
	"""Explicit signal_groups must match what the signals shape
	implies. Otherwise the user is silently saying contradictory
	things and downstream RC would be wrong."""

	with _patch_extract_loci(_fake_extract_loci_factory()):
		with pytest.raises(ValueError, match="disagrees"):
			PeakGenerator(**_minimal_peakgen_kwargs(
				signals=[["plus.bw", "minus.bw"]],
				signal_groups=[1, 1]))


def test_peak_generator_control_groups_disagrees_raises():
	with _patch_extract_loci(_fake_extract_loci_factory()):
		with pytest.raises(ValueError, match="disagrees"):
			PeakGenerator(**_minimal_peakgen_kwargs(
				signals=["atac.bw"],
				controls=[["plus.bw", "minus.bw"]],
				control_groups=[1, 1]))


def test_peak_generator_dict_signals_default_to_all_unstranded():
	"""When signals is a list of dicts (the in-memory shortcut),
	PeakGenerator can't infer grouping. With no explicit signal_groups
	it defaults to one unstranded group per dict — matching the
	flat-list semantics."""

	dict_signals = [{}, {}, {}]
	with _patch_extract_loci(_fake_extract_loci_factory()):
		loader = PeakGenerator(**_minimal_peakgen_kwargs(
			signals=dict_signals))

	# All groups size 1 -> permutation is identity.
	assert torch.equal(loader.dataset.signal_perm,
		torch.tensor([0, 1, 2], dtype=torch.long))


def test_peak_generator_dict_signals_with_explicit_groups():
	"""With explicit signal_groups the dict-input path uses them
	verbatim, as long as the sum matches the channel count."""

	dict_signals = [{}, {}, {}]
	with _patch_extract_loci(_fake_extract_loci_factory()):
		loader = PeakGenerator(**_minimal_peakgen_kwargs(
			signals=dict_signals, signal_groups=[1, 2]))

	expected = channel_permutation_from_groups([1, 2])
	assert torch.equal(loader.dataset.signal_perm, expected)


def test_peak_generator_dict_signals_groups_sum_mismatch_raises():
	dict_signals = [{}, {}, {}]
	with _patch_extract_loci(_fake_extract_loci_factory()):
		with pytest.raises(ValueError, match="signal_groups sum"):
			PeakGenerator(**_minimal_peakgen_kwargs(
				signals=dict_signals, signal_groups=[1, 1]))


def test_peak_generator_rejects_non_str_non_list_entries():
	"""A user-mistake input like [42, 'a.bw'] should fail at the
	PeakGenerator boundary, not silently fall through to the
	dict-handling path."""

	with _patch_extract_loci(_fake_extract_loci_factory()):
		with pytest.raises(TypeError):
			PeakGenerator(**_minimal_peakgen_kwargs(
				signals=[42, "a.bw"]))


# --------- M8: per-modality outlier filtering ------------------------------

def test_peak_generator_single_group_outlier_matches_legacy():
	"""For a single signal group the per-group filter is identical to
	the legacy "sum-across-everything" filter. We construct an N-locus
	tensor with one extreme outlier and confirm exactly one locus is
	dropped, matching the old contract."""

	N = 32
	fake = _fake_extract_loci_factory(n=N, outlier_idx=3)

	with _patch_extract_loci(fake):
		loader = PeakGenerator(**_minimal_peakgen_kwargs(
			signals=["atac.bw"]))

	# The outlier is filtered out of the peak set.
	assert loader.dataset.peak_signals.shape[0] == N - 1


def test_peak_generator_per_group_outlier_or_logic():
	"""With two signal groups of wildly different baseline magnitudes,
	an outlier in *either* group should be dropped — and an outlier
	in one group must not survive just because the other group's
	scale doesn't see it as one. Two outliers in different groups
	therefore filter exactly two loci."""

	N = 32

	# Channel 0 baseline = 1, channel 1 baseline = 100. Both have
	# their own outlier, at separate indices.
	def fake(loci, sequences, signals, in_signals, chroms, in_window,
			out_window, max_jitter, min_counts, max_counts, summits,
			exclusion_lists, ignore, return_mask, verbose):
		y = torch.ones(N, 2, out_window)
		y[:, 1, :] = 100.0
		y[5, 0, :] = 1_000_000.0    # outlier in group 0 only
		y[19, 1, :] = 100_000_000.0  # outlier in group 1 only
		X = torch.zeros(N, 4, in_window)
		mask = torch.ones(N, dtype=torch.bool)
		return X, y, mask

	with _patch_extract_loci(fake):
		loader = PeakGenerator(**_minimal_peakgen_kwargs(
			signals=["a.bw", "b.bw"]))

	# Both outliers should be filtered; legacy behavior would've
	# missed the lower-magnitude one because it'd be drowned out.
	assert loader.dataset.peak_signals.shape[0] == N - 2


@pytest.mark.parametrize("groups,expected_perm", [
	([1], [0]),
	([1, 1, 1], [0, 1, 2]),                # all unstranded -> identity
	([2], [1, 0]),                          # single stranded pair -> swap
	([1, 2], [0, 2, 1]),                    # unstranded + stranded
	([2, 1], [1, 0, 2]),                    # stranded + unstranded
	([1, 2, 1, 2], [0, 2, 1, 3, 5, 4]),     # multiple groups, mixed
])
def test_channel_permutation_from_groups(groups, expected_perm):
	perm = channel_permutation_from_groups(groups)
	assert perm.tolist() == expected_perm


# --------- PeakNegativeSampler with signal_perm ----------------------------

def _make_multichannel_sampler(group_sizes, control_group_sizes=None,
		n_peaks=4, in_window=8, out_window=4, max_jitter=0,
		random_state=0):
	"""Build a sampler whose signals have unique per-channel markers so
	we can verify that the post-RC tensor matches the expected permuted
	form exactly. Channel c of every peak is filled with the constant
	value ``c + 1``; the length dim is filled with the same constant so
	the length flip is also observable.
	"""
	n_signal_ch = sum(group_sizes)

	in_len = in_window + 2 * max_jitter
	out_len = out_window + 2 * max_jitter

	peak_seqs = torch.zeros(n_peaks, 4, in_len)
	for i in range(n_peaks):
		peak_seqs[i, 0, :] = float(i + 1)
	neg_seqs = torch.zeros(1, 4, in_len)

	# Signal: channel c has value (c+1) everywhere. Length positions are
	# numbered 0..out_len-1 in channel-relative units so we can also see
	# the length flip.
	peak_signals = torch.zeros(n_peaks, n_signal_ch, out_len)
	for c in range(n_signal_ch):
		for p in range(out_len):
			peak_signals[:, c, p] = (c + 1) * 100 + p
	neg_signals = torch.zeros(1, n_signal_ch, out_len)

	peak_controls = None
	neg_controls = None
	control_perm = None
	if control_group_sizes is not None:
		n_ctl_ch = sum(control_group_sizes)
		peak_controls = torch.zeros(n_peaks, n_ctl_ch, in_len)
		for c in range(n_ctl_ch):
			for p in range(in_len):
				peak_controls[:, c, p] = (c + 1) * 1000 + p
		neg_controls = torch.zeros(1, n_ctl_ch, in_len)
		control_perm = channel_permutation_from_groups(control_group_sizes)

	signal_perm = channel_permutation_from_groups(group_sizes)

	sampler = PeakNegativeSampler(
		peak_sequences=peak_seqs,
		peak_signals=peak_signals,
		negative_sequences=neg_seqs,
		negative_signals=neg_signals,
		peak_controls=peak_controls,
		negative_controls=neg_controls,
		in_window=in_window,
		out_window=out_window,
		max_jitter=max_jitter,
		reverse_complement=True,
		negative_ratio=0,
		random_state=random_state,
		signal_perm=signal_perm,
		control_perm=control_perm,
	)
	return sampler, peak_signals, peak_controls


def _rc_idx(sampler, n_peaks):
	"""Return one peak idx that was RC'd and one that was not, for the
	first epoch's rc-flag pattern."""

	rc, non_rc = None, None
	for i in range(n_peaks):
		if sampler._rc_flags[i] and rc is None:
			rc = i
		elif not sampler._rc_flags[i] and non_rc is None:
			non_rc = i
		if rc is not None and non_rc is not None:
			break
	return rc, non_rc


def test_signal_perm_validates_channel_count():
	"""signal_perm must agree with peak_signals.shape[1]."""

	sampler_kwargs = dict(
		peak_sequences=torch.zeros(2, 4, 8),
		peak_signals=torch.zeros(2, 3, 4),
		negative_sequences=torch.zeros(1, 4, 8),
		negative_signals=torch.zeros(1, 3, 4),
		in_window=8, out_window=4, max_jitter=0,
		negative_ratio=0, random_state=0,
	)
	with pytest.raises(ValueError, match="signal_perm length"):
		PeakNegativeSampler(signal_perm=torch.arange(2), **sampler_kwargs)
	# Matching length is accepted.
	PeakNegativeSampler(signal_perm=torch.arange(3), **sampler_kwargs)


def test_unstranded_group_does_not_swap_under_rc():
	"""A 1-channel (unstranded) group must keep its channel in place
	under RC — only the length dim flips."""

	# Single unstranded track: perm = [0]; RC leaves channel identity.
	sampler, peak_signals, _ = _make_multichannel_sampler(
		group_sizes=[1], n_peaks=16)

	rc_idx, non_rc_idx = _rc_idx(sampler, 16)
	assert rc_idx is not None and non_rc_idx is not None

	# Non-RC sample: matches the source slice exactly.
	X, y, _ = sampler[non_rc_idx]
	assert torch.equal(y, peak_signals[sampler._source_idx[non_rc_idx]])

	# RC sample: channel 0 stays at index 0; length is reversed.
	X, y, _ = sampler[rc_idx]
	src = sampler._source_idx[rc_idx]
	expected = peak_signals[src].flip(-1)
	assert torch.equal(y, expected)


def test_stranded_pair_swaps_under_rc():
	"""A 2-channel (+, -) group must swap channels AND flip length under
	RC. This is the legacy single-pair BPNet behavior — keep it."""

	sampler, peak_signals, _ = _make_multichannel_sampler(
		group_sizes=[2], n_peaks=16)

	rc_idx, _ = _rc_idx(sampler, 16)
	assert rc_idx is not None

	X, y, _ = sampler[rc_idx]
	src = sampler._source_idx[rc_idx]
	# Expected: channels swapped, then length reversed.
	expected = peak_signals[src][[1, 0]].flip(-1)
	assert torch.equal(y, expected)


def test_mixed_unstranded_and_stranded_under_rc():
	"""The headline bug fix: with signals = [unstranded, +, -], RC must
	leave the unstranded channel at index 0 and only swap the +/- pair
	at indices 1, 2 — NOT scramble all three."""

	sampler, peak_signals, _ = _make_multichannel_sampler(
		group_sizes=[1, 2], n_peaks=16)

	rc_idx, non_rc_idx = _rc_idx(sampler, 16)

	# Non-RC: identity (no permutation, no length flip).
	X, y, _ = sampler[non_rc_idx]
	src = sampler._source_idx[non_rc_idx]
	assert torch.equal(y, peak_signals[src])

	# RC: permutation [0, 2, 1] (unstranded stays put; pair swaps),
	# then length reversed.
	X, y, _ = sampler[rc_idx]
	src = sampler._source_idx[rc_idx]
	expected = peak_signals[src][[0, 2, 1]].flip(-1)
	assert torch.equal(y, expected)

	# Critically: under RC, the unstranded channel's per-bp values must
	# come from the SOURCE unstranded channel (not the TF-minus
	# channel). That's the exact regression the old `flip(yi, [0, 1])`
	# caused. Verify the marker value lives in channel 0 either way.
	# Channel 0 markers are 100..100+out_len-1 (forward) or reversed.
	assert y[0, 0].item() >= 100 and y[0, 0].item() < 200


def test_three_groups_independent_under_rc():
	"""Three independent groups [1, 2, 1] under RC: groups don't bleed
	into one another, and each group's internal channels swap correctly."""

	sampler, peak_signals, _ = _make_multichannel_sampler(
		group_sizes=[1, 2, 1], n_peaks=24)

	rc_idx, _ = _rc_idx(sampler, 24)
	X, y, _ = sampler[rc_idx]
	src = sampler._source_idx[rc_idx]
	expected = peak_signals[src][[0, 2, 1, 3]].flip(-1)
	assert torch.equal(y, expected)


def test_control_perm_follows_same_rule_under_rc():
	"""Controls use control_perm symmetrically: a stranded control pair
	swaps its two channels under RC, just like signals."""

	sampler, _, peak_controls = _make_multichannel_sampler(
		group_sizes=[1], control_group_sizes=[2], n_peaks=16)

	rc_idx, _ = _rc_idx(sampler, 16)
	X, X_ctl, y, _ = sampler[rc_idx]
	src = sampler._source_idx[rc_idx]
	expected_ctl = peak_controls[src][[1, 0]].flip(-1)
	assert torch.equal(X_ctl, expected_ctl)


@pytest.mark.parametrize("signal_group_sizes,control_group_sizes", [
	# Stranded signal + unstranded control: signal swaps, control just
	# flips length.
	([2],       [1]),
	# Stranded signal + stranded control: both swap independently.
	([2],       [2]),
	# Mixed signal + stranded control: the user's third scenario —
	# unstranded ATAC + stranded TF signals plus a stranded input
	# control. Each grouping is computed independently.
	([1, 2],    [2]),
	# Mixed signal + unstranded control.
	([1, 2],    [1]),
	# Mixed signal + mixed control with a *different* group layout —
	# confirms signal and control groupings are fully orthogonal.
	([1, 2],    [1, 1, 2]),
	# Two stranded signal groups + one stranded control: every group
	# is its own little swap.
	([2, 2],    [2]),
])
def test_signal_and_control_grouping_are_independent_under_rc(
		signal_group_sizes, control_group_sizes):
	"""Signal-group structure and control-group structure are
	orthogonal: each gets its own precomputed channel permutation, and
	the per-group swaps inside each happen independently. This is the
	contract that lets users mix a stranded ChIP control with an
	unstranded ATAC signal (or any other combination) without manual
	intervention."""

	sampler, peak_signals, peak_controls = _make_multichannel_sampler(
		group_sizes=signal_group_sizes,
		control_group_sizes=control_group_sizes,
		n_peaks=24)

	rc_idx, non_rc_idx = _rc_idx(sampler, 24)
	assert rc_idx is not None and non_rc_idx is not None

	# Non-RC: both tensors come through untouched.
	X, X_ctl, y, _ = sampler[non_rc_idx]
	src = sampler._source_idx[non_rc_idx]
	assert torch.equal(y, peak_signals[src])
	assert torch.equal(X_ctl, peak_controls[src])

	# RC: each tensor gets its own group's permutation, then length flip.
	X, X_ctl, y, _ = sampler[rc_idx]
	src = sampler._source_idx[rc_idx]

	expected_signal_perm = channel_permutation_from_groups(signal_group_sizes)
	expected_control_perm = channel_permutation_from_groups(control_group_sizes)
	expected_y = peak_signals[src][expected_signal_perm].flip(-1)
	expected_ctl = peak_controls[src][expected_control_perm].flip(-1)
	assert torch.equal(y, expected_y)
	assert torch.equal(X_ctl, expected_ctl)


def test_signal_and_control_can_have_different_group_counts():
	"""len(signal_groups) need not equal len(control_groups). Confirm
	the sampler doesn't impose any cross-shape constraint — controls
	are an independent input modality, not a per-signal-group thing."""

	sampler, _, _ = _make_multichannel_sampler(
		group_sizes=[1, 2, 1],          # 3 signal groups
		control_group_sizes=[2],         # 1 control group
		n_peaks=8)
	# Smoke: every index produces a valid sample.
	for i in range(len(sampler)):
		out = sampler[i]
		assert len(out) == 4   # X, X_ctl, y, label


def test_no_perm_falls_back_to_length_only_flip():
	"""When signal_perm=None and reverse_complement=True the channel dim
	stays in place — only length is flipped. (This is what a caller
	that opts out of grouping gets, and is what a single-channel
	sampler effectively does.)"""

	sampler, peak_signals, _ = _make_multichannel_sampler(
		group_sizes=[3], n_peaks=16)
	# Override: pretend the caller did not pass a perm.
	sampler.signal_perm = None

	rc_idx, _ = _rc_idx(sampler, 16)
	X, y, _ = sampler[rc_idx]
	src = sampler._source_idx[rc_idx]
	# Channels in their original order, length reversed.
	expected = peak_signals[src].flip(-1)
	assert torch.equal(y, expected)


def test_sampler_rejects_a_negative_draw_with_no_negatives():
	"""`negative_ratio > 0` with an empty negative set reserves slots it
	cannot fill: `__getitem__` indexes an empty array and dies with an
	IndexError naming neither the ratio nor the set. Say which it is."""

	peaks = torch.zeros(4, 4, 20)
	peaks[:, 0, :] = 1.0
	peak_signals = torch.zeros(4, 1, 10)

	with pytest.raises(ValueError, match="negative"):
		PeakNegativeSampler(
			peak_sequences=peaks, peak_signals=peak_signals,
			negative_sequences=torch.zeros(0, 4, 20),
			negative_signals=torch.zeros(0, 1, 10),
			negative_ratio=0.5, in_window=20, out_window=10,
			random_state=0)


def test_sampler_allows_no_negatives_when_the_ratio_is_zero():
	"""The control: an empty negative set is fine when none are drawn,
	which is how the grouped fit smoke test is set up."""

	peaks = torch.zeros(4, 4, 20)
	peaks[:, 0, :] = 1.0
	peak_signals = torch.zeros(4, 1, 10)

	sampler = PeakNegativeSampler(
		peak_sequences=peaks, peak_signals=peak_signals,
		negative_sequences=torch.zeros(0, 4, 20),
		negative_signals=torch.zeros(0, 1, 10),
		negative_ratio=0, in_window=20, out_window=10, random_state=0)

	assert len(sampler) == 4
