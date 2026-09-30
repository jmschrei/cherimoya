"""Tests for the cherimoya evaluate CLI's TSV output shape.

The evaluate command writes one row per signal group, in
signal_groups order, with the seven peak columns it always wrote followed
by five that use negatives (NaN without them). Single-group models
continue to emit exactly one data row (byte-identical in the peak
columns to the legacy `.mean()`-over-everything behavior); multi-group
models emit N rows.

These tests mock `tangermeme.io.extract_loci` so they don't need any
bigWig / FASTA fixtures.
"""

import argparse
import json
from unittest import mock

import pytest
import torch

from tangermeme.predict import predict

from cherimoya import Cherimoya


PEAK_COLUMNS = ['profile_mnll', 'profile_jsd', 'profile_pearson',
	'profile_spearman', 'count_pearson', 'count_spearman', 'count_mse']
NEGATIVE_COLUMNS = ['all_count_pearson', 'all_count_spearman',
	'all_count_mse', 'auroc', 'auprc']


def _build_and_save(tmp_path, signal_groups, name="m"):
	"""Build a tiny grouped Cherimoya and save it to tmp_path. Returns
	the checkpoint path."""
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=signal_groups,
		verbose=False, compile=False, random_state=0)
	ckpt = tmp_path / "{}.torch".format(name)
	model.save(str(ckpt))
	return ckpt, model


def _make_fake_extract_loci(n_loci, n_signal_ch, in_window, out_window,
		seed=0, n_negatives=3):
	"""Return a stand-in for tangermeme's extract_loci that yields
	deterministic synthetic peaks, or `n_negatives` negatives when the
	loci are "negatives.bed". Calls draw from one generator, so the peaks
	of a first call do not depend on whether negatives follow. evaluate.py
	always calls extract_loci with `return_mask` *unset* (default
	False), so the return is just (X, y) or (X, y, X_ctl)."""
	g = torch.Generator().manual_seed(seed)

	def fake(loci, sequences, signals, in_signals, chroms, in_window,
			out_window, exclusion_lists, max_jitter, ignore, verbose):
		n = n_negatives if loci == "negatives.bed" else n_loci
		X = torch.randn(n, 4, in_window, generator=g)
		y = torch.randint(0, 5, (n, n_signal_ch, out_window),
			generator=g).float()
		if in_signals is not None:
			X_ctl = torch.zeros(n, len(in_signals), in_window)
			return X, y, X_ctl
		return X, y
	return fake


def _run_evaluate(tmp_path, ckpt, signals, n_signal_ch, controls=None,
		in_window=2 * 49 + 64, out_window=64, n_loci=4, negatives=None):
	"""Run cherimoya evaluate end-to-end against the mocked
	extract_loci, returning the parsed TSV (header, rows)."""
	perf_path = tmp_path / "perf.tsv"
	cfg = {
		"sequences": "ignored.fa",
		"loci": "ignored.bed",
		"signals": signals,
		"controls": controls,
		"chroms": ["chr1"],
		"in_window": in_window,
		"out_window": out_window,
		"model": str(ckpt),
		"performance_filename": str(perf_path),
		"device": "cpu",
		"dtype": "float32",
		"batch_size": 4,
		"compile": False,
		"verbose": False,
		"reverse_complement_average": False,
		"exclusion_lists": None,
		"skip": False,
	}
	if negatives is not None:
		cfg["negatives"] = negatives
	json_path = tmp_path / "evaluate.json"
	json_path.write_text(json.dumps(cfg))

	from cherimoya_cli.commands import evaluate as evaluate_cmd

	fake = _make_fake_extract_loci(
		n_loci=n_loci, n_signal_ch=n_signal_ch,
		in_window=in_window, out_window=out_window)
	# evaluate.py imports extract_loci locally inside run(), so we
	# have to patch the source module rather than a re-export.
	with mock.patch("tangermeme.io.extract_loci", side_effect=fake):
		evaluate_cmd.run(argparse.Namespace(parameters=str(json_path)))

	lines = perf_path.read_text().splitlines()
	header = lines[0].split("\t")
	rows = [line.split("\t") for line in lines[1:]]
	return header, rows


# --- Single group: byte-identical layout to the legacy one-row TSV --------

def test_evaluate_single_unstranded_writes_one_row(tmp_path):
	ckpt, model = _build_and_save(tmp_path, [1])
	header, rows = _run_evaluate(tmp_path, ckpt,
		signals=["atac.bw"], n_signal_ch=1)

	assert header == PEAK_COLUMNS + NEGATIVE_COLUMNS
	assert len(rows) == 1, (
		"single-group model must emit exactly one data row; got {}"
		.format(len(rows)))
	# Every peak value parses as a finite float; without negatives the
	# rest are NaN.
	for v in rows[0][:len(PEAK_COLUMNS)]:
		assert float(v) == float(v), v   # NaN check via self-comparison
	assert rows[0][len(PEAK_COLUMNS):] == ["nan"] * len(NEGATIVE_COLUMNS)


def test_evaluate_single_stranded_pair_writes_one_row(tmp_path):
	"""A stranded BPNet-style experiment is still ONE signal group, so
	the TSV is still one row. The two strands are pooled into a single
	per-group profile metric and share a single per-group count."""
	ckpt, _ = _build_and_save(tmp_path, [2])
	header, rows = _run_evaluate(tmp_path, ckpt,
		signals=[["tf.+.bw", "tf.-.bw"]], n_signal_ch=2)
	assert len(rows) == 1


# --- Multi-group: N rows, one per signal group, in declaration order ------

def test_evaluate_mixed_groups_writes_one_row_per_group(tmp_path):
	"""signal_groups=[1, 2] yields two rows: row 0 is the unstranded
	ATAC group, row 1 is the stranded TF group. Row order is the
	contract — no extra group identifier column."""
	ckpt, _ = _build_and_save(tmp_path, [1, 2])
	header, rows = _run_evaluate(tmp_path, ckpt,
		signals=["atac.bw", ["tf.+.bw", "tf.-.bw"]], n_signal_ch=3)

	# Same columns as the single-group case — no `group` identifier
	# was added.
	assert header == PEAK_COLUMNS + NEGATIVE_COLUMNS
	assert len(rows) == 2


def test_evaluate_three_groups_writes_three_rows(tmp_path):
	ckpt, _ = _build_and_save(tmp_path, [1, 2, 1])
	header, rows = _run_evaluate(tmp_path, ckpt,
		signals=[
			"atac.bw",
			["tf1.+.bw", "tf1.-.bw"],
			"tf2.bw",
		],
		n_signal_ch=4)
	assert len(rows) == 3


# --- Byte-identical to a hand-computed single-group .mean() ---------------

def test_evaluate_single_group_value_equals_legacy_full_mean(tmp_path):
	"""For a single-group model the per-group row should equal the
	value the legacy `measures[name].mean()` code path produced. This
	is what 'existing one-group models yielding the exact same TSV'
	means in practice."""
	from cherimoya.performance import calculate_performance_measures

	ckpt, _ = _build_and_save(tmp_path, [1])
	header, rows = _run_evaluate(tmp_path, ckpt,
		signals=["atac.bw"], n_signal_ch=1, n_loci=8)

	# Re-run the same prediction path manually to derive the expected
	# .mean() values, then compare.
	model = Cherimoya.load(str(ckpt), compile=False).eval()
	fake = _make_fake_extract_loci(
		n_loci=8, n_signal_ch=1,
		in_window=2 * model.trimming + 64, out_window=64)
	X, y = fake(loci=None, sequences=None, signals=["atac.bw"],
		in_signals=None, chroms=None,
		in_window=2 * model.trimming + 64, out_window=64,
		exclusion_lists=None, max_jitter=0, ignore=None, verbose=False)

	# Predict through `predict` at the same batch size `evaluate` uses,
	# rather than with one unbatched forward. The two disagree in the
	# last float32 ULPs, and `profile_spearman` turns that into a
	# discrete jump -- it ranks with `argsort().argsort()`, so a single
	# swapped pair moves the metric far more than the float difference
	# that caused it. That is what made this test fail intermittently on
	# one leg of the CI matrix while passing everywhere else.
	y_hat_logits, y_hat_logcounts = predict(model, X, args=None,
		batch_size=4, device="cpu", dtype="float32", verbose=False)

	measures = calculate_performance_measures(
		y_hat_logits, y, y_hat_logcounts,
		signal_groups=[1])
	measure_names = ['profile_mnll', 'profile_jsd', 'profile_pearson',
		'profile_spearman', 'count_pearson', 'count_spearman', 'count_mse']
	expected = [measures[name].mean().item() for name in measure_names]

	# Both sides now see bit-identical predictions, so the only thing
	# left between them is the aggregation: `evaluate` takes
	# `value[:, offset:offset+g].mean()` over a slice while this takes
	# `value.mean()` over the whole tensor. For one group those cover
	# the same elements in a different reduction order. That difference
	# is smooth -- unlike a rank metric it cannot jump -- and measures
	# 0 here, so the bound is a guard rather than a fitted tolerance.
	assert [float(v) for v in rows[0][:len(PEAK_COLUMNS)]] == pytest.approx(
		expected, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("signal_groups,signals,n_signal_ch", [
	([1], ["atac.bw"], 1),
	([1, 2], ["atac.bw", ["tf.+.bw", "tf.-.bw"]], 3),
])
def test_evaluate_negatives_add_columns_and_leave_the_peak_ones(tmp_path,
	signal_groups, signals, n_signal_ch):
	"""Negatives leave the peak columns as they are without them, and the
	five columns that use them are computed over peaks and negatives, per
	group."""
	from sklearn.metrics import average_precision_score
	from sklearn.metrics import roc_auc_score

	from cherimoya.performance import spearman_corr

	ckpt, _ = _build_and_save(tmp_path, signal_groups)
	_, peak_rows = _run_evaluate(tmp_path, ckpt, signals=signals,
		n_signal_ch=n_signal_ch, n_loci=8)
	header, rows = _run_evaluate(tmp_path, ckpt, signals=signals,
		n_signal_ch=n_signal_ch, n_loci=8, negatives="negatives.bed")

	assert header == PEAK_COLUMNS + NEGATIVE_COLUMNS
	for row, peak_row in zip(rows, peak_rows):
		assert row[:len(PEAK_COLUMNS)] == peak_row[:len(PEAK_COLUMNS)]

	model = Cherimoya.load(str(ckpt), compile=False).eval()
	in_window = 2 * model.trimming + 64
	fake = _make_fake_extract_loci(n_loci=8, n_signal_ch=n_signal_ch,
		in_window=in_window, out_window=64)
	kwargs = dict(sequences=None, signals=None, in_signals=None, chroms=None,
		in_window=in_window, out_window=64, exclusion_lists=None,
		max_jitter=0, ignore=None, verbose=False)
	X_peaks, y_peaks = fake(loci="peaks.bed", **kwargs)
	X_negatives, y_negatives = fake(loci="negatives.bed", **kwargs)
	X = torch.cat([X_peaks, X_negatives])
	y = torch.cat([y_peaks, y_negatives])
	labels = [1] * len(X_peaks) + [0] * len(X_negatives)

	_, y_hat_logcounts = predict(model, X, args=None, batch_size=4,
		device="cpu", dtype="float32", verbose=False)

	ends = [0]
	for g in signal_groups:
		ends.append(ends[-1] + g)
	for i, row in enumerate(rows):
		target = torch.log(y[:, ends[i]:ends[i + 1]].sum(dim=(1, 2)) + 1)
		pred = y_hat_logcounts[:, i]
		expected = [
			torch.corrcoef(torch.stack([target, pred]))[0, 1].item(),
			spearman_corr(target, pred).item(),
			((target - pred) ** 2).mean().item(),
			roc_auc_score(labels, pred),
			average_precision_score(labels, pred),
		]
		assert [float(v) for v in row[len(PEAK_COLUMNS):]] == pytest.approx(
			expected, rel=1e-5, abs=1e-6)
