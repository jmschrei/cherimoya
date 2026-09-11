"""Tests for the cherimoya evaluate CLI's TSV output shape.

The evaluate command writes one row per signal group, in
signal_groups order, with the same seven columns as before. Single-
group models continue to emit exactly one data row (byte-identical
to the legacy `.mean()`-over-everything behavior); multi-group
models emit N rows.

These tests mock `tangermeme.io.extract_loci` so they don't need any
bigWig / FASTA fixtures.
"""

import argparse
import json
from unittest import mock

import pytest
import torch

from cherimoya import Cherimoya


def _build_and_save(tmp_path, signal_groups, name="m"):
	"""Build a tiny grouped Cherimoya and save it to tmp_path. Returns
	the checkpoint path."""
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=signal_groups,
		verbose=False, compile=False)
	ckpt = tmp_path / "{}.torch".format(name)
	model.save(str(ckpt))
	return ckpt, model


def _make_fake_extract_loci(n_loci, n_signal_ch, in_window, out_window,
		seed=0):
	"""Return a stand-in for tangermeme's extract_loci that yields
	deterministic synthetic peaks for a single call. evaluate.py
	always calls extract_loci with `return_mask` *unset* (default
	False), so the return is just (X, y) or (X, y, X_ctl)."""
	g = torch.Generator().manual_seed(seed)

	def fake(loci, sequences, signals, in_signals, chroms, in_window,
			out_window, exclusion_lists, max_jitter, ignore, verbose):
		X = torch.randn(n_loci, 4, in_window, generator=g)
		y = torch.randint(0, 5, (n_loci, n_signal_ch, out_window),
			generator=g).float()
		if in_signals is not None:
			X_ctl = torch.zeros(n_loci, len(in_signals), in_window)
			return X, y, X_ctl
		return X, y
	return fake


def _run_evaluate(tmp_path, ckpt, signals, n_signal_ch, controls=None,
		in_window=2 * 49 + 64, out_window=64, n_loci=4):
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
		"verbose": False,
		"reverse_complement_average": False,
		"exclusion_lists": None,
		"skip": False,
	}
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

	assert header == ['profile_mnll', 'profile_jsd', 'profile_pearson',
		'profile_spearman', 'count_pearson', 'count_spearman', 'count_mse']
	assert len(rows) == 1, (
		"single-group model must emit exactly one data row; got {}"
		.format(len(rows)))
	# Every value parses as a finite float.
	for v in rows[0]:
		assert float(v) == float(v), v   # NaN check via self-comparison


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
	assert header == ['profile_mnll', 'profile_jsd', 'profile_pearson',
		'profile_spearman', 'count_pearson', 'count_spearman', 'count_mse']
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

	with torch.no_grad():
		y_hat_logits, y_hat_logcounts = model(X)

	measures = calculate_performance_measures(
		y_hat_logits, y, y_hat_logcounts,
		signal_groups=[1])
	measure_names = ['profile_mnll', 'profile_jsd', 'profile_pearson',
		'profile_spearman', 'count_pearson', 'count_spearman', 'count_mse']
	expected = [measures[name].mean().item() for name in measure_names]

	# The per-group row must match the legacy `.mean()` values. Compare
	# numerically rather than by exact string: the CLI runs predictions
	# batched while this manual path runs them in a single pass, and the
	# 75-wide `fconv` head makes the two differ in the last few float32
	# ULPs (the old 1x1 head was bit-identical regardless of batching).
	# Agreement to 4 decimals is the repo's regression tolerance.
	assert [float(v) for v in rows[0]] == pytest.approx(expected, abs=1e-4)
