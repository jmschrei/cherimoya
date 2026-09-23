# This file is named for cherimoya_cli/commands/marginalize.py but does not
# cover all of it. It checks that the locus shuffle is seeded; the model load,
# the extract_loci call, and the report itself are untested here.
"""Wiring tests for `cherimoya marginalize` — confirms the shuffle honors
``random_state`` without building a real model or report."""

import argparse
import json
from unittest import mock

import numpy
import torch


def _marginalize_json(tmp_path, **overrides):
	"""Write a JSON that satisfies merge_parameters' required keys."""

	from cherimoya_cli.defaults import default_marginalize_parameters

	cfg = dict(default_marginalize_parameters)
	cfg['sequences'] = 'fake.fa'
	cfg['loci'] = 'fake.bed'
	cfg['motifs'] = 'fake.meme'
	cfg['model'] = 'fake.torch'
	cfg['device'] = 'cpu'
	cfg['shuffle'] = True
	cfg['n_loci'] = None
	cfg.update(overrides)

	path = tmp_path / "marginalize.json"
	path.write_text(json.dumps(cfg))
	return str(path)


class _StubModel:
	# marginalize.run only asks the model whether it needs a control
	# wrapper before handing it to the report, which is mocked out.
	n_control_tracks = 0


def _shuffled_order(path, n=16):
	"""Run marginalize.run and return the locus order the report saw.

	Each locus is filled with its own index so the returned tensor reads
	back as the permutation that was applied.
	"""

	from cherimoya_cli.commands import marginalize as marginalize_cmd

	X = torch.arange(n).reshape(n, 1, 1).expand(n, 4, 8).float()
	captured = {}

	def fake_report(model, motifs, X, output, **kwargs):
		captured['X'] = X

	with mock.patch("cherimoya.Cherimoya") as model_cls, \
			mock.patch("tangermeme.io.extract_loci", return_value=X), \
			mock.patch("bpnetlite.marginalize.marginalization_report",
				side_effect=fake_report):
		model_cls.load.return_value = _StubModel()
		marginalize_cmd.run(argparse.Namespace(parameters=path))

	assert 'X' in captured, "marginalization_report was never called"
	return captured['X'][:, 0, 0].tolist()


def test_default_marginalize_random_state_is_zero():
	from cherimoya_cli.defaults import default_marginalize_parameters

	assert default_marginalize_parameters['random_state'] == 0


def test_marginalize_shuffle_is_seeded(tmp_path):
	"""Two runs with the same seed must pick the same loci in the same
	order. The seed was documented as 0 long before it was read."""

	a = _shuffled_order(_marginalize_json(tmp_path, random_state=0))
	b = _shuffled_order(_marginalize_json(tmp_path, random_state=0))

	assert a == b


def test_marginalize_shuffle_uses_the_seed_it_is_given(tmp_path):
	"""A different seed must give a different order, which is what shows
	the value is read rather than merely accepted."""

	a = _shuffled_order(_marginalize_json(tmp_path, random_state=0))
	b = _shuffled_order(_marginalize_json(tmp_path, random_state=1))

	assert a != b


def test_marginalize_shuffle_matches_numpy_for_that_seed(tmp_path):
	"""Pin the permutation to RandomState so a later refactor to a
	different generator shows up as a failure rather than silently
	changing which loci every report is built from."""

	order = _shuffled_order(_marginalize_json(tmp_path, random_state=0))

	expected = numpy.arange(16)
	numpy.random.RandomState(0).shuffle(expected)
	assert order == expected.tolist()


def test_marginalize_without_shuffle_keeps_the_input_order(tmp_path):
	order = _shuffled_order(_marginalize_json(tmp_path, shuffle=False))

	assert order == list(range(16))


def _extracted_n_loci(path, n=16):
	"""Run marginalize.run and return the `n_loci` that reached
	`extract_loci`, plus the locus order the report saw."""

	from cherimoya_cli.commands import marginalize as marginalize_cmd

	X = torch.arange(n).reshape(n, 1, 1).expand(n, 4, 8).float()
	captured = {}

	def fake_extract(**kwargs):
		# `extract_loci` stops once it has `n_loci` usable sequences, so
		# it returns the *first* n_loci rows. Modelling that is what
		# makes these tests able to tell the two behaviours apart -- a
		# mock that returns everything regardless hides the bug.
		captured['n_loci'] = kwargs['n_loci']
		if kwargs['n_loci'] is None:
			return X
		return X[:kwargs['n_loci']]

	def fake_report(model, motifs, X, output, **kwargs):
		captured['X'] = X

	with mock.patch("cherimoya.Cherimoya") as model_cls, \
			mock.patch("tangermeme.io.extract_loci",
				side_effect=fake_extract), \
			mock.patch("bpnetlite.marginalize.marginalization_report",
				side_effect=fake_report):
		model_cls.load.return_value = _StubModel()
		marginalize_cmd.run(argparse.Namespace(parameters=path))

	return captured


def test_shuffle_samples_across_the_whole_locus_file(tmp_path):
	"""With `shuffle` on, the extraction must not be capped at `n_loci`.

	`extract_loci` returns the *first* `n_loci` rows it can use, so
	capping it and then shuffling permutes a set already chosen by file
	order — the report is built from the top of the BED every time,
	which is what `shuffle` exists to avoid.
	"""

	captured = _extracted_n_loci(
		_marginalize_json(tmp_path, shuffle=True, n_loci=4))

	assert captured['n_loci'] is None


def test_shuffle_still_truncates_to_n_loci(tmp_path):
	"""Extracting everything is how the sample is drawn, not what the
	report is given: the report still sees exactly `n_loci`."""

	captured = _extracted_n_loci(
		_marginalize_json(tmp_path, shuffle=True, n_loci=4))

	assert captured['X'].shape[0] == 4


def test_shuffle_picks_loci_from_beyond_the_first_n(tmp_path):
	"""The observable consequence: the sampled loci are not the first
	`n_loci` of the file."""

	captured = _extracted_n_loci(
		_marginalize_json(tmp_path, shuffle=True, n_loci=4))

	chosen = sorted(captured['X'][:, 0, 0].tolist())
	assert chosen != [0.0, 1.0, 2.0, 3.0]


def test_no_shuffle_still_caps_the_extraction(tmp_path):
	"""Without `shuffle` the cap is what keeps the unshuffled path from
	reading the whole file into memory, so it has to stay."""

	captured = _extracted_n_loci(
		_marginalize_json(tmp_path, shuffle=False, n_loci=4))

	assert captured['n_loci'] == 4
