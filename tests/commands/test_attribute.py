"""Tests that `cherimoya attribute` honours its window parameters.

`in_window` was declared in the schema and documented in the CLI
reference but never reached `extract_loci`, so a model trained at any
other window was fed 2114bp regardless. The attributed slice was a
hard-coded 400bp with no key at all.
"""

import argparse
import json
from unittest import mock

import numpy
import pytest
import torch

from cherimoya_cli.defaults import default_attribute_parameters


def _run_attribute(tmp_path, n_loci=3, **overrides):
	"""Run the command against a stubbed extract_loci and saturation
	mutagenesis, returning what each of them saw."""

	from cherimoya_cli.commands import attribute

	captured = {}

	def fake_extract(**kwargs):
		captured['extract'] = kwargs
		width = kwargs.get('in_window') or 2114
		X = torch.zeros(n_loci, 4, width)
		X[:, 0, :] = 1.0            # a clean one-hot, so nothing is filtered
		return X, torch.ones(n_loci, dtype=bool)

	def fake_sm(model, X, **kwargs):
		captured['sm'] = kwargs
		start, end = kwargs['start'], kwargs['end']
		return torch.zeros(X.shape[0], 4, end - start)

	cfg = dict(default_attribute_parameters)
	cfg.update({
		'sequences': 'f.fa', 'loci': 'f.bed', 'model': 'm.torch',
		'device': 'cpu', 'verbose': False,
		'ohe_filename': str(tmp_path / 'a.ohe.npz'),
		'attr_filename': str(tmp_path / 'a.attr.npz'),
		'idx_filename': str(tmp_path / 'a.idx.npy'),
	})
	cfg.update(overrides)

	path = tmp_path / 'attribute.json'
	with open(path, 'w') as f:
		json.dump(cfg, f)

	with mock.patch('cherimoya.Cherimoya') as model_cls, \
			mock.patch('tangermeme.io.extract_loci',
				side_effect=fake_extract), \
			mock.patch('tangermeme.saturation_mutagenesis.'
				'saturation_mutagenesis', side_effect=fake_sm):
		model_cls.load.return_value = mock.MagicMock(n_control_tracks=0)
		attribute.run(argparse.Namespace(parameters=str(path)))

	captured['ohe'] = numpy.load(cfg['ohe_filename'])['arr_0']
	captured['attr'] = numpy.load(cfg['attr_filename'])['arr_0']
	return captured


##


def test_in_window_reaches_extract_loci(tmp_path):
	"""The schema declares `in_window` and the CLI reference documents
	it; it has to be the window that is actually extracted."""

	captured = _run_attribute(tmp_path, in_window=4096)

	assert captured['extract']['in_window'] == 4096


def test_default_in_window_is_unchanged(tmp_path):
	"""The default still extracts 2114, so existing runs are
	unaffected."""

	captured = _run_attribute(tmp_path)

	assert captured['extract']['in_window'] == 2114


def test_attr_window_sets_the_attributed_slice(tmp_path):
	"""The slice handed to saturation mutagenesis is `attr_window` wide
	and centred in the extraction window."""

	captured = _run_attribute(tmp_path, in_window=2114, attr_window=600)

	assert captured['sm']['end'] - captured['sm']['start'] == 600
	assert captured['sm']['start'] == 2114 // 2 - 300


def test_default_attr_window_matches_the_old_hard_coded_slice(tmp_path):
	"""400bp centred in a 2114bp window, i.e. positions 857-1257, which
	is what the hard-coded ``mid - 200, mid + 200`` produced."""

	captured = _run_attribute(tmp_path)

	assert captured['sm']['start'] == 857
	assert captured['sm']['end'] == 1257


def test_saved_arrays_match_the_attributed_slice(tmp_path):
	"""The one-hot and attribution arrays written to disk must both be
	`attr_window` wide -- `cherimoya seqlets` reads the width off them
	to convert positions back to the genome."""

	captured = _run_attribute(tmp_path, attr_window=600)

	assert captured['ohe'].shape[-1] == 600
	assert captured['attr'].shape[-1] == 600


def test_attr_window_wider_than_in_window_raises(tmp_path):
	"""A slice that does not fit inside the extraction window is a
	configuration error, not something to silently clamp."""

	with pytest.raises(ValueError, match="attr_window"):
		_run_attribute(tmp_path, in_window=500, attr_window=600)
