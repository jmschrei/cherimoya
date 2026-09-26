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


def _run_attribute(tmp_path, n_loci=3, signal_groups=(1, 2), **overrides):
	"""Run the command against a stubbed extract_loci, saturation
	mutagenesis and DeepLIFT/SHAP, returning what each of them saw."""

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
		captured['wrapper'] = model
		start, end = kwargs['start'], kwargs['end']
		return torch.zeros(X.shape[0], 4, end - start)

	def fake_dls(model, X, **kwargs):
		captured['dls'] = kwargs
		captured['wrapper'] = model
		# Each column holds its own index, so the slice `attribute`
		# saves can be located in the full-window output.
		return torch.arange(X.shape[-1]).float().expand(X.shape).clone()

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
				'saturation_mutagenesis', side_effect=fake_sm), \
			mock.patch('tangermeme.deep_lift_shap.deep_lift_shap',
				side_effect=fake_dls):
		model_cls.load.return_value = mock.MagicMock(n_control_tracks=0,
			signal_groups=list(signal_groups))
		attribute.run(argparse.Namespace(parameters=str(path)))
		captured['load'] = model_cls.load.call_args.kwargs

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

	captured = _run_attribute(tmp_path, in_window=2114, attr_window=600,
		algorithm='saturation_mutagenesis')

	assert captured['sm']['end'] - captured['sm']['start'] == 600
	assert captured['sm']['start'] == 2114 // 2 - 300


def test_default_attr_window_matches_the_old_hard_coded_slice(tmp_path):
	"""400bp centred in a 2114bp window, i.e. positions 857-1257, which
	is what the hard-coded ``mid - 200, mid + 200`` produced."""

	captured = _run_attribute(tmp_path, algorithm='saturation_mutagenesis')

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



@pytest.mark.parametrize("output, wrapper", [("counts", "LogCountWrapper"),
	("profile", "ProfileWrapper")])
@pytest.mark.parametrize("algorithm", ["deep_lift_shap",
	"saturation_mutagenesis"])
def test_group_reaches_the_wrapper(tmp_path, output, wrapper, algorithm):
	"""`group` selects one signal group of a multi-group model as the
	attribution target, under either algorithm."""

	captured = _run_attribute(tmp_path, output=output, group=1,
		algorithm=algorithm)

	assert type(captured['wrapper']).__name__ == wrapper
	assert captured['wrapper'].group == 1


def test_default_group_is_the_first(tmp_path):
	"""The default attributes group 0, which for a single-group model is
	the whole output."""

	captured = _run_attribute(tmp_path)

	assert captured['wrapper'].group == 0


def test_null_group_attributes_every_group_under_ism(tmp_path):
	"""`group: null` leaves the wrapper unselected; saturation
	mutagenesis then averages over the count head's outputs."""

	captured = _run_attribute(tmp_path, group=None,
		algorithm='saturation_mutagenesis')

	assert captured['wrapper'].group is None


def test_null_group_on_multi_group_counts_raises_for_deep_lift_shap(
		tmp_path):
	"""DeepLIFT attributes one output, so a multi-group count head with
	no group selected is a configuration error rather than a silent
	choice of group 0."""

	with pytest.raises(ValueError, match="group"):
		_run_attribute(tmp_path, group=None)


def test_null_group_is_allowed_for_one_output(tmp_path):
	"""A single-group count head, or the profile wrapper, is already one
	output, so no group is needed."""

	(tmp_path / 'c').mkdir()
	(tmp_path / 'p').mkdir()
	counts = _run_attribute(tmp_path / 'c', group=None, signal_groups=(1,))
	profile = _run_attribute(tmp_path / 'p', group=None, output='profile')

	assert counts['wrapper'].group is None
	assert profile['wrapper'].group is None


def test_group_out_of_range_raises(tmp_path):
	"""A group index the model does not have is a configuration error."""

	with pytest.raises(ValueError, match="group"):
		_run_attribute(tmp_path, group=2)


##


def test_default_algorithm_is_deep_lift_shap(tmp_path):
	captured = _run_attribute(tmp_path)

	assert 'dls' in captured
	assert 'sm' not in captured


def test_deep_lift_shap_defaults_match_bpnet_lite(tmp_path):
	"""Hypothetical attributions against 20 dinucleotide shuffles, with
	Cherimoya's DeepLIFT rules registered."""

	from cherimoya.deep_lift_shap import attribution_ops

	kwargs = _run_attribute(tmp_path)['dls']

	assert kwargs['hypothetical'] is True
	assert kwargs['n_shuffles'] == 20
	assert kwargs['warning_threshold'] == 1e-3
	assert kwargs['random_state'] == 0
	assert kwargs['additional_nonlinear_ops'] == attribution_ops()


def test_deep_lift_shap_keys_are_forwarded(tmp_path):
	kwargs = _run_attribute(tmp_path, n_shuffles=5, warning_threshold=0.1,
		random_state=3, batch_size=7, dtype='bfloat16')['dls']

	assert kwargs['n_shuffles'] == 5
	assert kwargs['warning_threshold'] == 0.1
	assert kwargs['random_state'] == 3
	assert kwargs['batch_size'] == 7
	assert kwargs['dtype'] == 'bfloat16'
	assert kwargs['device'] == 'cpu'


def test_deep_lift_shap_saves_the_centred_slice(tmp_path):
	"""DeepLIFT attributes the whole window; the saved array is the same
	centred `attr_window` slice saturation mutagenesis is given."""

	captured = _run_attribute(tmp_path, in_window=2114, attr_window=600)

	# The stub writes each column's index into it.
	assert captured['attr'][0, 0, 0] == 2114 // 2 - 300
	assert captured['attr'][0, 0, -1] == 2114 // 2 + 299


@pytest.mark.parametrize("algorithm", ["deep_lift_shap",
	"saturation_mutagenesis"])
def test_both_algorithms_write_the_same_shapes(tmp_path, algorithm):
	"""`cherimoya seqlets` reads either output, so both write
	(n, 4, attr_window) arrays."""

	captured = _run_attribute(tmp_path, n_loci=3, attr_window=600,
		algorithm=algorithm)

	assert captured['ohe'].shape == (3, 4, 600)
	assert captured['attr'].shape == (3, 4, 600)


def test_default_batch_size_fits_deep_lift_shap(tmp_path):
	"""For DeepLIFT the batch is sequence-reference pairs, each run
	forward and backward; 512 of them at 2114bp need over 100 GB."""

	assert _run_attribute(tmp_path)['dls']['batch_size'] == 64


@pytest.mark.parametrize("algorithm, compiled", [("deep_lift_shap", False),
	("saturation_mutagenesis", True)])
def test_compile_applies_only_to_ism(tmp_path, algorithm, compiled):
	"""DeepLIFT's hooks defeat the compiled forward, so the model is
	loaded uncompiled for it even with `compile: true`."""

	captured = _run_attribute(tmp_path, algorithm=algorithm, compile=True)

	assert captured['load']['compile'] is compiled


def test_unknown_algorithm_raises(tmp_path):
	with pytest.raises(ValueError, match="algorithm"):
		_run_attribute(tmp_path, algorithm='integrated_gradients')
