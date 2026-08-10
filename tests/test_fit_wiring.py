"""Wiring tests for `cherimoya fit` — confirms parameters flow into the
right downstream calls without actually training."""

import argparse
import json
from unittest import mock

import pytest


@pytest.fixture
def fit_json(tmp_path):
	"""Write a minimal JSON that satisfies merge_parameters' required keys."""
	from cherimoya_cli.defaults import default_fit_parameters

	# Include every key from the defaults so merge_parameters' "missing
	# required" check passes (it errors when a key is absent and its
	# default is None outside a small whitelist). Then override the
	# values we care about for the test.
	cfg = dict(default_fit_parameters)
	cfg['sequences'] = 'fake.fa'
	cfg['loci'] = 'fake.bed'
	cfg['negatives'] = 'fake_negatives.bed'
	cfg['signals'] = ['fake.bw']
	cfg['name'] = 'fit_wiring_test'
	cfg['device'] = 'cpu'
	cfg['num_workers'] = 3   # the value we want to verify is forwarded
	cfg['batch_size'] = 16

	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))
	return str(path)


def test_fit_forwards_num_workers_to_peak_generator(fit_json):
	"""fit.run must pass parameters['num_workers'] to PeakGenerator. We
	stop execution immediately after the call by raising from a fake
	PeakGenerator and then inspect the kwargs."""

	from cherimoya_cli.commands import fit as fit_cmd

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_peak_generator(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	# Block import-time side effects of the heavy modules `fit.run`
	# pulls in. We only need to exercise the wiring up to PeakGenerator.
	with mock.patch("cherimoya.io.PeakGenerator", side_effect=fake_peak_generator):
		try:
			fit_cmd.run(argparse.Namespace(parameters=fit_json))
		except _StopFit:
			pass
		except Exception as e:
			# Any error AFTER PeakGenerator was called is fine — the
			# point of the test is whether it received the right kwargs.
			if not captured:
				raise

	assert captured, "PeakGenerator was never called"
	assert captured.get('num_workers') == 3, (
		"fit.run did not forward num_workers; got {!r}"
		.format(captured.get('num_workers'))
	)


def test_default_fit_parameters_default_num_workers_is_one():
	from cherimoya_cli.defaults import (
		default_fit_parameters,
		default_pipeline_parameters,
	)
	assert default_fit_parameters['num_workers'] == 1
	assert default_pipeline_parameters['fit_parameters']['num_workers'] == 1


def test_fit_forwards_grouped_signals_to_peak_generator(tmp_path):
	"""When the JSON gives a grouped signals spec, fit.run must
	forward the inferred signal_groups (and control_groups) to
	PeakGenerator. Without this the structured form would be flattened
	to "all unstranded" downstream."""

	from cherimoya_cli.commands import fit as fit_cmd
	from cherimoya_cli.defaults import default_fit_parameters

	cfg = dict(default_fit_parameters)
	cfg['sequences'] = 'fake.fa'
	cfg['loci'] = 'fake.bed'
	cfg['negatives'] = 'fake_negatives.bed'
	# One unstranded ATAC group + one stranded TF group; per-group
	# counts mean signal_groups=[1, 2].
	cfg['signals'] = ['atac.bw', ['ctcf.+.bw', 'ctcf.-.bw']]
	cfg['controls'] = [['ctl.+.bw', 'ctl.-.bw']]
	cfg['name'] = 'fit_wiring_groups_test'
	cfg['device'] = 'cpu'

	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_peak_generator(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	with mock.patch(
			"cherimoya.io.PeakGenerator", side_effect=fake_peak_generator):
		try:
			fit_cmd.run(argparse.Namespace(parameters=str(path)))
		except _StopFit:
			pass
		except Exception:
			if not captured:
				raise

	assert captured.get('signal_groups') == [1, 2], (
		"signal_groups not forwarded; got {!r}".format(
			captured.get('signal_groups')))
	assert captured.get('control_groups') == [2], (
		"control_groups not forwarded; got {!r}".format(
			captured.get('control_groups')))


def test_fit_does_not_flatten_signals_for_downstream_evaluate(tmp_path):
	"""fit.run hands ``parameters['signals']`` to PeakGenerator and
	(at the end of training) deepcopies the same dict into the
	evaluate JSON. If fit silently re-writes ``signals`` to its flat
	form, a stranded pair ``[[+, -]]`` becomes ``[+, -]`` in the
	evaluate JSON, which then re-parses as two *unstranded* channels
	— the same bug the grouping API was added to prevent. Pin the
	contract by snapshotting what PeakGenerator actually receives."""

	from cherimoya_cli.commands import fit as fit_cmd
	from cherimoya_cli.defaults import default_fit_parameters

	original_signals = [['ctcf.+.bw', 'ctcf.-.bw']]
	cfg = dict(default_fit_parameters)
	cfg['sequences'] = 'fake.fa'
	cfg['loci'] = 'fake.bed'
	cfg['negatives'] = 'fake_negatives.bed'
	cfg['signals'] = original_signals
	cfg['name'] = str(tmp_path / 'fit_eval_roundtrip')
	cfg['device'] = 'cpu'

	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_peak_generator(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	with mock.patch(
			"cherimoya.io.PeakGenerator", side_effect=fake_peak_generator):
		try:
			fit_cmd.run(argparse.Namespace(parameters=str(path)))
		except _StopFit:
			pass
		except Exception:
			if not captured:
				raise

	# PeakGenerator must see the *structured* signals form. If fit had
	# pre-flattened it the assertion below would fail with
	# `signals == ['ctcf.+.bw', 'ctcf.-.bw']` (two unstranded tracks).
	assert captured.get('signals') == original_signals, (
		"fit flattened the structured signals form before PeakGenerator: "
		"got {!r}".format(captured.get('signals')))


# --------- optimizer routing ---------------------------------------------
#
# These import `_split_parameters` from the fit command rather than
# restating the rule, so an edit to the routing logic is caught here
# instead of silently diverging from a copy.

def _routing(model):
	"""Return (name -> buckets) plus the raw lists, for readable asserts."""

	from cherimoya_cli.commands.fit import _split_parameters

	muon, adam, lw = _split_parameters(model)
	by_id = {}
	for bucket, params in (('muon', muon), ('adam', adam), ('lw', lw)):
		for p in params:
			by_id.setdefault(id(p), []).append(bucket)

	names = {}
	for name, p in model.named_parameters():
		names[name] = by_id.get(id(p), [])

	return names, muon, adam, lw


@pytest.fixture
def routed_model():
	import torch
	from cherimoya import Cherimoya

	torch.manual_seed(0)
	return Cherimoya(n_filters=16, n_layers=3, signal_groups=[1, 2],
		n_control_tracks=2, verbose=False)


def test_fit_routes_projection_weights_to_muon(routed_model):
	"""The MLP projections inside each block are what Muon is for."""

	names, muon, adam, lw = _routing(routed_model)

	projections = [n for n in names
		if n.endswith('linear1.weight') or n.endswith('linear2.weight')]
	assert len(projections) == 6, "expected 2 projections per block"

	for name in projections:
		assert names[name] == ['muon'], f"{name} -> {names[name]}"


def test_fit_routes_depthwise_conv_weight_to_adamw(routed_model):
	"""``conv_weight`` is 2D and matches "weight", so it would land in
	Muon without the explicit exclusion. It sits on the depth-wise path
	rather than being a projection matmul, so it belongs in AdamW. The
	exclusion is a substring test, which is what lets it keep working
	now that the parameter lives on the ``conv`` submodule."""

	names, muon, adam, lw = _routing(routed_model)

	conv = [n for n in names if n.endswith('conv_weight')]
	assert len(conv) == 3, f"expected one per block, got {conv}"

	for name in conv:
		assert names[name] == ['adam'], f"{name} -> {names[name]}"


def test_fit_routes_loss_balancing_weights_to_sgd(routed_model):
	"""lw0/lw1 are matched by exact name, not by shape."""

	names, muon, adam, lw = _routing(routed_model)

	assert names.get('lw0') == ['lw']
	assert names.get('lw1') == ['lw']
	assert len(lw) == 2


def test_fit_routes_output_head_to_adamw(routed_model):
	"""``linear.weight`` is the output head and is excluded by exact
	name, so it must not be swept into Muon with the projections."""

	names, _, _, _ = _routing(routed_model)

	assert names['linear.weight'] == ['adam']


def test_fit_assigns_every_parameter_exactly_once(routed_model):
	"""The three buckets must partition the parameters -- no parameter
	trained by two optimizers, and none left untrained."""

	names, muon, adam, lw = _routing(routed_model)

	duplicated = {n: b for n, b in names.items() if len(b) > 1}
	assert not duplicated, f"parameters in more than one optimizer: {duplicated}"

	unrouted = [n for n, b in names.items() if not b]
	assert not unrouted, f"parameters in no optimizer: {unrouted}"

	total = len(muon) + len(adam) + len(lw)
	assert total == len(names), f"{total} routed vs {len(names)} parameters"


def test_fit_routing_has_no_duplicate_parameter_objects(routed_model):
	"""Guards against a parameter being registered twice on the model
	(e.g. an alias accidentally re-registering it), which would hand the
	same tensor to an optimizer twice and double its updates."""

	import torch

	model = routed_model
	names, muon, adam, lw = _routing(model)

	for bucket, params in (('muon', muon), ('adam', adam), ('lw', lw)):
		ids = [id(p) for p in params]
		assert len(ids) == len(set(ids)), f"{bucket} lists a parameter twice"

	# remove_duplicate=False exposes any tensor reachable under two names.
	all_named = list(model.named_parameters(remove_duplicate=False))
	assert len(all_named) == len(list(model.named_parameters()))

	# ...and each optimizer must accept its bucket without complaint.
	for params in (muon, adam, lw):
		torch.optim.AdamW(params)


def test_fit_routing_is_stable_without_control_tracks():
	"""Routing must not depend on the control-track configuration."""

	import torch
	from cherimoya import Cherimoya

	torch.manual_seed(0)
	model = Cherimoya(n_filters=16, n_layers=3, signal_groups=[1],
		n_control_tracks=0, verbose=False)
	names, muon, adam, lw = _routing(model)

	assert not [n for n, b in names.items() if len(b) != 1]
	for name in names:
		if name.endswith('linear1.weight') or name.endswith('linear2.weight'):
			assert names[name] == ['muon'], f"{name} -> {names[name]}"
		elif name.endswith('conv_weight'):
			assert names[name] == ['adam'], f"{name} -> {names[name]}"
