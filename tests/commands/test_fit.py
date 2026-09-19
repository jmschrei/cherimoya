# This file is named for cherimoya_cli/commands/fit.py but does not cover all
# of it. It checks that parameters reach the right downstream calls and that
# _split_parameters routes the optimizer; fit.run's execution and its error
# paths are untested. It was test_fit_wiring.py until the rename, which carried
# that scope in the name -- widen the file rather than trusting the name.
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


def _run_capturing_peak_generator(path):
	"""Run fit.run and return the kwargs PeakGenerator was called with.

	Execution stops at that call, which is far enough to see every
	parameter the sampler is given but short of any real IO.
	"""

	from cherimoya_cli.commands import fit as fit_cmd

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_peak_generator(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	with mock.patch("cherimoya.io.PeakGenerator",
			side_effect=fake_peak_generator):
		try:
			fit_cmd.run(argparse.Namespace(parameters=str(path)))
		except _StopFit:
			pass
		except Exception:
			if not captured:
				raise

	assert captured, "PeakGenerator was never called"
	return captured


def test_default_fit_parameters_random_state_is_zero():
	"""Training is seeded by default. The nested pipeline copy stays
	None so that `_extract_set` lets the top-level value through — a 0
	there would shadow whatever the user set at the top level."""

	from cherimoya_cli.defaults import (
		default_fit_parameters,
		default_pipeline_parameters,
	)
	assert default_fit_parameters['random_state'] == 0
	assert default_pipeline_parameters['random_state'] == 0
	assert default_pipeline_parameters['fit_parameters']['random_state'] is None


def test_pipeline_random_state_reaches_the_fit_json():
	"""The top-level pipeline seed must survive `_extract_set` into the
	fit JSON, since that is the dict the fit step actually reads."""

	from cherimoya_cli.defaults import (
		default_fit_parameters,
		default_pipeline_parameters,
	)
	from cherimoya_cli.utils import _extract_set

	parameters = dict(default_pipeline_parameters)
	parameters['random_state'] = 7

	extracted = _extract_set(parameters, default_fit_parameters,
		'fit_parameters')
	assert extracted['random_state'] == 7


def test_fit_forwards_random_state_to_peak_generator(fit_json):
	"""The sampler's draw order is the half of reproducibility that was
	already wired; confirm an explicit seed still reaches it."""

	import json as _json

	cfg = _json.loads(open(fit_json).read())
	cfg['random_state'] = 42
	open(fit_json, 'w').write(_json.dumps(cfg))

	captured = _run_capturing_peak_generator(fit_json)
	assert captured.get('random_state') == 42


def test_fit_accepts_a_json_without_random_state(tmp_path):
	"""merge_parameters rejects a missing key whose default is None, so
	before the default became 0 a hand-written JSON that left the seed
	out failed outright rather than falling back to it."""

	from cherimoya_cli.defaults import default_fit_parameters

	cfg = dict(default_fit_parameters)
	del cfg['random_state']
	cfg['sequences'] = 'fake.fa'
	cfg['loci'] = 'fake.bed'
	cfg['negatives'] = 'fake_negatives.bed'
	cfg['signals'] = ['fake.bw']
	cfg['name'] = 'fit_random_state_default_test'
	cfg['device'] = 'cpu'

	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))

	captured = _run_capturing_peak_generator(path)
	assert captured.get('random_state') == 0


def test_fit_draws_and_announces_a_null_random_state(fit_json, capsys):
	"""A null seed means "pick one and tell me", not "stay unseeded".
	The drawn value has to be printed regardless of `verbose`, because
	a run that dies before the evaluate JSON is written leaves no other
	record of it."""

	import json as _json

	cfg = _json.loads(open(fit_json).read())
	cfg['random_state'] = None
	cfg['verbose'] = False
	open(fit_json, 'w').write(_json.dumps(cfg))

	captured = _run_capturing_peak_generator(fit_json)

	drawn = captured.get('random_state')
	assert isinstance(drawn, int), (
		"a null random_state must be resolved to an integer before the "
		"sampler is built; got {!r}".format(drawn))

	out = capsys.readouterr().out
	assert "Drew random_state={}".format(drawn) in out


def test_fit_seeds_the_model_initialization(fit_json):
	"""The seed has to reach the model as well as the sampler — the
	initialization is the larger source of run-to-run variance, and it
	was the half that nothing seeded before."""

	import torch

	from cherimoya_cli.commands import fit as fit_cmd

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_extract_loci(**kwargs):
		# (sequences, signals); controls are absent in the fixture JSON.
		return torch.zeros(1, 4, 16), torch.zeros(1, 1, 8)

	def fake_model(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	with mock.patch("cherimoya.io.PeakGenerator", return_value=object()), \
			mock.patch("tangermeme.io.extract_loci",
				side_effect=fake_extract_loci), \
			mock.patch("cherimoya.Cherimoya", side_effect=fake_model):
		try:
			fit_cmd.run(argparse.Namespace(parameters=fit_json))
		except _StopFit:
			pass
		except Exception:
			if not captured:
				raise

	assert captured, "Cherimoya was never constructed"
	assert captured.get('random_state') == 0
