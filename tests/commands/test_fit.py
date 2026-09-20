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


def test_default_fit_parameters_disable_early_stopping():
	"""Early stopping is off by default in both the fit defaults and the
	fit block of the pipeline defaults, matching Cherimoya.fit's own
	``early_stopping=None``. A non-None default here would cut the cosine
	learning rate schedule -- laid out over ``max_epochs`` -- short."""

	from cherimoya_cli.defaults import (
		default_fit_parameters,
		default_pipeline_parameters,
	)
	assert default_fit_parameters['early_stopping'] is None
	assert default_pipeline_parameters['fit_parameters']['early_stopping'] is None


def test_fit_json_without_early_stopping_merges_to_none(tmp_path):
	"""A hand-written fit JSON that omits ``early_stopping`` must merge to
	None rather than raising 'Must provide value', since merge_parameters
	rejects a missing key whose default is None unless it is whitelisted."""

	from cherimoya_cli.defaults import default_fit_parameters
	from cherimoya_cli.utils import merge_parameters

	cfg = dict(default_fit_parameters)
	cfg['sequences'] = 'fake.fa'
	cfg['loci'] = 'fake.bed'
	cfg['negatives'] = 'fake_negatives.bed'
	cfg['signals'] = ['fake.bw']
	del cfg['early_stopping']

	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))

	assert merge_parameters(str(path), default_fit_parameters)['early_stopping'] is None


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


# --------- the minimum step count ----------------------------------------
#
# These import `_max_epochs_for_min_steps` from the fit command for the same
# reason the routing tests import `_split_parameters`: the rule lives in one
# place and an edit to it is caught here rather than diverging from a copy.

def test_default_min_total_steps_is_twenty_thousand():
	"""An epoch is one pass over the peaks, so `max_epochs` alone buys a
	step count proportional to peak count. The floor is on by default in
	both the fit defaults and the fit block of the pipeline defaults."""

	from cherimoya_cli.defaults import (
		default_fit_parameters,
		default_pipeline_parameters,
	)
	assert default_fit_parameters['min_total_steps'] == 20000
	assert (default_pipeline_parameters['fit_parameters']['min_total_steps']
		== 20000)


def test_min_steps_extends_a_short_run():
	"""14 batches of peaks over 20 epochs is 280 steps, far under the
	floor, so the run is extended to the epoch count that reaches it."""

	from cherimoya_cli.commands.fit import _max_epochs_for_min_steps

	assert _max_epochs_for_min_steps(20, 14, 20000) == 1429
	assert 1429 * 14 >= 20000


def test_min_steps_rounds_up_rather_than_landing_short():
	"""The division has to round up: 20 epochs of 999 is 19,980, and an
	epoch count that lands one step under the floor would defeat it."""

	from cherimoya_cli.commands.fit import _max_epochs_for_min_steps

	assert _max_epochs_for_min_steps(20, 999, 20000) == 21
	assert 21 * 999 >= 20000


def test_min_steps_leaves_a_long_enough_run_alone():
	"""An accessibility-shaped experiment has thousands of batches per
	epoch and already clears the floor, so nothing changes. Equality
	counts as clearing it."""

	from cherimoya_cli.commands.fit import _max_epochs_for_min_steps

	assert _max_epochs_for_min_steps(20, 2700, 20000) == 20
	assert _max_epochs_for_min_steps(20, 1000, 20000) == 20


def test_min_steps_null_disables_the_floor():
	"""`min_total_steps: null` is the escape hatch for a short run, e.g. a
	smoke test that sets max_epochs to 1."""

	from cherimoya_cli.commands.fit import _max_epochs_for_min_steps

	assert _max_epochs_for_min_steps(20, 14, None) == 20
	assert _max_epochs_for_min_steps(1, 14, None) == 1


def test_min_steps_tolerates_an_empty_training_set():
	"""`len(training_data)` can be zero if every locus was filtered out.
	Dividing by it would raise before the clearer error downstream."""

	from cherimoya_cli.commands.fit import _max_epochs_for_min_steps

	assert _max_epochs_for_min_steps(20, 0, 20000) == 20


def test_fit_json_without_min_total_steps_merges_to_the_default(tmp_path):
	"""A fit JSON written before this parameter existed still runs, and
	picks up the floor rather than failing on the missing key."""

	from cherimoya_cli.defaults import default_fit_parameters
	from cherimoya_cli.utils import merge_parameters

	cfg = dict(default_fit_parameters)
	del cfg['min_total_steps']
	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))

	assert merge_parameters(str(path), default_fit_parameters
		)['min_total_steps'] == 20000


# --------- fixed loss weights ---------------------------------------------

def test_default_loss_weights_is_none():
	"""The Kendall weights stay the default. Turning the fixed weights on
	is opt-in, in both the fit defaults and the fit block of the pipeline
	defaults."""

	from cherimoya_cli.defaults import (
		default_fit_parameters,
		default_pipeline_parameters,
	)
	assert default_fit_parameters['loss_weights'] is None
	assert default_pipeline_parameters['fit_parameters']['loss_weights'] is None


def test_fit_json_without_loss_weights_merges_to_none(tmp_path):
	"""``loss_weights`` defaults to None, and ``merge_parameters`` rejects a
	missing key whose default is None unless it is whitelisted. A fit JSON
	written before this parameter existed must still run."""

	from cherimoya_cli.defaults import default_fit_parameters
	from cherimoya_cli.utils import merge_parameters

	cfg = dict(default_fit_parameters)
	del cfg['loss_weights']
	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))

	assert merge_parameters(str(path), default_fit_parameters
		)['loss_weights'] is None


def test_fit_json_loss_weights_survives_as_a_pair(tmp_path):
	"""JSON has no tuple, so the pair arrives as a list; ``fit`` unpacks it
	either way and the values must not be coerced or reordered."""

	from cherimoya_cli.defaults import default_fit_parameters
	from cherimoya_cli.utils import merge_parameters

	cfg = dict(default_fit_parameters)
	cfg['loss_weights'] = [1.333, 0.274]
	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))

	merged = merge_parameters(str(path), default_fit_parameters)
	w0, w1 = merged['loss_weights']
	assert (w0, w1) == (1.333, 0.274)


# --------- the verbose loss-balance line ----------------------------------
#
# `_loss_balance_summary` is imported directly for the same reason as
# `_max_epochs_for_min_steps` above: `fit.run` is not executed by these
# tests, so the line it prints is checked where the rule lives.

def test_loss_balance_summary_reports_the_sgd_optimizer_by_default():
	"""With the Kendall weights in use, `lw_optimizer` is training `lw0`
	and `lw1`, so its hyperparameters are what the run is governed by."""

	from cherimoya_cli.commands.fit import _loss_balance_summary

	summary = _loss_balance_summary(None, 0.01, 0.0, 0.9)
	assert summary == "SGD Optimizer (lw): lr=0.01, wd=0.0, momentum=0.9"


def test_loss_balance_summary_reports_the_fixed_weights():
	"""When `loss_weights` is given, `lw0` and `lw1` stop receiving
	gradient, so the SGD hyperparameters describe an optimizer that does
	nothing. The constants actually balancing the two losses are reported
	instead, and none of the SGD values leak into the line."""

	from cherimoya_cli.commands.fit import _loss_balance_summary

	summary = _loss_balance_summary((1.333, 0.274), 0.01, 0.0, 0.9)
	assert "1.333" in summary
	assert "0.274" in summary
	assert "SGD" not in summary
	assert "momentum" not in summary


def test_loss_balance_summary_accepts_the_pair_as_a_list():
	"""JSON has no tuple, so `parameters['loss_weights']` arrives as a
	list and must be reported the same way a tuple is."""

	from cherimoya_cli.commands.fit import _loss_balance_summary

	assert (_loss_balance_summary([1.333, 0.274], 0.01, 0.0, 0.9)
		== _loss_balance_summary((1.333, 0.274), 0.01, 0.0, 0.9))


@pytest.mark.parametrize("loss_weights,expected,forbidden", [
	(None, "SGD Optimizer (lw): lr=", "Fixed Loss Weights"),
	([1.333, 0.274], "Fixed Loss Weights: profile=1.333, count=0.274",
		"SGD Optimizer (lw)"),
])
def test_fit_banner_names_the_loss_balancing_in_force(fit_json, capsys,
	loss_weights, expected, forbidden):
	"""End to end through `fit.run`: with `verbose` on, the banner must
	name whichever scheme the run uses -- the `lw_*` optimizer when the
	weights are learned, the constants when they are fixed and that
	optimizer is inert. Training is cut off at `model.fit`, which is the
	call the banner immediately precedes."""

	import json as _json

	import torch

	from cherimoya_cli.commands import fit as fit_cmd

	cfg = _json.loads(open(fit_json).read())
	cfg['loss_weights'] = loss_weights
	cfg['verbose'] = True
	cfg['n_layers'] = 2
	cfg['n_filters'] = 8
	open(fit_json, 'w').write(_json.dumps(cfg))

	class _StopFit(Exception):
		pass

	def fake_extract_loci(**kwargs):
		return torch.zeros(1, 4, 16), torch.zeros(1, 1, 8)

	class _FakeDataset:
		peak_sequences = torch.zeros(4, 4, 16)
		negative_sequences = torch.zeros(4, 4, 16)

	class _FakeLoader(list):
		dataset = _FakeDataset()

	with mock.patch("cherimoya.io.PeakGenerator",
				return_value=_FakeLoader([None] * 4)), \
			mock.patch("tangermeme.io.extract_loci",
				side_effect=fake_extract_loci), \
			mock.patch("cherimoya.Cherimoya.fit", side_effect=_StopFit()):
		with pytest.raises(_StopFit):
			fit_cmd.run(argparse.Namespace(parameters=fit_json))

	out = capsys.readouterr().out
	assert expected in out
	assert forbidden not in out
