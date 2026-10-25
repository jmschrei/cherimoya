"""Tests of the fast DeepLIFT/SHAP engine against tangermeme's
`deep_lift_shap`, on the CPU.

Both sides get the same 13 sequences (12 random, 1 AT-rich) and the same
injected references, K = 5 per sequence, of four kinds: dinucleotide
shuffles, random one-hots, r = x, where every rule takes the plain
gradient, and x with 3 mutated positions, where the rescale rule's inputs
differ by less than 1e-6 at some position of every GELU and it takes the
gradient there. The oracle is `deep_lift_shap` as `cherimoya attribute`
calls it, on the CPU, at batch_size 7, which splits a sequence's pairs
across batches. The engine runs in steps of 8 sequences, so 13 sequences
fill one step and pad the second with 3 copies of the last one.

* Hypothetical attributions, per sequence: <= 1e-9 relative L2 in float64
  and 1e-5 in float32.
* Convergence deltas in float64: |delta_ours - delta_tangermeme| <= 1e-9 x
  max(1, |y_x - y_r|), with tangermeme's deltas computed from its
  multipliers.
* Steps of 3 and 8 sequences give the results of steps of 1 to 1e-12.

The engine's CPU forward is the model's CPU path, which normalizes in
float32 even in a float64 model, as the oracle's forward does.
"""

import functools
import re
import warnings

import pytest
import torch

from cherimoya import _deep_lift_shap_engine as fdls
from cherimoya.deep_lift_shap import _make_wrapper

from . import deep_lift_shap_engine_helpers as tiny


K = 5
F64 = torch.float64
N_SEQUENCES = tiny.N_RANDOM + tiny.N_AT_RICH

HYP_TOL = {torch.float32: 1e-5, F64: 1e-9}
DELTA_FP64 = 1e-9
Y_FP64 = 1e-12
STEP_TOL = 1e-12
RULES_MIN = 0.1

# Every kind of reference on the small configs A and B; shuffled ones, the
# default path's, on C, whose float64 oracle takes seconds per call.
PARITY_CASES = [(config, dtype, kind) for config in ("A", "B")
	for dtype, kind in ((torch.float32, "shuffled"), (torch.float32, "random"),
	(F64, "shuffled"), (F64, "random"), (F64, "identical"), (F64, "mutated"))
	] + [("C", torch.float32, "shuffled"), ("C", F64, "shuffled")]


@functools.cache
def _case(config, kind):
	"""A config's 13 sequences, int8 (13, 4, L), and K references of one
	kind, float32 (13, K, 4, L)."""

	X = tiny.sequences(config)
	if kind == "shuffled":
		return X, tiny.shuffled_references(X, K)

	if kind == "random":
		return X, tiny.random_references(X, K, seed=3)

	if kind == "identical":
		return X, tiny.identical_references(X, K)

	if kind == "mutated":
		return X, tiny.mutated_references(X, K, seed=5)

	raise ValueError(kind)


def _engine(config, dtype, **kwargs):
	"""An Engine on a fresh tiny model, attributing the config's group."""

	model = tiny.tiny_model(config, dtype=dtype)
	return fdls.Engine(model, group=tiny.CONFIGS[config].group, **kwargs)


@functools.cache
def _ours(config, dtype, kind, step=None):
	X, R = _case(config, kind)
	return _engine(config, dtype, seqs_per_step=step).run(X, R)


def _wrapped(config, dtype):
	model = tiny.tiny_model(config, dtype=dtype)
	return _make_wrapper(model, "logcounts", tiny.CONFIGS[config].group)


def _dy(result):
	"""|y_x - y_r| per pair, (N, K), in float64."""

	return (result.y_x[:, None] - result.y_r).double().abs()


##


@pytest.mark.parametrize("config, dtype, kind", PARITY_CASES)
def test_attributions_match_tangermeme(config, dtype, kind):
	X, R = _case(config, kind)
	ours = _ours(config, dtype, kind)
	theirs = tiny.oracle(_wrapped(config, dtype), X, R, dtype)
	assert ours.attr.shape == theirs.shape == X.shape
	assert ours.attr.dtype == ours.deltas.dtype == ours.y_x.dtype == dtype
	assert ours.deltas.shape == ours.y_r.shape == (N_SEQUENCES, K)
	assert float(tiny.rows_rel_l2(ours.attr, theirs).max()) <= HYP_TOL[dtype]


@pytest.mark.parametrize("config", ["A", "B"])
def test_outputs_and_deltas_match_tangermeme(config):
	"""In float64: the outputs are the module's, the deltas are tangermeme's
	expression over its own multipliers, and those multipliers are at least
	10% from the plain gradient for every pair, so the tests here exercise
	the rules."""

	X, R = _case(config, "shuffled")
	wrapped = _wrapped(config, F64)
	ours = _ours(config, F64, "shuffled")
	m = tiny.oracle(wrapped, X, R, F64, raw_outputs=True)

	x, r = X.to(F64), R.to(F64)
	with torch.no_grad():
		y_x = wrapped(x)[:, 0]
		y_r = wrapped(r.flatten(0, 1))[:, 0].view(N_SEQUENCES, K)

	assert float(((ours.y_x - y_x).abs() / y_x.abs().clamp_min(1)).max()) \
		<= Y_FP64
	assert float(((ours.y_r - y_r).abs() / y_r.abs().clamp_min(1)).max()) \
		<= Y_FP64

	deltas = ((y_x[:, None] - y_r) - ((x[:, None] - r) * m).sum(dim=(-2,
		-1))).abs()
	err = (ours.deltas - deltas).abs() / _dy(ours).clamp_min(1)
	assert float(err.max()) <= DELTA_FP64

	with torch.enable_grad():
		leaf = x.clone().requires_grad_()
		gradient, = torch.autograd.grad(wrapped(leaf)[:, 0].sum(), leaf)

	rel = tiny.rows_rel_l2(m.flatten(0, 1), gradient.repeat_interleave(K, 0))
	assert float(rel.min()) >= RULES_MIN


@pytest.mark.parametrize("config", ["A", "B"])
def test_results_do_not_depend_on_the_step_size(config):
	"""Steps of 3 and 8 sequences against steps of 1, for 13 sequences, the
	last step padded with copies of the last sequence. Rows are independent
	in every operation, so only reassociation can differ."""

	base = _ours(config, F64, "shuffled", 1)
	for step in (3, 8):
		ours = _ours(config, F64, "shuffled", step)
		y_ref = torch.cat([base.y_x[:, None], base.y_r], dim=1).double()
		y = torch.cat([ours.y_x[:, None], ours.y_r], dim=1).double()
		errs = {
			"attr": float(tiny.rows_rel_l2(ours.attr, base.attr).max()),
			"deltas": float(((ours.deltas - base.deltas).abs()
				/ _dy(base).clamp_min(1)).max()),
			"y": float(((y - y_ref).abs() / y_ref.abs().clamp_min(1)).max()),
		}

		assert max(errs.values()) <= STEP_TOL, (step, errs)


def test_run_warns_once_per_check(monkeypatch):
	"""One RuntimeWarning per run when a delta exceeds the threshold, none
	for a threshold of None, and one when the mean delta exceeds
	RELATIVE_DELTA_MAX of the mean output difference."""

	X, R = _case("A", "shuffled")

	def caught(**kwargs):
		with warnings.catch_warnings(record=True) as records:
			warnings.simplefilter("always")
			result = _engine("A", torch.float32, **kwargs).run(X, R)

		return result, [str(w.message) for w in records
			if issubclass(w.category, RuntimeWarning)]

	result, messages = caught(warning_threshold=0.0)
	assert len(messages) == 1
	assert re.fullmatch(r"Convergence deltas too high: \d+ of 65 pairs > 0, "
		r"the largest \S+", messages[0])

	assert caught(warning_threshold=None)[1] == []
	assert caught(warning_threshold=float(result.deltas.max()))[1] == []

	relative = float(result.deltas.double().mean() / _dy(result).mean())
	assert 0 < relative < fdls.RELATIVE_DELTA_MAX
	monkeypatch.setattr(fdls, "RELATIVE_DELTA_MAX", relative / 2)
	messages = caught(warning_threshold=None)[1]
	assert len(messages) == 1
	assert re.fullmatch(r"Convergence deltas are \S+% of the output "
		r"differences over 65 pairs, above \S+%: the attributions do not sum "
		r"to the change in the output", messages[0])
