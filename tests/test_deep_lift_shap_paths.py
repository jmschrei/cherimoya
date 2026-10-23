"""Tests of `cherimoya.deep_lift_shap.deep_lift_shap`, which attributes with
tangermeme's `deep_lift_shap` (the default path) or with the engine in
`cherimoya._deep_lift_shap_engine` (the fast path), on the CPU, with the tiny
models of `deep_lift_shap_engine_helpers`.

Every job here would take the default path less than thirty seconds, so the
tests that need the fast path set `_FAST_MIN_SECONDS` to 0 (`forced`).
`engines` counts the engines the function builds.
"""

import warnings

import pytest
import torch
import tangermeme.deep_lift_shap

from tangermeme.ersatz import dinucleotide_shuffle

from cherimoya._deep_lift_shap_engine import Engine
from cherimoya.deep_lift_shap import attribution_ops
from cherimoya.deep_lift_shap import deep_lift_shap

from . import deep_lift_shap_engine_helpers as tiny


K = 3
REL_L2_MAX = 1e-5


@pytest.fixture
def forced(monkeypatch):
	monkeypatch.setattr("cherimoya.deep_lift_shap._FAST_MIN_SECONDS", 0.0)


@pytest.fixture
def engines(monkeypatch):
	built = []

	def counting(*args, **kwargs):
		built.append(Engine(*args, **kwargs))
		return built[-1]

	monkeypatch.setattr("cherimoya.deep_lift_shap.Engine", counting)
	return built


@pytest.fixture(scope="module")
def model():
	return tiny.tiny_model("A")


@pytest.fixture(scope="module")
def X():
	return tiny.sequences("A")


def _ours(model, X, **kwargs):
	kwargs.setdefault("n_shuffles", K)
	kwargs.setdefault("random_state", 0)
	kwargs.setdefault("warning_threshold", None)
	return deep_lift_shap(model, X, device="cpu", **kwargs)


def _stock(model, X, output="logcounts", group=0, hypothetical=False,
	references=dinucleotide_shuffle):
	return tangermeme.deep_lift_shap.deep_lift_shap(
		tiny.wrapper(model, output, group), X, n_shuffles=K,
		references=references, random_state=0, hypothetical=hypothetical,
		warning_threshold=float("inf"),
		additional_nonlinear_ops=attribution_ops(), device="cpu")


def _rows_rel_l2(a, b):
	a, b = a.double().flatten(1), b.double().flatten(1)
	return float(((a - b).norm(dim=1) / b.norm(dim=1)).max())


##


@pytest.mark.parametrize("hypothetical", [False, True])
@pytest.mark.parametrize("output", ["logcounts", "profile"])
def test_fast_false_is_tangermemes_deep_lift_shap(model, X, engines, output,
	hypothetical):
	ours = _ours(model, X, output=output, hypothetical=hypothetical,
		fast=False)
	assert torch.equal(ours, _stock(model, X, output, 0, hypothetical))
	assert engines == []


@pytest.mark.parametrize("hypothetical", [False, True])
def test_the_fast_path_agrees_with_the_default_path(model, X, forced, engines,
	hypothetical):
	ours = _ours(model, X, hypothetical=hypothetical)
	assert len(engines) == 1
	theirs = _ours(model, X, hypothetical=hypothetical, fast=False)
	assert ours.shape == theirs.shape == X.shape
	assert ours.dtype == theirs.dtype == torch.float32
	assert _rows_rel_l2(ours, theirs) <= REL_L2_MAX
	if not hypothetical:
		assert bool((ours[X == 0] == 0).all())


def test_a_quick_job_takes_the_default_path(model, X, engines):
	"""At the default threshold every job here is quick. The first four
	sequences are attributed in two calls of two, against the references
	tangermeme draws, and the attributions are tangermeme's bit for bit."""

	ours = _ours(model, X)
	assert engines == []
	assert torch.equal(ours, _stock(model, X))


# bfloat16 rounding is above tangermeme's 1% check on these tiny models.
@pytest.mark.filterwarnings("ignore:Convergence deltas are")
@pytest.mark.parametrize("case", ["four sequences", "profile", "fast=False",
	"bfloat16", "reference function"])
def test_what_takes_the_default_path_in_one_call(model, X, forced, engines,
	case):
	kwargs = {"four sequences": {}, "profile": {"output": "profile"},
		"fast=False": {"fast": False}, "bfloat16": {"dtype": "bfloat16"},
		"reference function": {"references": lambda X, n, random_state=None:
		dinucleotide_shuffle(X, n=n, random_state=random_state)}}[case]
	Xs = X[:4] if case == "four sequences" else X
	calls = []
	stock = tangermeme.deep_lift_shap.deep_lift_shap

	def counting(*args, **kwargs):
		calls.append(len(args[1]))
		return stock(*args, **kwargs)

	with pytest.MonkeyPatch.context() as patch:
		patch.setattr(tangermeme.deep_lift_shap, "deep_lift_shap", counting)
		_ours(model, Xs, **kwargs)

	assert engines == []
	assert calls == [len(Xs)]


def test_given_references_reach_both_paths(model, X, forced, engines):
	references = tiny.random_references(X, K, seed=3)
	ours = _ours(model, X, references=references)
	assert len(engines) == 1
	theirs = _ours(model, X, references=references, fast=False)
	assert _rows_rel_l2(ours, theirs) <= REL_L2_MAX
	assert _rows_rel_l2(ours, _ours(model, X, fast=False)) > 1e-3


def test_random_state_none_draws_one_seed(model, X, forced, engines):
	ours = _ours(model, X, random_state=None)
	assert len(engines) == 1 and engines[0].random_state is not None
	assert ours.shape == X.shape and bool(torch.isfinite(ours).all())


@pytest.mark.parametrize("failure", ["disagrees", "raises"])
def test_a_failing_fast_path_falls_back_with_a_warning(model, X, forced,
	monkeypatch, failure):
	"""The fallback gives what a quick job gives: the default path's
	attributions, from the same calls."""

	with pytest.MonkeyPatch.context() as patch:
		patch.setattr("cherimoya.deep_lift_shap._FAST_MIN_SECONDS", 1e30)
		expected = _ours(model, X)

	if failure == "disagrees":
		run = Engine.run

		def scaled(self, *args, **kwargs):
			result = run(self, *args, **kwargs)
			return result._replace(attr=result.attr * 1.1)

		monkeypatch.setattr(Engine, "run", scaled)
		message = "its attributions of the first 4 sequences"
	else:
		def broken(self, *args, **kwargs):
			raise RuntimeError("no engine\nsecond line")

		monkeypatch.setattr(Engine, "__init__", broken)
		message = "it raised RuntimeError: no engine\\. Pass fast=False"

	with pytest.warns(RuntimeWarning, match="deep_lift_shap is using the "
			"default path because the fast path " + message):
		ours = _ours(model, X)

	assert torch.equal(ours, expected)


def test_a_multi_group_model_needs_a_group_on_either_path():
	grouped = tiny.tiny_model("B")
	X = tiny.sequences("B")
	for fast in (True, False):
		with pytest.raises(ValueError, match="attributes one output"):
			_ours(grouped, X, group=None, fast=fast)


@pytest.mark.parametrize("fast", [True, False])
def test_warning_threshold_none_gives_no_threshold_warning(model, X, forced,
	fast):
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		_ours(model, X, warning_threshold=None, fast=fast)

	assert not [w for w in caught if "Convergence deltas too high"
		in str(w.message)]
