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
from cherimoya.deep_lift_shap import _agreement
from cherimoya.deep_lift_shap import _make_wrapper
from cherimoya.deep_lift_shap import attribution_ops
from cherimoya.deep_lift_shap import deep_lift_shap

from . import deep_lift_shap_engine_helpers as tiny


K = 3
REL_L2_MAX = 1e-5


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
		_make_wrapper(model, output, group), X, n_shuffles=K,
		references=references, random_state=0, hypothetical=hypothetical,
		warning_threshold=float("inf"),
		additional_nonlinear_ops=attribution_ops(), device="cpu")


class _NotCherimoya(torch.nn.Module):
	"""A module that runs a Cherimoya without being one, as the module
	torch.compile returns does."""

	def __init__(self, model):
		super().__init__()
		self.model = model
		self.signal_groups = model.signal_groups
		self.n_control_tracks = model.n_control_tracks

	def forward(self, *args):
		return self.model(*args)


##


@pytest.mark.parametrize("hypothetical", [False, True])
@pytest.mark.parametrize("output", ["logcounts", "profile"])
def test_fast_false_is_tangermemes_deep_lift_shap(model, X, engines, output,
	hypothetical):
	ours = _ours(model, X, output=output, hypothetical=hypothetical,
		fast=False)
	assert torch.equal(ours, _stock(model, X, output, 0, hypothetical))
	assert engines == []


@pytest.mark.parametrize("hypothetical, group", [(False, 0), (True, 0),
	(True, None)])
def test_the_fast_path_agrees_with_the_default_path(model, X, forced, engines,
	hypothetical, group):
	ours = _ours(model, X, hypothetical=hypothetical, group=group)
	assert len(engines) == 1
	theirs = _ours(model, X, hypothetical=hypothetical, group=group,
		fast=False)
	assert ours.shape == theirs.shape == X.shape
	assert ours.dtype == theirs.dtype == torch.float32
	assert _agreement(ours, theirs)[1] <= REL_L2_MAX
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
	"bfloat16", "reference function", "not a Cherimoya"])
def test_what_takes_the_default_path_in_one_call(model, X, forced, engines,
	case):
	kwargs = {"four sequences": {}, "profile": {"output": "profile"},
		"fast=False": {"fast": False}, "bfloat16": {"dtype": "bfloat16"},
		"reference function": {"references": lambda X, n, random_state=None:
		dinucleotide_shuffle(X, n=n, random_state=random_state)},
		"not a Cherimoya": {}}[case]
	Xs = X[:4] if case == "four sequences" else X
	attributed = _NotCherimoya(model) if case == "not a Cherimoya" else model
	calls = []
	stock = tangermeme.deep_lift_shap.deep_lift_shap

	def counting(*args, **kwargs):
		calls.append(len(args[1]))
		return stock(*args, **kwargs)

	with pytest.MonkeyPatch.context() as patch:
		patch.setattr(tangermeme.deep_lift_shap, "deep_lift_shap", counting)
		_ours(attributed, Xs, **kwargs)

	assert engines == []
	assert calls == [len(Xs)]


def test_given_references_reach_both_paths(model, X, forced, engines):
	references = tiny.random_references(X, K, seed=3)
	ours = _ours(model, X, references=references)
	assert len(engines) == 1
	theirs = _ours(model, X, references=references, fast=False)
	assert _agreement(ours, theirs)[1] <= REL_L2_MAX
	assert _agreement(ours, _ours(model, X, fast=False))[1] > 1e-3


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

		with pytest.raises(ValueError, match="output must be"):
			_ours(grouped, X, output="counts", fast=fast)


@pytest.mark.parametrize("where", [1, 9])
def test_an_unknown_character_raises_on_either_path(model, X, forced,
	engines, where):
	"""An N in a sequence the check attributes (1) or in one only the fast
	path would (9) raises tangermeme's error, with no fallback warning."""

	with_n = X.clone()
	with_n[where, :, 7] = 0
	errors = []
	for fast in (True, False):
		with warnings.catch_warnings():
			warnings.simplefilter("error", RuntimeWarning)
			with pytest.raises(ValueError) as error:
				_ours(model, with_n, fast=fast)

		errors.append(str(error.value))

	assert errors[0] == errors[1]
	assert engines == []


def test_the_agreement_bounds():
	"""Each sequence's Pearson correlation and relative L2 distance; a NaN
	fails both bounds."""

	gen = torch.Generator().manual_seed(0)
	stock = torch.randn(6, 4, 50, generator=gen, dtype=torch.float64)
	assert _agreement(stock, stock) == (pytest.approx(1), 0.0)

	noise = torch.randn(stock.shape, generator=gen, dtype=torch.float64)
	noise *= (stock.flatten(1).norm(dim=1) / noise.flatten(1).norm(dim=1))[
		:, None, None]
	assert _agreement(stock + 2e-2 * noise, stock)[1] == pytest.approx(2e-2)

	# A large constant offset: the centred values decorrelate (Pearson below
	# 0.9999) while the relative L2 stays under 1e-2.
	offset = stock.clone()
	offset[2] = 100.0 + 0.1 * torch.randn(offset[2].shape, generator=gen,
		dtype=torch.float64)
	ours = offset.clone()
	ours[2] += 0.02 * torch.randn(ours[2].shape, generator=gen,
		dtype=torch.float64)
	pearson, rel_l2 = _agreement(ours, offset)
	assert pearson < 0.9999 and rel_l2 <= 1e-2

	nan = stock.clone()
	nan[1, 0, 0] = float("nan")
	pearson, rel_l2 = _agreement(nan, stock)
	assert not pearson >= 0.9999 and not rel_l2 <= 1e-2


@pytest.mark.parametrize("fast", [True, False])
def test_warning_threshold_none_gives_no_threshold_warning(model, X, forced,
	fast):
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		_ours(model, X, warning_threshold=None, fast=fast)

	assert not [w for w in caught if "Convergence deltas too high"
		in str(w.message)]
