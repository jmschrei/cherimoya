"""Tests of `cherimoya.pisa.pisa`, which attributes with tangermeme's `pisa`
(the default path) or with the engine in `cherimoya._pisa_engine` (the
fast path), on the CPU, with the tiny models of
`deep_lift_shap_engine_helpers`.

Every job here would take the default path less than thirty seconds, so the
tests that need the fast path set `_FAST_MIN_SECONDS` to 0 (`forced`).
`engines` counts the engines the function builds.
"""

import warnings

import pytest
import torch
import tangermeme.pisa

from tangermeme.ersatz import dinucleotide_shuffle

from cherimoya._pisa_engine import Engine
from cherimoya.deep_lift_shap import _agreement
from cherimoya.deep_lift_shap import attribution_ops
from cherimoya.pisa import _ProfileChannelWrapper
from cherimoya.pisa import pisa

from . import deep_lift_shap_engine_helpers as tiny


K = 3
N = 5
REL_L2_MAX = 1e-5


@pytest.fixture(scope="module")
def model():
	return tiny.tiny_model("A")


@pytest.fixture(scope="module")
def X():
	return tiny.sequences("A")[:N]


@pytest.fixture
def forced(monkeypatch):
	"""`pisa` takes the fast path for any job, however quick, where it
	applies."""

	monkeypatch.setattr("cherimoya.pisa._FAST_MIN_SECONDS", 0.0)


@pytest.fixture
def engines(monkeypatch):
	"""The fast-path engines `pisa` builds, in order."""

	built = []

	def counting(*args, **kwargs):
		built.append(Engine(*args, **kwargs))
		return built[-1]

	monkeypatch.setattr("cherimoya.pisa.Engine", counting)
	return built


def _ours(model, X, **kwargs):
	kwargs.setdefault("n_shuffles", K)
	kwargs.setdefault("random_state", 0)
	kwargs.setdefault("warning_threshold", None)
	return pisa(model, X, device="cpu", **kwargs)


def _stock(model, X, channel=0, hypothetical=False,
	references=dinucleotide_shuffle):
	return tangermeme.pisa.pisa(_ProfileChannelWrapper(model, channel),
		X.float(), n_shuffles=K, references=references, random_state=0,
		hypothetical=hypothetical, warning_threshold=float("inf"),
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
def test_fast_false_is_tangermemes_pisa(model, X, engines, hypothetical):
	ours = _ours(model, X, hypothetical=hypothetical, fast=False)
	assert torch.equal(ours, _stock(model, X, hypothetical=hypothetical))
	assert engines == []


@pytest.mark.parametrize("hypothetical", [False, True])
def test_the_fast_path_agrees_with_the_default_path(model, X, forced, engines,
	hypothetical):
	ours = _ours(model, X, hypothetical=hypothetical)
	assert len(engines) == 1
	theirs = _ours(model, X, hypothetical=hypothetical, fast=False)
	assert ours.shape == theirs.shape == (N, 150, 4, 256)
	assert ours.dtype == theirs.dtype == torch.float32
	assert _agreement(ours, theirs)[1] <= REL_L2_MAX
	if not hypothetical:
		assert bool((ours[(X == 0)[:, None].expand_as(ours)] == 0).all())


def test_a_channel_of_a_stranded_group_with_controls(forced, engines):
	"""Config B's channel 2, the second strand of its stranded group, with
	its control track fed as zeros, on both paths."""

	stranded = tiny.tiny_model("B")
	Xb = tiny.sequences("B")[:N]
	ours = _ours(stranded, Xb, channel=2)
	assert len(engines) == 1 and engines[0].channel == 2
	theirs = _stock(stranded, Xb, channel=2)
	assert torch.equal(_ours(stranded, Xb, channel=2, fast=False), theirs)
	assert _agreement(ours, theirs)[1] <= REL_L2_MAX
	assert _agreement(ours, _stock(stranded, Xb, channel=1))[1] > 1e-3


def test_a_quick_job_takes_the_default_path(model, X, engines):
	"""At the default threshold every job here is quick. The first two
	sequences are attributed in two calls of one, against the references
	tangermeme draws, and the attributions are tangermeme's bit for bit."""

	ours = _ours(model, X)
	assert engines == []
	assert torch.equal(ours, _stock(model, X))


@pytest.mark.parametrize("case", ["two sequences", "fast=False", "float64",
	"reference function", "not a Cherimoya"])
def test_what_takes_the_default_path_in_one_call(model, X, forced, engines,
	case):
	kwargs = {"fast=False": {"fast": False}, "reference function": {
		"references": lambda X, n, random_state=None: dinucleotide_shuffle(X,
		n=n, random_state=random_state)}}.get(case, {})
	Xs = X[:2] if case == "two sequences" else X
	attributed = {"not a Cherimoya": _NotCherimoya(model), "float64":
		tiny.tiny_model("A", dtype=torch.float64)}.get(case, model)
	calls = []
	stock = tangermeme.pisa.pisa

	def counting(*args, **kwargs):
		calls.append(len(args[1]))
		return stock(*args, **kwargs)

	with pytest.MonkeyPatch.context() as patch:
		patch.setattr(tangermeme.pisa, "pisa", counting)
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
	assert ours.shape == (N, 150, 4, 256) and bool(torch.isfinite(ours).all())


@pytest.mark.parametrize("failure", ["disagrees", "raises"])
def test_a_failing_fast_path_falls_back_with_a_warning(model, X, forced,
	monkeypatch, failure):
	"""The fallback gives what a quick job gives: the default path's
	attributions, from the same calls."""

	with pytest.MonkeyPatch.context() as patch:
		patch.setattr("cherimoya.pisa._FAST_MIN_SECONDS", 1e30)
		expected = _ours(model, X)

	if failure == "disagrees":
		run = Engine.run

		def scaled(self, *args, **kwargs):
			result = run(self, *args, **kwargs)
			return result._replace(attr=result.attr * 1.1)

		monkeypatch.setattr(Engine, "run", scaled)
		message = "its attributions of the first 2 sequences"
	else:
		def broken(self, *args, **kwargs):
			raise RuntimeError("no engine\nsecond line")

		monkeypatch.setattr(Engine, "__init__", broken)
		message = "it raised RuntimeError: no engine\\. Pass fast=False"

	with pytest.warns(RuntimeWarning, match="pisa is using the default path "
			"because the fast path " + message):
		ours = _ours(model, X)

	assert torch.equal(ours, expected)


@pytest.mark.parametrize("channel", [-1, 3, 1.0, True, None])
def test_a_channel_must_index_the_profile_channels(channel):
	stranded = tiny.tiny_model("B")
	Xb = tiny.sequences("B")[:N]
	for fast in (True, False):
		with pytest.raises(ValueError, match=r"channel must be an int in "
				r"\[0, 3\) for a model with signal_groups=\[1, 2\]"):
			_ours(stranded, Xb, channel=channel, fast=fast)


@pytest.mark.parametrize("where", [1, 3])
def test_an_unknown_character_raises_on_either_path(model, X, forced,
	engines, where):
	"""An N in a sequence the check attributes (1) or in one only the fast
	path would (3) raises tangermeme's error, with no fallback warning."""

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


@pytest.mark.parametrize("fast", [True, False])
def test_warning_threshold_none_gives_no_threshold_warning(model, X, forced,
	fast):
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		_ours(model, X, warning_threshold=None, fast=fast)

	assert not [w for w in caught if "Convergence deltas too high"
		in str(w.message)]
