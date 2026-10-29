"""Tests of the fast PISA engine against tangermeme's `pisa`, on the CPU.

Both sides get the first 3 of the tiny models' sequences, 1 for config C,
and K = 3 references per sequence, of five kinds: those both draw themselves, the
dinucleotide shuffles `pisa` draws, with the same seed; random one-hots,
r = x, where every rule takes the plain gradient; and x with 3 mutated
positions, where the rescale rule's inputs differ by less than 1e-6 at some
position of every GELU and it takes the gradient there. Each attributes
the profile logits of the config's last channel: channel 0 of A and C, and
channel 2 of B, the second strand of its stranded group, with one control
track. The oracle is `pisa` on the CPU at batch_size 16, which leaves an
edge chunk of output positions. The engine's chunks of 16 output positions
at most are, for B's 102 positions, 7 chunks of 15, the last padded.

* Hypothetical attributions, per sequence and output position: <= 1e-9
  relative L2 in float64 and 1e-5 in float32.
* Convergence deltas in float64: |delta_ours - delta_tangermeme| <= 1e-9 x
  max(1, |y_x - y_r|), with tangermeme's deltas computed from its
  multipliers.
* Chunks of 1 and 7 output positions give the results of the default chunks
  to 1e-12.
"""

import functools
import re
import warnings

import pytest
import torch
import tangermeme.pisa

from tangermeme.ersatz import dinucleotide_shuffle

from cherimoya import _pisa_engine as fpisa
from cherimoya.deep_lift_shap import attribution_ops
from cherimoya.pisa import _ProfileChannelWrapper

from . import deep_lift_shap_engine_helpers as tiny


K = 3
N = {"A": 3, "B": 3, "C": 1}
SEED = 1
F64 = torch.float64
CHANNEL = {"A": 0, "B": 2, "C": 0}

HYP_TOL = {torch.float32: 1e-5, F64: 1e-9}
DELTA_FP64 = 1e-9
Y_FP64 = 1e-12
CHUNK_TOL = 1e-12
RULES_MIN = 0.1

# Every kind of reference on the small configs A and B; drawn ones, the
# default path's, on C in float32, whose float64 oracle takes a minute.
PARITY_CASES = [(config, dtype, kind) for config in ("A", "B")
	for dtype, kind in ((torch.float32, "drawn"), (torch.float32, "random"),
	(F64, "drawn"), (F64, "shuffled"), (F64, "random"), (F64, "identical"),
	(F64, "mutated"))] + [("C", torch.float32, "drawn")]


@functools.cache
def _case(config, kind):
	"""A config's first N sequences, int8 (N, 4, L), and K references of one
	kind, float32 (N, K, 4, L), or None for those both sides draw."""

	X = tiny.sequences(config)[:N[config]]
	if kind == "drawn":
		return X, None

	if kind == "shuffled":
		return X, tiny.pisa_references(X, K, SEED)

	if kind == "random":
		return X, tiny.random_references(X, K, seed=3)

	if kind == "identical":
		return X, tiny.identical_references(X, K)

	if kind == "mutated":
		return X, tiny.mutated_references(X, K, seed=5)

	raise ValueError(kind)


def _engine(config, dtype, **kwargs):
	"""An Engine on a fresh tiny model, attributing the config's channel."""

	model = tiny.tiny_model(config, dtype=dtype)
	kwargs.setdefault("n_shuffles", K)
	kwargs.setdefault("random_state", SEED)
	return fpisa.Engine(model, channel=CHANNEL[config], **kwargs)


@functools.cache
def _ours(config, dtype, kind, chunk=None):
	X, R = _case(config, kind)
	return _engine(config, dtype, outputs_per_chunk=chunk).run(X, R)


def _wrapped(config, dtype):
	model = tiny.tiny_model(config, dtype=dtype)
	return _ProfileChannelWrapper(model, CHANNEL[config])


def _oracle(config, dtype, kind, raw_outputs=False):
	"""tangermeme's `pisa` on the CPU, as `cherimoya.pisa.pisa`'s default
	path calls it, with hypothetical=True."""

	X, R = _case(config, kind)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="Convergence deltas too high",
			category=RuntimeWarning)
		return tangermeme.pisa.pisa(_wrapped(config, dtype), X.to(dtype),
			references=dinucleotide_shuffle if R is None else R.to(dtype), n_shuffles=K, batch_size=16, hypothetical=True,
			raw_outputs=raw_outputs, random_state=SEED,
			additional_nonlinear_ops=attribution_ops(), device="cpu")


def _dy(result):
	"""|y_x - y_r| per pair and output position, (N, K, n_outputs), in
	float64."""

	return (result.y_x[:, None] - result.y_r).double().abs()


##


@pytest.mark.parametrize("config, dtype, kind", PARITY_CASES)
def test_attributions_match_tangermeme(config, dtype, kind):
	X, _ = _case(config, kind)
	ours = _ours(config, dtype, kind)
	theirs = _oracle(config, dtype, kind)
	n_outputs = X.shape[-1] - 2 * tiny.tiny_model(config).trimming
	assert ours.attr.shape == theirs.shape == (len(X), n_outputs, *X.shape[1:])
	assert ours.attr.dtype == ours.deltas.dtype == ours.y_x.dtype == dtype
	assert ours.deltas.shape == ours.y_r.shape == (len(X), K, n_outputs)
	err = tiny.rows_rel_l2(ours.attr.flatten(0, 1), theirs.flatten(0, 1))
	assert float(err.max()) <= HYP_TOL[dtype]


def test_the_references_are_tangermemes():
	"""The references the engine draws are those `pisa` draws, bit for bit:
	every sequence's with the seed random_state."""

	X, _ = _case("A", "drawn")
	engine = _engine("A", torch.float32)
	ours = torch.stack([engine._references(X, None, i, K)
		for i in range(len(X))])
	theirs = tangermeme.pisa.pisa(_wrapped("A", torch.float32), X.float(),
		n_shuffles=K, random_state=SEED, return_references=True,
		warning_threshold=float("inf"),
		additional_nonlinear_ops=attribution_ops(), device="cpu").references
	assert ours.dtype == torch.int8
	assert torch.equal(ours, theirs.to(torch.int8))
	assert torch.equal(ours.float(), tiny.pisa_references(X, K, SEED))


@pytest.mark.parametrize("config", ["A", "B"])
def test_outputs_and_deltas_match_tangermeme(config):
	"""In float64: the outputs are the module's, the deltas are tangermeme's
	expression over its own multipliers, and those multipliers are at least
	10% from the plain gradient for every pair and output position, so the
	tests here exercise the rules."""

	X, _ = _case(config, "shuffled")
	R = tiny.pisa_references(X, K, SEED).to(F64)
	wrapped = _wrapped(config, F64)
	ours = _ours(config, F64, "shuffled")
	m = _oracle(config, F64, "shuffled", raw_outputs=True)

	x = X.to(F64)
	with torch.no_grad():
		y_x = wrapped(x)
		y_r = wrapped(R.flatten(0, 1)).view(len(X), K, -1)

	assert float(((ours.y_x - y_x).abs() / y_x.abs().clamp_min(1)).max()) \
		<= Y_FP64
	assert float(((ours.y_r - y_r).abs() / y_r.abs().clamp_min(1)).max()) \
		<= Y_FP64

	deltas = ((y_x[:, None] - y_r) - ((x[:, None, None] - R[:, :, None])
		* m).sum(dim=(-2, -1))).abs()
	err = (ours.deltas - deltas).abs() / _dy(ours).clamp_min(1)
	assert float(err.max()) <= DELTA_FP64

	gradients = []
	with torch.enable_grad():
		for j in range(y_x.shape[1]):
			leaf = x.clone().requires_grad_()
			gradients.append(torch.autograd.grad(wrapped(leaf)[:, j].sum(),
				leaf)[0])

	gradient = torch.stack(gradients, dim=1)[:, None].expand_as(m)
	rel = tiny.rows_rel_l2(m.flatten(0, 2), gradient.flatten(0, 2))
	assert float(rel.min()) >= RULES_MIN


@pytest.mark.parametrize("config", ["A", "B"])
def test_results_do_not_depend_on_the_chunk_size(config):
	"""Chunks of 1 and 7 output positions against the default chunks of 15,
	the last of 7 padded. Rows are independent in every operation, so only
	reassociation can differ."""

	base = _ours(config, F64, "shuffled")
	for chunk in (1, 7):
		ours = _ours(config, F64, "shuffled", chunk)
		y_ref = torch.cat([base.y_x[:, None], base.y_r], dim=1).double()
		y = torch.cat([ours.y_x[:, None], ours.y_r], dim=1).double()
		errs = {
			"attr": float(tiny.rows_rel_l2(ours.attr.flatten(0, 1),
				base.attr.flatten(0, 1)).max()),
			"deltas": float(((ours.deltas - base.deltas).abs()
				/ _dy(base).clamp_min(1)).max()),
			"y": float(((y - y_ref).abs() / y_ref.abs().clamp_min(1)).max()),
		}

		assert max(errs.values()) <= CHUNK_TOL, (chunk, errs)


def test_the_chunks_cover_the_outputs_evenly():
	"""B is the size of the equal chunks that cover the output positions
	with as few chunks of at most the given size as there can be."""

	engine = _engine("A", torch.float32)
	sizes = {}
	for given in (1, 7, 15, 16, 150, 1000):
		engine.outputs_per_chunk = given
		sizes[given] = engine._chunk_size(150, K, 256)

	assert sizes == {1: 1, 7: 7, 15: 15, 16: 15, 150: 150, 1000: 150}


def test_run_warns_once():
	"""One RuntimeWarning per run when a delta exceeds the threshold, none for
	a threshold of None or of the largest delta."""

	X, R = _case("A", "shuffled")

	def caught(**kwargs):
		with warnings.catch_warnings(record=True) as records:
			warnings.simplefilter("always")
			result = _engine("A", torch.float32, **kwargs).run(X, R)

		return result, [str(w.message) for w in records
			if issubclass(w.category, RuntimeWarning)]

	result, messages = caught(warning_threshold=0.0)
	assert len(messages) == 1
	assert re.fullmatch(r"Convergence deltas too high: \d+ of 1350 pairs and "
		r"outputs > 0, the largest \S+", messages[0])

	assert caught(warning_threshold=None)[1] == []
	assert caught(warning_threshold=float(result.deltas.max()))[1] == []
