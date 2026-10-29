"""Tests of the fast PISA engine on a GPU: the accuracy of its CUDA passes,
a run at the default model's size, and the recovery from running out of
memory.

As in `test_deep_lift_shap_engine_cuda`, every fp32 path is held to the
float64 engine on the GPU, run eagerly, within bounds on the relative L2
distance of each sequence's attributions, over all its output positions:
1e-4 at the median and 2e-4 at the max.

`tests/conftest.py` disables dynamo for the suite, under which
`torch.compile` returns the function unchanged; the tests that compile turn
it back on for this module (`dynamo_enabled`) and check that they did.
"""

import re

import pytest
import torch

from unittest import mock

from cherimoya import Cherimoya
from cherimoya import _pisa_engine as fpisa

from . import deep_lift_shap_engine_helpers as tiny


pytestmark = pytest.mark.cuda

K = 3
N = 4
CHANNEL = {"A": 0, "B": 2, "C": 0}
ACCURACY_MEDIAN, ACCURACY_MAX = 1e-4, 2e-4
MEMORY_FACTOR = 1.3
RERUN_MAX = 1e-6
REAL_K, REAL_B, REAL_PEAKS, REAL_LENGTH = 20, 50, 2, 2114
RETAINED_MAX = 100e6


@pytest.fixture(scope="module")
def dynamo_enabled():
	"""Undo conftest's suite-wide dynamo opt-out for this module's compiled
	engines. torch.compile reads it when it wraps a function, so the engines
	are built inside this fixture's scope."""

	patch = pytest.MonkeyPatch()
	patch.setenv("TORCHDYNAMO_DISABLE", "0")
	with torch._dynamo.config.patch(disable=False):
		torch._dynamo.reset()
		yield
		torch._dynamo.reset()

	patch.undo()


@pytest.fixture
def strict_fp32():
	"""Strict IEEE float32 matrix products and convolutions in the test."""

	cudnn = torch.backends.cudnn
	previous = torch.get_float32_matmul_precision(), cudnn.allow_tf32
	torch.set_float32_matmul_precision("highest")
	cudnn.allow_tf32 = False
	yield
	torch.set_float32_matmul_precision(previous[0])
	cudnn.allow_tf32 = previous[1]


def _engine(config, dtype=torch.float32, eager=False, **kwargs):
	model = tiny.tiny_model(config, dtype=dtype)
	kwargs.setdefault("device", "cuda")
	kwargs.setdefault("n_shuffles", K)
	engine = fpisa.Engine(model, channel=CHANNEL[config], **kwargs)
	if eager:
		engine.forward, engine.rule_terms, engine.backward, engine.epilogue = (
			fpisa.forward_save, fpisa.rule_terms, fpisa.backward_pass,
			fpisa.epilogue)

	return engine


def _inaccurate(name, attr, truth):
	"""The violations of the accuracy bounds, each sequence's relative L2
	distance from the truth at most `ACCURACY_MEDIAN` at the median and
	`ACCURACY_MAX` at the max."""

	r = tiny.rows_rel_l2(attr, truth)
	median, largest = float(r.median()), float(r.max())
	if median <= ACCURACY_MEDIAN and largest <= ACCURACY_MAX:
		return []

	return ["{}: median {:.3g}, max {:.3g}".format(name, median, largest)]


##


@pytest.mark.triton
@pytest.mark.parametrize("config", list(tiny.CONFIGS))
def test_the_cuda_passes_are_as_accurate_as_the_cpu(config, dynamo_enabled,
	strict_fp32):
	"""In strict fp32 the eager CUDA passes, the shift-add convolution and the
	rule's statistics, and the compiled ones are within the bounds the CPU
	path, the model's own, meets."""

	from torch._dynamo.utils import counters

	X = tiny.sequences(config)[:N]
	R = tiny.pisa_references(X, K)
	truth = _engine(config, torch.float64, eager=True).run(X, R).attr
	cpu = _engine(config, device="cpu").run(X, R).attr
	eager = _engine(config, eager=True).run(X, R).attr
	counters.clear()
	compiled = _engine(config).run(X, R).attr
	assert counters["stats"]["unique_graphs"] >= 2  # it did compile

	failures = (_inaccurate("cpu", cpu, truth) + _inaccurate("eager", eager,
		truth) + _inaccurate("compiled", compiled, truth))
	assert not failures, failures


@pytest.mark.triton
def test_a_run_at_the_default_size(dynamo_enabled):
	"""Two runs of one engine on the default architecture, 2 sequences with
	20 references each, in chunks of 50 output positions.

	* The forward, the rule terms, the backward and the epilogue compile
	  once each, with no graph break, and not again on the second run. The 9
	  blocks unroll into one graph each way, each with 2 cuBLAS matrix
	  products per block, and no Triton matrix product or convolution.
	* The second run raises on any implicit host synchronization
	  (torch.cuda.set_sync_debug_mode("error")): the host waits only on each
	  sequence's event.
	* The compiling first run gives its traced sequence back, and the second
	  allocates at most 1.3 x the estimate the automatic chunk size relies
	  on, `sequence_bytes` and B x `output_bytes`, above what was allocated
	  before it.
	* The two runs agree, to cuDNN's non-repeatable input gradient.
	"""

	from torch._dynamo.utils import counters
	from torch._inductor.graph import GraphLowering

	model = tiny.redraw_weights(Cherimoya(compile=False, random_state=0,
		verbose=False), 0)
	X = tiny.random_onehot(REAL_PEAKS, REAL_LENGTH, seed=11)
	R = tiny.pisa_references(X, REAL_K)
	engine = fpisa.Engine(model, device="cuda", outputs_per_chunk=REAL_B)

	codes = []
	torch.cuda.empty_cache()
	at_start = torch.cuda.memory_allocated()
	counters.clear()
	with mock.patch.object(GraphLowering, "save_output_code", codes.append):
		first = engine.run(X, R)

	assert counters["stats"]["unique_graphs"] == 4
	assert dict(counters["graph_break"]) == {}

	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()
	before = torch.cuda.memory_allocated()
	torch.cuda.set_sync_debug_mode("error")
	try:
		with torch._dynamo.config.patch(error_on_recompile=True):
			second = engine.run(X, R)
	finally:
		torch.cuda.set_sync_debug_mode(0)

	peak = torch.cuda.max_memory_allocated() - before
	assert counters["stats"]["unique_graphs"] == 4

	assert len(codes) == 4
	mm = [len(re.findall(r"extern_kernels\.(?:mm|bmm|addmm)\(", code))
		for code in codes]
	assert sorted(mm) == [0, 0, 18, 18]
	for code in codes:
		assert "tl.dot" not in code and "triton_tem" not in code
		assert not re.search(r"convolution|conv1d|cudnn", code,
			flags=re.IGNORECASE)

	estimate = (fpisa.sequence_bytes(REAL_K, REAL_LENGTH, 128, 256, 9)
		+ REAL_B * fpisa.output_bytes(REAL_K, REAL_LENGTH, 128, 256))
	assert before - at_start <= RETAINED_MAX
	assert peak <= MEMORY_FACTOR * estimate

	for name in ("y_x", "y_r"):
		assert torch.equal(getattr(first, name), getattr(second, name))

	err = tiny.rows_rel_l2(second.attr.flatten(0, 1), first.attr.flatten(0, 1))
	assert float(err.max()) <= RERUN_MAX
	assert bool(torch.isfinite(first.attr).all())


@pytest.mark.parametrize("at_call, calls", [(1, [8, 4, 4, 4]),
	(2, [8, 8, 4, 4])])
def test_out_of_memory_halves_the_chunk_and_retries(at_call, calls):
	"""An out-of-memory error in the first or the second of 3 sequences, in
	chunks of 8 output positions, with the references drawn: the sequences
	before it keep their chunks of 8, the rest run in chunks of 4 and give
	what an engine planned at that size gives, to cuDNN's non-repeatable
	rounding, and every sequence is attributed once. The deltas are rounding
	themselves, so they are held to the scale of |y_x - y_r|."""

	X = tiny.sequences("C")[:3]
	engine = _engine("C", outputs_per_chunk=8)
	seen = []
	sequence = engine._sequence

	def failing(inputs, K, B, n_outputs):
		seen.append(B)
		if len(seen) == at_call:
			raise torch.cuda.OutOfMemoryError("injected by the test")

		return sequence(inputs, K, B, n_outputs)

	engine._sequence = failing
	got = engine.run(X)
	assert seen == calls

	kept = at_call - 1
	by_8 = _engine("C", outputs_per_chunk=8).run(X)
	by_4 = _engine("C", outputs_per_chunk=4).run(X)
	for theirs, part in ((by_8, slice(None, kept)), (by_4, slice(kept, None))):
		if part.stop == 0:
			continue

		dy = (theirs.y_x[part, None] - theirs.y_r[part]).abs().max()
		for name in ("attr", "deltas", "y_x", "y_r"):
			ours = getattr(got, name)[part]
			expected = getattr(theirs, name)[part]
			scale = dy if name == "deltas" else expected.abs().max()
			err = (ours - expected).abs().max() / scale
			assert float(err) <= RERUN_MAX, name
