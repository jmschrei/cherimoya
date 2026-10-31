"""Tests of the fast PISA engine on a GPU: the accuracy of its CUDA passes
and of each path of the blocks' Triton kernels, the fallback to torch where
a kernel does not fit the GPU, when the kernels run in TF32, a run at the
default model's size, and the recovery from running out of memory.

As in `test_deep_lift_shap_engine_cuda`, every fp32 path is held to the
float64 engine on the GPU, run eagerly, within bounds on the relative L2
distance of each sequence's attributions, over all its output positions:
1e-4 at the median and 2e-4 at the max.

`tests/conftest.py` disables dynamo for the suite, under which
`torch.compile` returns the function unchanged; the tests that compile turn
it back on for this module (`dynamo_enabled`) and check that they did.
"""

import logging
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
# The kernels in IEEE float32 sit within 1e-5 of float64 at the median and
# 5e-5 at the max; a block's product off by 0.1% read 9.7e-5 at the median.
PATH_MEDIAN, PATH_MAX = 2e-5, 1e-4
MEMORY_FACTOR = 1.3
RERUN_MAX = 1e-6
REAL_K, REAL_B, REAL_PEAKS, REAL_LENGTH = 20, 50, 2, 2114
RETAINED_MAX = 100e6
# Tiny models that take each kernel path at a filter count configs A to C do
# not reach: (filters, length, outputs_per_chunk, path). By position needs
# at least `position_min_outputs` outputs per chunk.
PATHS = [(32, 1024, None, "position"), (64, 512, None, "position"),
	(256, 1024, None, "position"), (128, 512, 8, "block")]


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


def _path_engine(C, length, dtype=torch.float32, **kwargs):
	"""An eager engine on a 2-layer model of C filters with redrawn weights,
	float32 or float64, for `PATHS`."""

	model = tiny.redraw_weights(Cherimoya(n_filters=C, n_layers=2,
		compile=False, random_state=0, verbose=False), 0)
	if dtype == torch.float64:
		model = tiny.enable_float64(model.double())

	engine = fpisa.Engine(model, device="cuda", n_shuffles=K, **kwargs)
	engine.forward, engine.rule_terms, engine.backward, engine.epilogue = (
		fpisa.forward_save, fpisa.rule_terms, fpisa.backward_pass,
		fpisa.epilogue)
	return engine


def _spies():
	"""Mocks that count the calls of the three kernels' wrappers."""

	return {name: mock.patch.object(fpisa, name, wraps=getattr(fpisa, name))
		for name in ("block_backward", "position_backward", "stem_backward")}


def _inaccurate(name, attr, truth, bounds=(ACCURACY_MEDIAN, ACCURACY_MAX)):
	"""The violations of the accuracy bounds, each sequence's relative L2
	distance from the truth at most bounds[0], `ACCURACY_MEDIAN` by default,
	at the median and bounds[1], `ACCURACY_MAX`, at the max."""

	r = tiny.rows_rel_l2(attr, truth)
	median, largest = float(r.median()), float(r.max())
	if median <= bounds[0] and largest <= bounds[1]:
		return []

	return ["{}: median {:.3g}, max {:.3g}".format(name, median, largest)]


##


@pytest.mark.triton
@pytest.mark.parametrize("ieee_kernels", [False, True])
@pytest.mark.parametrize("config", list(tiny.CONFIGS))
def test_the_cuda_passes_are_as_accurate_as_the_cpu(config, ieee_kernels,
	dynamo_enabled, strict_fp32, monkeypatch):
	"""In strict fp32 the eager CUDA passes, the shift-add convolution and the
	rule's statistics, and the compiled ones are within the bounds the CPU
	path, the model's own, meets: with the blocks in torch, as the engine
	runs them with TF32 off, and in the blocks' kernels in IEEE float32
	(`IEEE_KERNELS`)."""

	from torch._dynamo.utils import counters

	monkeypatch.setattr(fpisa, "IEEE_KERNELS", ieee_kernels)

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

	* The forward, the rule terms, the backward, the epilogue and the dense
	  parts' fields compile once each, with no graph break, and not again on
	  the second run. The 9 blocks unroll into one graph each way, with no
	  convolution: the forward with 2 cuBLAS matrix products per block, the
	  backward with the top block's 2 cuBLAS products and one
	  `_block_backward_kernel` launch for each block below it, and the
	  fields with 2 cuBLAS products for each block below the top one. The
	  only Triton matrix products are that
	  kernel's, in TF32 when ``torch.backends.cuda.matmul.allow_tf32`` is
	  on. On a tiny model with it off, the blocks run in torch, and in the
	  kernel in IEEE float32 with `IEEE_KERNELS` set.
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

	assert counters["stats"]["unique_graphs"] == 5
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
	assert counters["stats"]["unique_graphs"] == 5

	def launches(code):
		return len(re.findall(r"_block_backward_kernel_\d+\.run\(", code))

	def precisions(code):
		return set(re.findall(r"'PRECISION': '(\w+)'", code)) - {"constexpr"}

	assert len(codes) == 5
	mm = [len(re.findall(r"extern_kernels\.(?:mm|bmm|addmm)\(", code))
		for code in codes]
	assert sorted(zip(mm, map(launches, codes))) == [(0, 0), (0, 0), (2, 8),
		(16, 0), (18, 0)]
	tf32 = torch.backends.cuda.matmul.allow_tf32
	for code in codes:
		assert "triton_tem" not in code
		assert code.count("tl.dot") == 2 * code.count(
			"async_compile.triton('_block_backward_kernel'")
		assert precisions(code) == ({"tf32" if tf32 else "ieee"}
			if launches(code) else set())
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

	def tf32_off(ieee_kernels):
		codes = []
		previous = torch.get_float32_matmul_precision()
		torch.set_float32_matmul_precision("highest")
		try:
			assert not torch.backends.cuda.matmul.allow_tf32
			with mock.patch.object(GraphLowering, "save_output_code",
					codes.append), mock.patch.object(fpisa, "IEEE_KERNELS",
					ieee_kernels):
				_engine("A").run(tiny.sequences("A")[:1])
		finally:
			torch.set_float32_matmul_precision(previous)

		return codes

	off = tf32_off(False)
	assert off and sum(map(launches, off)) == 0
	assert all(precisions(code) == set() for code in off)

	ieee = tf32_off(True)
	assert sorted(map(launches, ieee))[-1] == tiny.CONFIGS["A"].n_layers - 1
	for code in ieee:
		assert precisions(code) == ({"ieee"} if launches(code) else set())


@pytest.mark.triton
@pytest.mark.parametrize("C, length, B, path", PATHS)
def test_each_kernel_path_is_as_accurate_as_float64(C, length, B, path,
	strict_fp32, monkeypatch):
	"""The blocks' kernels in IEEE float32 (`IEEE_KERNELS`), eager, at the
	filter counts configs A to C do not reach: by position at 32, 64 and 256
	filters, and `block_backward` at 128 filters, where it does not build
	the gradient at a block's output itself. Each path runs and is within
	`PATH_MEDIAN` and `PATH_MAX` of the float64 engine."""

	monkeypatch.setattr(fpisa, "IEEE_KERNELS", True)
	X = tiny.random_onehot(2, length, seed=5)
	R = tiny.pisa_references(X, K)
	truth = _path_engine(C, length, torch.float64).run(X, R).attr

	spies = _spies()
	with spies["block_backward"] as block, spies["position_backward"] as \
			position, spies["stem_backward"] as stem:
		attr = _path_engine(C, length, outputs_per_chunk=B).run(X, R).attr

	assert stem.call_count > 0
	assert (position if path == "position" else block).call_count > 0
	assert (block if path == "position" else position).call_count == 0
	failures = _inaccurate(path, attr, truth, (PATH_MEDIAN, PATH_MAX))
	assert not failures, failures


@pytest.mark.triton
@pytest.mark.parametrize("cause", ["shared memory", "compile"])
def test_kernels_that_do_not_fit_run_in_torch(cause, strict_fp32,
	monkeypatch, caplog):
	"""Where no setting of a kernel fits the GPU's 1,024 B of shared memory
	per block, or Triton compiles none, the engine logs a warning and runs
	that part in torch: the blocks' kernels with either cause, and the stem's
	when Triton compiles nothing (its smallest tile needs no shared memory).
	The attributions stay within the accuracy bounds."""

	monkeypatch.setattr(fpisa, "IEEE_KERNELS", True)
	monkeypatch.setattr(fpisa, "_SETTINGS", {})
	if cause == "shared memory":
		real = fpisa.device_properties
		monkeypatch.setattr(fpisa, "device_properties",
			lambda device: (1024, real(device)[1]))
	else:
		def fails(*args, **kwargs):
			raise RuntimeError("injected by the test")

		monkeypatch.setattr(fpisa, "_resources", fails)

	X = tiny.sequences("C")[:N]
	R = tiny.pisa_references(X, K)
	truth = _engine("C", torch.float64, eager=True).run(X, R).attr

	spies = _spies()
	with caplog.at_level(logging.WARNING, logger=fpisa.__name__), \
			spies["block_backward"] as block, spies["position_backward"] as \
			position, spies["stem_backward"] as stem:
		attr = _engine("C", eager=True).run(X, R).attr

	assert block.call_count == 0 and position.call_count == 0
	assert (stem.call_count == 0) == (cause == "compile")
	assert "no setting runs on this GPU" in caplog.text
	failures = _inaccurate("fallback", attr, truth)
	assert not failures, failures


@pytest.mark.triton
def test_the_kernels_run_in_tf32_only_where_the_gpu_has_it(monkeypatch):
	"""`use_tf32` holds only with ``torch.backends.cuda.matmul.allow_tf32``
	on and a GPU of compute capability 8.0 or more, and where it does not,
	the engine gives the blocks no kernel settings: they run in torch."""

	monkeypatch.setattr(fpisa, "_SETTINGS", {})
	real = fpisa.device_properties
	previous = torch.backends.cuda.matmul.allow_tf32
	try:
		for allow, capability, expected in ((True, (8, 0), True), (True,
				(9, 0), True), (True, (7, 5), False), (False, (9, 0), False)):
			torch.backends.cuda.matmul.allow_tf32 = allow
			monkeypatch.setattr(fpisa, "device_properties",
				lambda device, c=capability: (real(device)[0], c))
			assert fpisa.use_tf32("cuda") is expected
			settings = _engine("A")._settings()
			assert (settings[0] is not None) is expected
	finally:
		torch.backends.cuda.matmul.allow_tf32 = previous


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
