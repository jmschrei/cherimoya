"""Tests of the fast DeepLIFT/SHAP engine on a GPU: the accuracy of its CUDA
passes, a run at the default model's size, and the recovery from running out
of memory.

Every fp32 path through these models sits 1e-6 to 1e-4 (relative L2, per
sequence) from float64, because the rescale rule's secants, (g(z_x) -
g(z_r)) / (z_x - z_r), turn a last-bit difference in a GELU input into a
relative error of about one ulp / |z_x - z_r|. Any reordering of a sum shows
up this way: another matrix-product blocking, a fused reduction, another
device. So each path is held to the float64 engine on the GPU, run eagerly,
as the reference: it must be no farther from it than the path it replaces,
by 1.2 x at the median and 2 x at the max, and within 1e-3 of that path.
The max over the sequences is one element's amplified rounding, and moves
between compilations: Inductor picks its reduction configurations by timing.

`tests/conftest.py` disables dynamo for the suite, under which
`torch.compile` returns the function unchanged; the tests that compile turn
it back on for this module (`dynamo_enabled`) and check that they did.
"""

import re

import pytest
import torch

from unittest import mock

from cherimoya import Cherimoya
from cherimoya import _deep_lift_shap_engine as fdls

from . import deep_lift_shap_engine_helpers as tiny


pytestmark = pytest.mark.cuda

K = 5
ACCURACY_FACTORS = {"median": 1.2, "max": 2.0}
SANITY_MAX = 1e-3
MEMORY_FACTOR = 1.3
RERUN_MAX = 1e-6
REAL_K, REAL_S, REAL_PEAKS, REAL_LENGTH = 20, 4, 13, 2114
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
	engine = fdls.Engine(model, group=tiny.CONFIGS[config].group, **kwargs)
	if eager:
		engine.forward, engine.backward, engine.epilogue = (fdls.forward_save,
			fdls.backward_pass, fdls.epilogue)

	return engine


def _rows_rel_l2(a, b):
	a, b = a.double().flatten(1), b.double().flatten(1)
	return (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)


def _accuracy_relative(a, b, truth):
	"""The violations of the accuracy-relative bound: a no farther from the
	truth than b, by `ACCURACY_FACTORS` at the median and the max, and the
	two within `SANITY_MAX` of each other."""

	da, db = _rows_rel_l2(a, truth), _rows_rel_l2(b, truth)
	failures = []
	for stat, fn in (("median", torch.median), ("max", torch.max)):
		if not float(fn(da)) <= ACCURACY_FACTORS[stat] * float(fn(db)):
			failures.append("{} {:.3g} > {} x {:.3g}".format(stat,
				float(fn(da)), ACCURACY_FACTORS[stat], float(fn(db))))

	between = float(_rows_rel_l2(a, b).max())
	if not between <= SANITY_MAX:
		failures.append("between the paths {:.3g}".format(between))

	return failures


##


@pytest.mark.triton
@pytest.mark.parametrize("config", list(tiny.CONFIGS))
def test_the_cuda_passes_are_as_accurate_as_the_cpu(config, dynamo_enabled,
	strict_fp32):
	"""In strict fp32 the eager CUDA passes, the shift-add convolution and the
	rule's statistics, are as accurate as the CPU path, the model's own; and
	the compiled passes are as accurate as the eager ones."""

	from torch._dynamo.utils import counters

	X = tiny.sequences(config)
	R = tiny.shuffled_references(X, K)
	truth = _engine(config, torch.float64, eager=True).run(X, R).attr
	cpu = _engine(config, device="cpu").run(X, R).attr
	eager = _engine(config, eager=True).run(X, R).attr
	counters.clear()
	compiled = _engine(config).run(X, R).attr
	assert counters["stats"]["unique_graphs"] >= 2  # it did compile

	failures = _accuracy_relative(eager, cpu, truth)
	failures += _accuracy_relative(compiled, eager, truth)
	assert not failures, failures


@pytest.mark.triton
def test_a_run_at_the_default_size(dynamo_enabled):
	"""Two runs of one engine on the default architecture, 13 sequences in
	steps of 4 with 20 references each.

	* The forward, the backward and the epilogue compile once each, with no
	  graph break, and not again on the second run. The 9 blocks unroll into
	  one graph each way, each with 2 cuBLAS matrix products per block, and
	  no Triton matrix product or convolution.
	* The second run raises on any implicit host synchronization
	  (torch.cuda.set_sync_debug_mode("error")): the host waits only on each
	  step's event.
	* The compiling first run gives its traced step back, and the second
	  allocates at most 1.3 x `step_bytes` above what was allocated before
	  it, the estimate the automatic step size relies on.
	* The two runs agree, to cuDNN's non-repeatable input gradient.
	"""

	from torch._dynamo.utils import counters
	from torch._inductor.graph import GraphLowering

	model = tiny.redraw_weights(Cherimoya(compile=False, random_state=0,
		verbose=False), 0)
	X = tiny.random_onehot(REAL_PEAKS, REAL_LENGTH, seed=11)
	R = tiny.shuffled_references(X, REAL_K)
	engine = fdls.Engine(model, device="cuda", seqs_per_step=REAL_S)

	codes = []
	torch.cuda.empty_cache()
	at_start = torch.cuda.memory_allocated()
	counters.clear()
	with mock.patch.object(GraphLowering, "save_output_code", codes.append):
		first = engine.run(X, R)

	assert counters["stats"]["unique_graphs"] == 3
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
	assert counters["stats"]["unique_graphs"] == 3

	assert len(codes) == 3
	mm = [len(re.findall(r"extern_kernels\.(?:mm|bmm|addmm)\(", code))
		for code in codes]
	assert sorted(mm) == [0, 18, 18]
	for code in codes:
		assert "tl.dot" not in code and "triton_tem" not in code
		assert not re.search(r"convolution|conv1d|cudnn", code,
			flags=re.IGNORECASE)

	estimate = fdls.step_bytes(REAL_S, REAL_K, REAL_LENGTH, 128, 256, 9)
	assert before - at_start <= RETAINED_MAX
	assert peak <= MEMORY_FACTOR * estimate

	for name in ("y_x", "y_r"):
		assert torch.equal(getattr(first, name), getattr(second, name))

	assert float(_rows_rel_l2(second.attr, first.attr).max()) <= RERUN_MAX
	assert bool(torch.isfinite(first.attr).all())


@pytest.mark.parametrize("at_call, calls", [(1, [4] + [2] * 7),
	(2, [4, 4] + [2] * 5)])
def test_out_of_memory_halves_the_step_and_retries_it(at_call, calls):
	"""An out-of-memory error in the first or the second step of 4 sequences,
	with the references drawn: the sequences before it keep their step of 4,
	the rest run in steps of 2 and give what an engine planned at that size
	gives, and every sequence is attributed once."""

	X = tiny.sequences("C")
	engine = _engine("C", seqs_per_step=4, n_shuffles=K)
	seen = []
	step = engine._step

	def failing(inputs, S, K, G0):
		seen.append(S)
		if len(seen) == at_call:
			raise torch.cuda.OutOfMemoryError("injected by the test")

		return step(inputs, S, K, G0)

	engine._step = failing
	got = engine.run(X)
	assert seen == calls

	kept = 4 * (at_call - 1)
	by_4 = _engine("C", seqs_per_step=4, n_shuffles=K).run(X)
	by_2 = _engine("C", seqs_per_step=2, n_shuffles=K).run(X)
	for name in ("attr", "deltas", "y_x", "y_r"):
		ours = getattr(got, name)
		for theirs, part in ((by_4, slice(None, kept)), (by_2,
				slice(kept, None))):
			expected = getattr(theirs, name)[part]
			if len(expected):
				err = (ours[part] - expected).abs().max() / expected.abs().max()
				assert float(err) <= RERUN_MAX, name
