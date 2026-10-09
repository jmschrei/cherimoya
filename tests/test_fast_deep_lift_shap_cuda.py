"""Tests of the fast DeepLIFT/SHAP engine on a GPU: the compiled passes, the
pipelined steps and the recovery from running out of memory.

Every fp32 path through these models sits 1e-6 to 1e-4 (relative L2, per
pair) from float64, because the rescale rule's secants, (g(z_x) -
g(z_r)) / (z_x - z_r), turn a last-bit difference in a GELU input into a
relative error of about one ulp / |z_x - z_r|. Any reordering of a sum shows
up this way: another matrix-product blocking, a fused reduction, another
device. So the compiled passes and the GPU are held to the float64 engine on
the CPU, as the reference: they must be no farther from it than the path
they replace, by 1.2 x at the median and 2 x at the max, and within 1e-3 of
that path. The max over the pairs is one element's amplified rounding, and
moves between compilations: Inductor picks its reduction configurations by
timing, and the same inputs have given 1.10 and 1.25 x eager's max.

`tests/conftest.py` disables dynamo for the suite, under which
`torch.compile` returns the function unchanged; the tests that compile turn
it back on for this module (`dynamo_enabled`) and check that they did.
"""

import re

import numpy
import pytest
import torch

from unittest import mock

from cherimoya import Cherimoya
from cherimoya.fast_deep_lift_shap import engine as fdls
from cherimoya.fast_deep_lift_shap import references

from . import fast_deep_lift_shap_helpers as tiny


pytestmark = pytest.mark.cuda

K = 5
HEAD_ROWS = 14
MEM_BUDGET_GB = 4.0
ACCURACY_FACTORS = {"median": 1.2, "max": 2.0}
SANITY_MAX = 1e-3
D3_FACTOR = 1.3
D4_MAX = 1e-5
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


def _window(length, width):
	start = length // 2 - width // 2
	return start, start + width


_CASES = {}


def _case(config, kind):
	if (config, kind) not in _CASES:
		X = tiny.sequences(config)
		if kind == "shuffled":
			R = tiny.shuffled_references(X, K)
		else:
			R = tiny.random_references(X, K, seed=3)

		_CASES[config, kind] = (X, R)

	return _CASES[config, kind]


def _tiny_engine(config, **kwargs):
	model = tiny.tiny_model(config)
	kwargs.setdefault("device", "cuda")
	kwargs.setdefault("mem_budget_gb", MEM_BUDGET_GB)
	return fdls.Engine(model, group=tiny.CONFIGS[config].group,
		head_rows=HEAD_ROWS, **kwargs)


def _rows_rel_l2(a, b):
	a, b = a.double().flatten(1), b.double().flatten(1)
	return (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)


_TRUTH = {}


def _truth(config, kind):
	"""The float64 engine on the CPU with "two_pass" statistics: the
	float64 reference for these comparisons."""

	if (config, kind) not in _TRUTH:
		X, R = _case(config, kind)
		model = tiny.tiny_model(config, dtype=torch.float64)
		engine = fdls.Engine(model, group=tiny.CONFIGS[config].group,
			head_rows=HEAD_ROWS, forward_stats="two_pass")
		_TRUTH[config, kind] = engine.raw(X, R)

	return _TRUTH[config, kind]


def _accuracy_relative(a, b, between):
	"""The paths' violations of the accuracy-relative bound: a no farther
	from the truth than b, by `ACCURACY_FACTORS` at the median and the max,
	and the two within 1e-3 of each other."""

	failures = []
	for stat, fn in (("median", torch.median), ("max", torch.max)):
		da, db = float(fn(a)), float(fn(b))
		if not da <= ACCURACY_FACTORS[stat] * db:
			failures.append("{} {:.3g} > {} x {:.3g}".format(stat, da,
				ACCURACY_FACTORS[stat], db))

	if not float(between.max()) <= SANITY_MAX:
		failures.append("between the paths {:.3g}".format(
			float(between.max())))

	return failures


##


@pytest.mark.triton
@pytest.mark.parametrize("kind", ["shuffled", "random"])
@pytest.mark.parametrize("config", list(tiny.CONFIGS))
def test_compiled_passes_are_as_accurate_as_eager(config, kind,
	dynamo_enabled):
	"""Compiled against eager, both in strict fp32 on the GPU, measured
	against the float64 engine on the CPU."""

	from torch._dynamo.utils import counters

	X, R = _case(config, kind)
	start, end = _window(X.shape[-1], X.shape[-1] // 4)
	counters.clear()
	compiled_engine = _tiny_engine(config, precision="fp32", compile="default")
	eager_engine = _tiny_engine(config, precision="fp32", compile="none")
	assert compiled_engine.fwd_c is not fdls.forward_save

	compiled = compiled_engine.raw(X, R)
	assert counters["stats"]["unique_graphs"] >= 2  # it did compile
	eager = eager_engine.raw(X, R)
	truth = _truth(config, kind)

	to_truth = {name: _rows_rel_l2(r.m.flatten(0, 1), truth.m.flatten(0, 1))
		for name, r in (("compiled", compiled), ("eager", eager))}
	between = _rows_rel_l2(compiled.m.flatten(0, 1), eager.m.flatten(0, 1))
	failures = _accuracy_relative(to_truth["compiled"], to_truth["eager"],
		between)
	assert not failures, failures

	for result in (compiled, eager):
		flags = result.debug["precision_flags"]
		assert flags["torch.get_float32_matmul_precision()"] == "highest"
		assert flags["torch.backends.cudnn.allow_tf32"] is False

	# run() shares raw()'s forward: the same outputs.
	run = compiled_engine.run(X, start, end, references=R)
	assert torch.equal(run.y_x, compiled.y_x)
	assert torch.equal(run.y_r, compiled.y_r)
	assert float(_rows_rel_l2(run.attr, eager_engine.run(X, start, end,
		references=R).attr).max()) <= SANITY_MAX


@pytest.mark.parametrize("config", list(tiny.CONFIGS))
def test_the_gpu_is_as_accurate_as_the_cpu(config):
	"""The same "two_pass" arithmetic, eager, on the GPU and on the CPU, in
	strict fp32, measured against the float64 engine on the CPU."""

	X, R = _case(config, "shuffled")
	gpu = _tiny_engine(config, precision="fp32", compile="none").raw(X, R)
	cpu = _tiny_engine(config, precision="fp32", device="cpu",
		forward_stats="two_pass").raw(X, R)
	truth = _truth(config, "shuffled")
	to_truth = {name: _rows_rel_l2(r.m.flatten(0, 1), truth.m.flatten(0, 1))
		for name, r in (("gpu", gpu), ("cpu", cpu))}
	between = _rows_rel_l2(gpu.m.flatten(0, 1), cpu.m.flatten(0, 1))
	failures = _accuracy_relative(to_truth["gpu"], to_truth["cpu"], between)
	assert not failures, failures


def test_cpu_mirror_is_refused_on_cuda():
	model = tiny.tiny_model("A")
	with pytest.raises(ValueError, match="cpu_mirror"):
		fdls.Engine(model, device="cuda", forward_stats="cpu_mirror")

	engine = fdls.Engine(model, device="cuda")
	assert (engine.forward_stats, engine.compile, engine.precision) == (
		"two_pass", "default", "tf32")

	w = engine.weights
	z0 = fdls.stem_forward(tiny.sequences("A")[:2].float().cuda(), w.W0, w.b0)
	with pytest.raises(ValueError, match="cpu_mirror"):
		fdls.forward_save(z0, w.blocks, w.rs, w.eps, "cpu_mirror")

	assert w.W0.data_ptr() != model.iconv.weight.data_ptr()
	assert torch.equal(w.W0.cpu(), model.iconv.weight.detach())


##


@pytest.fixture(scope="module")
def real_runs(dynamo_enabled):
	"""Two runs of an engine on the default architecture, recording what
	compiled, the generated code, the memory and the synchronizations."""

	from torch._dynamo.utils import counters
	from torch._inductor.graph import GraphLowering

	model = tiny.redraw_weights(Cherimoya(compile=False, random_state=0,
		verbose=False), 0)
	X = tiny.random_onehot(REAL_PEAKS, REAL_LENGTH, seed=11)
	R = tiny.shuffled_references(X, REAL_K)
	start, end = _window(REAL_LENGTH, 400)
	engine = fdls.Engine(model, device="cuda", seqs_per_step=REAL_S,
		mem_budget_gb=MEM_BUDGET_GB)

	codes = []
	torch.cuda.empty_cache()
	at_start = torch.cuda.memory_allocated()
	counters.clear()
	with mock.patch.object(GraphLowering, "save_output_code", codes.append):
		first = engine.run(X, start, end, references=R)

	graphs_first = counters["stats"]["unique_graphs"]
	breaks = {str(k): v for k, v in counters["graph_break"].items()}

	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()
	before = torch.cuda.memory_allocated()
	torch.cuda.set_sync_debug_mode("error")
	try:
		with torch._dynamo.config.patch(error_on_recompile=True):
			second = engine.run(X, start, end, references=R)
	finally:
		torch.cuda.set_sync_debug_mode(0)

	yield {
		"first": first,
		"second": second,
		"codes": codes,
		"graphs_first": graphs_first,
		"graphs_second": counters["stats"]["unique_graphs"],
		"breaks": breaks,
		"peak": torch.cuda.max_memory_allocated() - before,
		"before": before,
		"at_start": at_start,
		"estimate": fdls.step_bytes(REAL_S, REAL_K, REAL_LENGTH, 128, 256, 9),
	}

	torch.cuda.empty_cache()


@pytest.mark.triton
def test_one_host_synchronization_per_step(real_runs):
	"""The second run ran under torch.cuda.set_sync_debug_mode("error"),
	which raises on any implicit synchronization; the engine's one wait per
	step, on the previous step's event, is counted."""

	second = real_runs["second"].meta
	assert second["host_syncs"] == second["steps"] == 4
	first = real_runs["first"].meta
	assert first["host_syncs"] == first["steps"]


@pytest.mark.triton
def test_each_pass_compiles_once_to_extern_matrix_products(real_runs):
	"""The forward, the backward and the epilogue compile once each, with no
	graph break, and not again on the second run; the 9 blocks unroll into
	one graph each way, each with 2 cuBLAS matrix products per block and no
	Triton matrix product or convolution."""

	assert real_runs["first"].meta["steps"] == 4
	assert real_runs["graphs_first"] == 3
	assert real_runs["graphs_second"] == 3
	assert real_runs["breaks"] == {}

	codes = real_runs["codes"]
	assert len(codes) == 3
	mm = [len(re.findall(r"extern_kernels\.(?:mm|bmm|addmm)\(", code))
		for code in codes]
	assert sorted(mm) == [0, 18, 18]
	for code in codes:
		assert "tl.dot" not in code and "triton_tem" not in code
		assert not re.search(r"convolution|conv1d|cudnn", code,
			flags=re.IGNORECASE)


@pytest.mark.triton
def test_step_memory_is_within_the_estimate(real_runs):
	"""The most the second run allocates above what was allocated before it
	is within 1.3 x `step_bytes`, the estimate automatic sizing uses; and
	the compiling first run gives its traced step back."""

	assert real_runs["second"].meta["step_estimate_bytes"] \
		== real_runs["estimate"]
	assert real_runs["peak"] <= D3_FACTOR * real_runs["estimate"]
	assert real_runs["before"] - real_runs["at_start"] <= RETAINED_MAX


@pytest.mark.triton
def test_two_runs_of_one_engine_agree(real_runs):
	first, second = real_runs["first"], real_runs["second"]
	for name in ("y_x", "y_r"):
		assert torch.equal(getattr(first, name), getattr(second, name))

	# cuDNN's input gradient is not bitwise repeatable.
	assert float(_rows_rel_l2(second.attr, first.attr).max()) <= RERUN_MAX
	assert bool(torch.isfinite(first.attr).all())


@pytest.mark.triton
def test_two_pass_normalization_matches_the_triton_forward():
	"""The forward's normalization, `dw3` then (y - mu) * v with the rule's
	statistics, against cherimoya's Triton forward on zero-mean input at
	every dilation of the default model."""

	from cherimoya.cheri import FusedDilatedConvNormFunc

	gen = torch.Generator(device="cuda").manual_seed(4)
	x = torch.randn(4, REAL_LENGTH, 128, generator=gen, device="cuda")
	x = (x - x.mean(dim=(1, 2), keepdim=True)).contiguous()
	cw = torch.randn(3, 128, generator=gen, device="cuda") * 0.8
	eps = fdls.CONV_NORM_EPS
	worst = 0.0
	for i in range(9):
		d = 2**i
		with torch.enable_grad():
			theirs = FusedDilatedConvNormFunc.apply(x.requires_grad_(),
				cw.requires_grad_(), d).detach()

		x, cw = x.detach(), cw.detach()
		with torch.no_grad():
			ours = fdls.norm_two_pass(fdls.dw3(x, cw, d), eps)[0]

		err = ((ours - theirs).abs() / theirs.abs().clamp_min(1)).max()
		worst = max(worst, float(err))

	assert worst <= D4_MAX


##


def test_automatic_step_size_fits_the_budget():
	model = tiny.redraw_weights(Cherimoya(compile=False, random_state=0,
		verbose=False), 0)
	per_peak = fdls.step_bytes(1, REAL_K, REAL_LENGTH, 128, 256, 9)
	free, _ = torch.cuda.mem_get_info()
	for budget, expected in ((2.0, 2), (0.5, 1), (4.0, 4)):
		if not 0.6 * free > budget * 1e9:
			pytest.skip("the GPU has too little free memory for this test")

		engine = fdls.Engine(model, device="cuda", mem_budget_gb=budget)
		assert max(1, int(budget * 1e9 // per_peak)) == expected
		assert engine.step_size(100, REAL_LENGTH, REAL_K) == expected
		assert engine.step_size(1, REAL_LENGTH, REAL_K) == 1

	fixed = fdls.Engine(model, device="cuda", seqs_per_step=3,
		mem_budget_gb=0.5)
	assert fixed.step_size(100, REAL_LENGTH, REAL_K) == 3


def _inject_oom(engine, at_call, when_s):
	"""Make the engine's at_call-th device step raise
	torch.cuda.OutOfMemoryError if it runs when_s sequences."""

	calls = []
	step = engine._device_step

	def failing(tokens, S, K, G0, window):
		calls.append(S)
		if len(calls) == at_call and S == when_s:
			raise torch.cuda.OutOfMemoryError("injected by the test")

		return step(tokens, S, K, G0, window)

	engine._device_step = failing
	return calls


def test_out_of_memory_in_the_warm_up_halves_the_step():
	"""All steps run at the halved size, and the results equal, bit for bit,
	a deterministic engine that ran at that size from the start."""

	X, R = _case("C", "shuffled")
	start, end = _window(X.shape[-1], X.shape[-1] // 4)
	kwargs = dict(precision="fp32", deterministic=True)
	engine = _tiny_engine("C", seqs_per_step=4, **kwargs)
	calls = _inject_oom(engine, at_call=1, when_s=4)
	got = engine.run(X, start, end, references=R)
	expected = _tiny_engine("C", seqs_per_step=2, **kwargs).run(X, start, end,
		references=R)

	assert got.meta["oom"] == [{"step": "warm-up", "from": 4, "to": 2}]
	assert got.meta["seqs_per_step"] == 2 and got.meta["steps"] == 7
	assert calls == [4, 2] + [2] * 7
	for name in ("attr", "deltas", "y_x", "y_r"):
		assert torch.equal(getattr(got, name), getattr(expected, name)), name


def test_out_of_memory_in_a_step_halves_the_step_and_retries_it():
	"""An error in the second of 4-sequence steps: sequences 0-3 keep their
	step of 4, sequences 4-12 run in steps of 2, bit for bit as the S = 4
	and S = 2 engines compute them, and every sequence is attributed
	once."""

	X, R = _case("C", "shuffled")
	kwargs = dict(precision="fp32", deterministic=True)
	engine = _tiny_engine("C", seqs_per_step=4, **kwargs)
	calls = _inject_oom(engine, at_call=3, when_s=4)
	got = engine.raw(X, R)
	by_4 = _tiny_engine("C", seqs_per_step=4, **kwargs).raw(X, R)
	by_2 = _tiny_engine("C", seqs_per_step=2, **kwargs).raw(X, R)

	assert got.debug["oom"] == [{"step": 2, "from": 4, "to": 2}]
	assert calls == [4, 4, 4, 2, 2, 2, 2, 2]
	assert got.debug["steps"] == 6 and got.debug["padded_peaks"] == 1
	assert got.debug["host_syncs"] == got.debug["steps"]
	for name in ("m", "y_x", "y_r", "deltas"):
		ours = getattr(got, name)
		assert torch.equal(ours[:4], getattr(by_4, name)[:4]), name
		assert torch.equal(ours[4:], getattr(by_2, name)[4:]), name


def test_a_warm_up_halving_recuts_the_pools_chunks():
	"""The pool starts drawing chunks of 5 sequences before the warm-up,
	which halves the step to 2: the run takes the 7 steps and 1 padded
	sequence of an engine planned at 2, with its results bit for bit."""

	X, R = _case("C", "shuffled")
	start, end = _window(X.shape[-1], X.shape[-1] // 4)
	kwargs = dict(precision="fp32", deterministic=True, n_shuffles=K,
		random_state=0)
	engine = _tiny_engine("C", seqs_per_step=5, ref_workers=2, **kwargs)
	calls = _inject_oom(engine, at_call=1, when_s=5)
	got = engine.run(X, start, end, keep_references=list(range(len(X))))
	expected = _tiny_engine("C", seqs_per_step=2, **kwargs).run(X, start, end,
		references=R)

	assert got.meta["oom"] == [{"step": "warm-up", "from": 5, "to": 2}]
	assert calls == [5, 2] + [2] * 7
	assert (got.meta["steps"], got.meta["padded_peaks"]) == (7, 1)
	numpy.testing.assert_array_equal(got.references,
		references.onehot_to_tokens(R))
	for name in ("attr", "deltas", "y_x", "y_r"):
		assert torch.equal(getattr(got, name), getattr(expected, name)), name
