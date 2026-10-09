"""Tests of the fast DeepLIFT/SHAP engine against Cherimoya's own forward and
tangermeme's `deep_lift_shap`, on the CPU.

The forward. With "cpu_mirror" statistics the engine's forward runs the
model's CPU path operation for operation, so on the same rows it gives the
model's own output: within 1e-12 relative in float64 and 2e-6 in float32,
and its saved activations are the module's.

The whole engine. Both sides get the same 13 sequences (12 random, 1
AT-rich) and the same injected references, K = 5 per sequence, of four
kinds: dinucleotide shuffles, random one-hots, r = x, and x with 3 mutated
positions. The oracle is `deep_lift_shap` as `cherimoya attribute` calls
it, on the CPU, at batch_size 7, which splits a sequence's pairs across
batches. The engine runs in steps of 8 sequences, so 13 sequences fill one
step and pad the second with 3 copies of the last one.

* Raw multipliers, per pair, relative L2 over (4, L): <= 1e-9 in float64; a
  median <= 1e-6 and a max <= 1e-5 in float32.
* Hypothetical attributions, per sequence, over the full length and over a
  centred window: <= 1e-9 in float64 and 1e-5 in float32.
* Convergence deltas in float64: |delta_ours - delta_tangermeme| <= 1e-9 x
  max(1, |y_x - y_r|).
* Conservation: with "two_pass" statistics the rules sum to y_x - y_r, so
  every delta is <= 1e-10 x max(1, |y_x - y_r|) in float64.
* r = x gives the plain gradient; 3 mutated positions exercise the rescale
  rule's gradient fallback, |z_x - z_r| < 1e-6, at every GELU.
* Steps of 3 and 8 sequences give the results of steps of 1 to 1e-12.
* The rules matter: with shuffled references every pair's multipliers are at
  least 10% from the plain gradient.

"cpu_mirror" normalizes in float32 even in a float64 model, as the model
does, so its forward is the oracle's; "two_pass" sits about 1e-8 from the
oracle in float64 but conserves to float64 accuracy.
"""

import copy
import functools
import logging
import os
import re
import subprocess
import sys
import warnings

import numpy
import pytest
import torch

from cherimoya import Cherimoya
from cherimoya import ControlWrapper
from cherimoya import LogCountWrapper
from cherimoya.cheri import CONV_NORM_EPS
from cherimoya.cheri import _cheri_conv
from cherimoya.deep_lift_shap import attribution_ops
from cherimoya.fast_deep_lift_shap import engine as fdls
from cherimoya.fast_deep_lift_shap import references

from . import fast_deep_lift_shap_helpers as tiny


K = 5
C1_TOL = {torch.float32: 2e-6, torch.float64: 1e-12}
E1_FP32_TOL = 5e-5
DTYPES = [torch.float32, torch.float64]
CONFIG_NAMES = list(tiny.CONFIGS)
F64 = torch.float64


def _rel(a, b):
	"""The largest |a - b| / max(1, |b|)."""

	return float(((a - b).abs() / b.abs().clamp_min(1)).max())


def _tm_rule_stats(y, eps):
	"""Per-row mu and v as tangermeme's `_layer_normalization_helper`
	computes them over the trailing two axes."""

	mu = y.mean(dim=(-2, -1), keepdim=True)
	a = y - mu
	var = (a**2).mean(dim=(-2, -1), keepdim=True)
	v = (var + eps) ** (-0.5)
	return mu.flatten(), v.flatten()


@pytest.fixture(scope="module")
def batches():
	"""Per config, the engine's joint batch: 13 sequences, then their K
	shuffles, float32 (13 + 13K, 4, L)."""

	out = {}
	for name in CONFIG_NAMES:
		X = tiny.sequences(name)
		out[name] = torch.cat([X.float(),
			tiny.shuffled_references(X, K).flatten(0, 1)])

	return out


def _module_forward(model, X1h, group):
	"""The module's count output as `deep_lift_shap` computes it, and the
	activations forward hooks see: fconv's input, each block's input and
	each block's linear1 output."""

	acts = {"block_in": [], "u": []}
	handles = [model.fconv.register_forward_pre_hook(lambda m, args:
		acts.update(fconv_in=args[0].detach().clone()))]
	for block in model.blocks:
		handles.append(block.conv.register_forward_pre_hook(lambda m, args:
			acts["block_in"].append(args[0].detach())))
		handles.append(block.linear1.register_forward_hook(lambda m, args,
			out: acts["u"].append(out.detach())))

	try:
		with torch.enable_grad():
			y = LogCountWrapper(ControlWrapper(model), group=group)(
				X1h.clone().requires_grad_())[:, 0]
	finally:
		for handle in handles:
			handle.remove()

	return y.detach(), acts


##


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_cpu_mirror_forward_matches_the_module(config, dtype, batches):
	cfg = tiny.CONFIGS[config]
	model = tiny.tiny_model(config, dtype=dtype)
	X1h = batches[config].to(dtype)
	y_ref, acts = _module_forward(model, X1h, cfg.group)

	w = fdls.CheriWeights.from_model(model)
	with torch.no_grad():
		ours = fdls.model_forward(X1h, w, forward_stats="cpu_mirror",
			output="counts", group=cfg.group)

	assert ours.y.dtype == dtype and ours.y.shape == y_ref.shape
	assert len(ours.saved) == len(acts["block_in"]) == cfg.n_layers

	assert _rel(ours.y, y_ref) <= C1_TOL[dtype]

	# fconv reads cat([stream, controls]).
	assert _rel(ours.hL.transpose(1, 2), acts["fconv_in"][:, :w.C]) \
		<= C1_TOL[dtype]

	for s, block, block_in, u in zip(ours.saved, w.blocks, acts["block_in"],
			acts["u"]):
		y_rule = _cheri_conv(block_in, block.cw, block.d)
		mu, v = _tm_rule_stats(y_rule, w.eps)
		for a, b in ((s.y, y_rule), (s.u, u), (s.mu, mu), (s.v, v)):
			assert _rel(a, b) <= C1_TOL[dtype]


def test_float64_patch_is_the_stock_forward_in_float32(batches):
	"""The helper that lets the oracle run a float64 model widens only the
	head's casts: in float32 both heads are bitwise the model's own."""

	for name in CONFIG_NAMES:
		stock = tiny.tiny_model(name)
		patched = tiny.enable_float64(copy.deepcopy(stock))
		X1h = batches[name][:20]
		with torch.enable_grad():
			profile, counts = ControlWrapper(stock)(X1h.clone().requires_grad_())
			profile_p, counts_p = ControlWrapper(patched)(
				X1h.clone().requires_grad_())

		assert torch.equal(profile, profile_p) and torch.equal(counts, counts_p)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_two_pass_and_one_pass_compute_the_same_function(config, dtype,
	batches):
	""""two_pass" and "one_pass" differ from "cpu_mirror" only in the
	statistics' formula and reduction order. The saved statistics are the
	rule's in every mode."""

	cfg = tiny.CONFIGS[config]
	w = fdls.CheriWeights.from_model(tiny.tiny_model(config, dtype=dtype))
	X1h = batches[config].to(dtype)
	with torch.no_grad():
		runs = {s: fdls.model_forward(X1h, w, forward_stats=s,
			group=cfg.group) for s in fdls.FORWARD_STATS}

	mirror = runs["cpu_mirror"]
	for s in ("two_pass", "one_pass"):
		assert _rel(runs[s].y, mirror.y) <= E1_FP32_TOL

	if dtype == F64:
		# The same function in exact arithmetic, and the same conv in block 0.
		assert _rel(runs["one_pass"].y, runs["two_pass"].y) <= C1_TOL[F64]
		assert _rel(runs["two_pass"].saved[0].y, mirror.saved[0].y) <= 1e-14

	for s in ("two_pass", "one_pass"):
		for got in runs[s].saved:
			mu, v = _tm_rule_stats(got.y, w.eps)
			assert torch.equal(got.mu, mu) and torch.equal(got.v, v)


def test_counts_head_groups_and_controls(batches):
	"""Config B, with two count groups and one control track: every group,
	and group None (target 0), as the module computes them."""

	model = tiny.tiny_model("B")
	w = fdls.CheriWeights.from_model(model)
	X1h = batches["B"]
	with torch.no_grad():
		hL = fdls.model_forward(X1h, w, forward_stats="cpu_mirror",
			group=0).hL

	for group in (0, 1, None):
		y_ref, _ = _module_forward(model, X1h, group)
		with torch.no_grad():
			y = fdls.head_forward(hL, w, output="counts", group=group)

		assert _rel(y, y_ref) <= C1_TOL[torch.float32]

	with torch.no_grad():
		y0, y1 = (fdls.head_forward(hL, w, group=g) for g in (0, 1))

	assert not torch.equal(y0, y1)
	with pytest.raises(NotImplementedError, match="count head only"):
		fdls.head_forward(hL, w, output="profile", group=1)

	for bad in (2, -1, True, 0.0):
		with pytest.raises(ValueError, match="group"):
			fdls.head_forward(hL, w, group=bad)

	with pytest.raises(ValueError, match="output"):
		fdls.head_forward(hL, w, output="logits")

	with pytest.raises(ValueError, match="trimming"):
		fdls.head_forward(hL[:, :2 * w.T], w)


def test_make_wrapper_runs_the_reference_group_checks():
	model = tiny.tiny_model("B")
	for output, group, cls in (("counts", 1, "LogCountWrapper"),
			("profile", None, "ProfileWrapper")):
		wrapped = fdls.make_wrapper(model, output, group)
		assert type(wrapped).__name__ == cls
		assert isinstance(wrapped.model, ControlWrapper)

	with pytest.raises(ValueError, match="attributes one output"):
		fdls.make_wrapper(model, "counts", None)

	for output in ("counts", "profile"):
		for bad in (2, True):
			with pytest.raises(ValueError, match="group must be"):
				fdls.make_wrapper(model, output, bad)

	with pytest.raises(ValueError, match="output"):
		fdls.make_wrapper(model, "logits", 0)


def test_forward_stats_and_compile_resolution(batches):
	assert fdls.resolve_forward_stats("auto", "cpu") == "cpu_mirror"
	assert fdls.resolve_forward_stats("auto", torch.device("cuda", 0)) \
		== "two_pass"
	assert fdls.resolve_forward_stats("one_pass", "cpu") == "one_pass"
	for cuda in ("cuda", torch.device("cuda", 0)):
		with pytest.raises(ValueError, match="cpu_mirror"):
			fdls.resolve_forward_stats("cpu_mirror", cuda)

	assert fdls.resolve_compile("auto", "cpu") == "none"
	assert fdls.resolve_compile("auto", "cuda") == "default"
	assert fdls.resolve_compile("none", "cuda") == "none"
	with pytest.raises(ValueError, match="compile"):
		fdls.resolve_compile("default", "cpu")

	for bad in ("max-autotune", "", None):
		with pytest.raises(ValueError, match="compile"):
			fdls.resolve_compile(bad, "cuda")

	for bad in ("three_pass", "", None):
		with pytest.raises(ValueError, match="forward_stats"):
			fdls.resolve_forward_stats(bad, "cpu")

	w = fdls.CheriWeights.from_model(tiny.tiny_model("A"))
	z0 = fdls.stem_forward(batches["A"][:2], w.W0, w.b0)
	with pytest.raises(ValueError, match="stats"):
		fdls.forward_save(z0, w.blocks, w.rs, w.eps, "auto")


def _module_tensors(model):
	tensors = [model.iconv.weight, model.iconv.bias]
	for block in model.blocks:
		tensors += [block.conv.conv_weight, block.linear1.weight,
			block.linear2.weight]

	return tensors + [model.fconv.weight, model.fconv.bias,
		model.linear.weight, model.linear.bias]


def _weights_tensors(w):
	tensors = [w.W0, w.b0]
	for block in w.blocks:
		tensors += [block.cw, block.W1, block.W2]

	return tensors + [w.Wf, w.bf, w.Wl, w.bl]


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_weights_are_copies_of_the_module_tensors(config):
	cfg = tiny.CONFIGS[config]
	for dtype in DTYPES:
		model = tiny.tiny_model(config, dtype=dtype)
		w = fdls.CheriWeights.from_model(model)
		for ours, theirs in zip(_weights_tensors(w), _module_tensors(model),
				strict=True):
			assert torch.equal(ours, theirs) and ours.dtype == dtype
			assert not ours.requires_grad
			assert ours.data_ptr() != theirs.data_ptr()

		assert [b.d for b in w.blocks] == [2**i for i in range(cfg.n_layers)]
		assert (w.T, w.rs, w.eps) == (model.trimming, model.residual_scale,
			CONV_NORM_EPS)
		assert (w.C, w.H, w.n_ctl) == (cfg.n_filters,
			cfg.expansion * cfg.n_filters, cfg.n_control_tracks)
		assert (w.n_out, w.n_groups, w.signal_groups) == (
			sum(cfg.signal_groups), len(cfg.signal_groups), cfg.signal_groups)
		assert (w.dtype, w.device) == (dtype, torch.device("cpu"))

	with pytest.raises(ValueError, match="CPU only"):
		fdls.CheriWeights.from_model(model, device="meta")


def test_both_depthwise_key_spellings_load(tmp_path, batches):
	"""Checkpoints spell the depthwise taps blocks.{i}.conv_weight or
	blocks.{i}.conv.conv_weight; both load into the same weights."""

	model = tiny.tiny_model("A")
	saved = tmp_path / "saved.torch"
	model.save(saved)
	payload = torch.load(saved, weights_only=True)
	old = re.compile(r"^blocks\.(\d+)\.conv_weight$")
	assert sum(bool(old.match(key)) for key in payload["state_dict"]) \
		== model.n_layers

	renamed = tmp_path / "renamed.torch"
	state = {old.sub(r"blocks.\1.conv.conv_weight", key): value
		for key, value in payload["state_dict"].items()}
	torch.save({"config": payload["config"], "state_dict": state}, renamed)

	outputs = []
	for path in (saved, renamed):
		loaded = Cherimoya.load(str(path), device="cpu", compile=False)
		w = fdls.CheriWeights.from_model(loaded)
		for ours, theirs in zip(_weights_tensors(w), _module_tensors(model),
				strict=True):
			assert torch.equal(ours, theirs)

		with torch.no_grad():
			outputs.append(fdls.model_forward(batches["A"], w,
				forward_stats="cpu_mirror").y)

	assert torch.equal(*outputs)


def test_architecture_changes_are_refused():
	model = tiny.tiny_model("A")
	C, H = 16, 32

	def replace(path, value):
		def tamper(m):
			*parents, name = path.split(".")
			target = m
			for part in parents:
				target = target[int(part)] if part.isdigit() else getattr(target,
					part)

			setattr(target, name, value)

		return tamper

	tampers = {
		"depthwise taps": replace("blocks.0.conv.conv_weight",
			torch.nn.Parameter(torch.zeros(3, C + 1))),
		"linear1 with a bias": replace("blocks.1.linear1",
			torch.nn.Linear(C, H)),
		"linear2 shape": replace("blocks.2.linear2",
			torch.nn.Linear(H + 1, C, bias=False)),
		"count head shape": replace("linear", torch.nn.Linear(C + 1, 1)),
		"count head without bias": replace("linear",
			torch.nn.Linear(C, 1, bias=False)),
		"fconv channels": replace("fconv",
			torch.nn.Conv1d(C + 1, 1, 75, padding=37)),
		"fconv padding": replace("fconv", torch.nn.Conv1d(C, 1, 75,
			padding=36)),
		"iconv kernel": replace("iconv", torch.nn.Conv1d(4, C, 19, padding=9)),
		"iconv dilation": replace("iconv", torch.nn.Conv1d(4, C, 21,
			padding=10, dilation=2)),
		"erf GELU in a block": replace("blocks.1.activation", torch.nn.GELU()),
		"erf GELU in the stem": replace("igelu", torch.nn.GELU()),
		"dilation": replace("blocks.2.dilation", 3),
		"residual scale": replace("blocks.0.residual_scale", 0.2),
		"a block missing": replace("blocks", model.blocks[:2]),
		"float16": lambda m: m.half(),
		"mixed dtypes": lambda m: m.linear.double(),
	}

	for name, tamper in tampers.items():
		changed = copy.deepcopy(model)
		tamper(changed)
		with pytest.raises(ValueError, match="contract") as raised:
			fdls.CheriWeights.from_model(changed)

		assert str(raised.value).count("\n") >= 1, name

	with pytest.raises(ValueError, match="Cherimoya"):
		fdls.CheriWeights.from_model(ControlWrapper(model))

	assert fdls.contract_violations(model) == []


def test_an_unvalidated_tangermeme_warns(monkeypatch):
	"""The engine reproduces tangermeme's rules, so an unvalidated release
	warns; a validated one does not."""

	import tangermeme

	model = tiny.tiny_model("A")
	if tangermeme.__version__ in fdls.VALIDATED_TANGERMEME_VERSIONS:
		with warnings.catch_warnings():
			warnings.simplefilter("error")
			fdls.Engine(model)
	else:
		with pytest.warns(UserWarning, match="validated against"):
			fdls.Engine(model)

	monkeypatch.setattr(tangermeme, "__version__", "9.9.9")
	with pytest.warns(UserWarning, match="tangermeme 9.9.9 is installed"):
		fdls.Engine(model)

	assert fdls.check_tangermeme_version() == "9.9.9"


def test_importing_the_engine_changes_no_environment_variable():
	"""In a fresh process, importing the fast engine leaves os.environ as it
	was after importing cherimoya, which imports the engine's dependencies."""

	code = ("import os; import cherimoya; before = dict(os.environ); "
		"import cherimoya.fast_deep_lift_shap; "
		"changed = sorted(k for k in set(before) | set(os.environ) "
		"if before.get(k) != os.environ.get(k)); "
		"assert not changed, changed")
	env = dict(os.environ)
	env.pop("NUMBA_CACHE_DIR", None)
	env.pop("TRITON_CACHE_AUTOTUNING", None)
	done = subprocess.run([sys.executable, "-c", code], env=env,
		capture_output=True, text=True, timeout=600)
	assert done.returncode == 0, done.stderr[-3000:]


def test_shuffled_references_are_deep_lift_shaps_own():
	"""With no references passed, deep_lift_shap draws exactly the helper's
	shuffles, which the parity tests inject."""

	model = tiny.tiny_model("A")
	X = tiny.sequences("A")[[0, 1, 12]]
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="Convergence deltas too high",
			category=RuntimeWarning)
		drawn = tiny.TDLS.deep_lift_shap(tiny.wrapper(model), X, n_shuffles=2,
			random_state=0, return_references=True, hypothetical=True,
			additional_nonlinear_ops=attribution_ops(), dtype="float32",
			device="cpu").references

	assert torch.equal(drawn, tiny.shuffled_references(X, 2, random_state=0))


def test_the_oracle_returns_tangermemes_deltas_in_pair_order():
	"""The print sink returns (N, K) deltas in pair order: recomputed per
	batch of 7 pairs, packed as deep_lift_shap packs them, they are
	bitwise equal. A tolerance would hide a pair-order error."""

	model = tiny.tiny_model("A")
	X = tiny.sequences("A")
	R = tiny.shuffled_references(X, K)
	wrapped = tiny.wrapper(model, "counts", 0)
	threads = torch.get_num_threads()
	result = tiny.oracle(wrapped, X, R, raw_outputs=True, dtype=torch.float32,
		batch_size=7)
	assert torch.get_num_threads() == threads
	assert "print" not in vars(tiny.TDLS)
	assert result.attributions.shape == (13, K, 4, 256)
	assert result.deltas.shape == (13, K)

	pairs = [(e, j) for e in range(len(X)) for j in range(K)]
	expected = []
	torch.set_num_threads(1)
	try:
		for b in range(0, len(pairs), 7):
			e, j = (list(t) for t in zip(*pairs[b:b + 7]))
			x, r = X[e].float(), R[e, j]
			with torch.enable_grad():
				y = wrapped(torch.cat([x, r]).requires_grad_())[:, 0].detach()

			dy = y[:len(e)] - y[len(e):]
			expected.append((dy - ((x - r) * result.attributions[e,
				j]).sum(dim=(1, 2))).abs())
	finally:
		torch.set_num_threads(threads)

	assert torch.equal(result.deltas, torch.cat(expected).view(len(X), K))


##


ORACLE_BATCH = 7
HEAD_ROWS = 2 * ORACLE_BATCH
N_SEQUENCES = tiny.N_RANDOM + tiny.N_AT_RICH

RAW_FP64_MAX, RAW_FP32_MEDIAN, RAW_FP32_MAX = 1e-9, 1e-6, 1e-5
HYP_TOL = {torch.float32: 1e-5, F64: 1e-9}
DELTA_FP64 = 1e-9
CONSERVATION_FP64 = 1e-10
GRADIENT_TOL = 1e-6
STEP_TOL = 1e-12
RULES_MIN = 0.1

# Every kind of reference on the small configs A and B; shuffled ones, the
# default engine's, on C, whose float64 oracle takes seconds per call.
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


def _window(config):
	"""`cherimoya attribute`'s centred window, a quarter of the input."""

	L = tiny.CONFIGS[config].length
	start = L // 2 - (L // 4) // 2
	return start, start + L // 4


def _engine(config, dtype, **kwargs):
	"""An Engine on a fresh tiny model, attributing the config's group, with
	the oracle's head rows."""

	model = tiny.tiny_model(config, dtype=dtype)
	return fdls.Engine(model, group=tiny.CONFIGS[config].group,
		head_rows=HEAD_ROWS, **kwargs)


@functools.cache
def _ours_raw(config, dtype, kind, stats="cpu_mirror", step="auto"):
	X, R = _case(config, kind)
	return _engine(config, dtype, forward_stats=stats,
		seqs_per_step=step).raw(X, R)


@functools.cache
def _ours_run(config, dtype, kind, start, end, stats="cpu_mirror",
	step="auto"):
	X, R = _case(config, kind)
	return _engine(config, dtype, forward_stats=stats,
		seqs_per_step=step).run(X, start, end, references=R)


@functools.cache
def _theirs(config, dtype, kind, raw_outputs):
	"""tangermeme's deep_lift_shap on the same injected references."""

	X, R = _case(config, kind)
	model = tiny.tiny_model(config, dtype=dtype)
	wrapped = tiny.wrapper(model, "counts", tiny.CONFIGS[config].group)
	return tiny.oracle(wrapped, X, R, raw_outputs=raw_outputs, dtype=dtype,
		batch_size=ORACLE_BATCH, num_threads=torch.get_num_threads())


@functools.cache
def _gradient(config):
	"""The plain gradient of the attributed count at each sequence,
	(13, 4, L), by autograd through the float64 wrapper, with no rules."""

	model = tiny.tiny_model(config, dtype=F64)
	wrapped = tiny.wrapper(model, "counts", tiny.CONFIGS[config].group)
	X = tiny.sequences(config).to(F64).requires_grad_()
	with torch.enable_grad():
		grad, = torch.autograd.grad(wrapped(X)[:, 0].sum(), X)

	return grad


def _rows_rel_l2(a, b):
	"""||a - b|| / ||b|| for each row, in float64."""

	a, b = a.double().flatten(1), b.double().flatten(1)
	return (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)


def _dy(result):
	"""|y_x - y_r| per pair, (N, K), in float64."""

	return (result.y_x[:, None] - result.y_r).double().abs()


@pytest.mark.parametrize("config, dtype, kind", PARITY_CASES)
def test_raw_multipliers_and_deltas_match_tangermeme(config, dtype, kind):
	L = tiny.CONFIGS[config].length
	ours, theirs = _ours_raw(config, dtype, kind), _theirs(config, dtype,
		kind, True)
	assert ours.m.shape == theirs.attributions.shape == (N_SEQUENCES, K, 4, L)
	assert ours.deltas.shape == ours.y_r.shape == (N_SEQUENCES, K)
	assert ours.m.dtype == ours.deltas.dtype == ours.y_x.dtype == dtype
	assert ours.debug["seqs_per_step"] == 8 and ours.debug["steps"] == 2
	assert ours.debug["padded_peaks"] == 3

	rel = _rows_rel_l2(ours.m.flatten(0, 1), theirs.attributions.flatten(0, 1))
	if dtype == F64:
		assert float(rel.max()) <= RAW_FP64_MAX
		d_delta = ((ours.deltas.double() - theirs.deltas.double()).abs()
			/ _dy(ours).clamp_min(1))
		assert float(d_delta.max()) <= DELTA_FP64
	else:
		assert float(rel.median()) <= RAW_FP32_MEDIAN
		assert float(rel.max()) <= RAW_FP32_MAX


@pytest.mark.parametrize("config, dtype, kind", PARITY_CASES)
def test_hypothetical_attributions_match_tangermeme(config, dtype, kind):
	"""`Engine.run` against deep_lift_shap(hypothetical=True), per sequence,
	over the full length and the window. The window is the full-length run
	sliced, bit for bit, and both carry `Engine.raw`'s deltas and outputs."""

	L = tiny.CONFIGS[config].length
	start, end = _window(config)
	full = _ours_run(config, dtype, kind, 0, L)
	window = _ours_run(config, dtype, kind, start, end)
	theirs = _theirs(config, dtype, kind, False).attributions
	assert full.attr.shape == theirs.shape == (N_SEQUENCES, 4, L)
	assert window.attr.shape == (N_SEQUENCES, 4, end - start)
	assert full.attr.dtype == window.attr.dtype == dtype
	assert torch.equal(window.attr, full.attr[..., start:end])

	raw = _ours_raw(config, dtype, kind)
	for result in (full, window):
		for name in ("deltas", "y_x", "y_r"):
			assert torch.equal(getattr(result, name), getattr(raw, name)), name

	assert float(_rows_rel_l2(full.attr, theirs).max()) <= HYP_TOL[dtype]
	assert float(_rows_rel_l2(window.attr, theirs[..., start:end]).max()) \
		<= HYP_TOL[dtype]


@pytest.mark.parametrize("kind", ["shuffled", "random", "mutated"])
@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_two_pass_statistics_conserve(config, kind):
	"""With "two_pass" the forward normalizes with the rule's own
	statistics, so the multipliers sum to y_x - y_r."""

	ours = _ours_raw(config, F64, kind, "two_pass")
	ratio = ours.deltas / _dy(ours).clamp_min(1)
	assert float(ratio.max()) <= CONSERVATION_FP64


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_identical_references_give_the_gradient(config):
	"""r = x: every rule falls back to the ordinary gradient. The model's
	float32 normalization, in its backward too, sets the bound."""

	grad = _gradient(config).repeat_interleave(K, dim=0)
	for stats in ("cpu_mirror", "two_pass"):
		ours = _ours_raw(config, F64, "identical", stats)
		assert float(_rows_rel_l2(ours.m.flatten(0, 1), grad).max()) \
			<= GRADIENT_TOL, stats


@pytest.mark.parametrize("config", ["A", "B"])
def test_near_identical_references_take_the_gradient_fallback(config):
	"""3 mutated positions: at every GELU, some inputs of a sequence and its
	reference differ by less than 1e-6, where the rescale rule takes the
	gradient; the raw multipliers still match tangermeme's."""

	cfg = tiny.CONFIGS[config]
	X, R = _case(config, "mutated")
	w = fdls.CheriWeights.from_model(tiny.tiny_model(config, dtype=F64))
	X1h = torch.cat([X, R.flatten(0, 1)]).to(F64)
	with torch.no_grad():
		fwd = fdls.model_forward(X1h, w, forward_stats="cpu_mirror",
			group=cfg.group)

	sites = [fwd.z0] + [saved.u for saved in fwd.saved]
	for z in sites:
		zx, zr = fdls.split_rows(z, N_SEQUENCES, K)
		assert int(((zx - zr).abs() < fdls.RESCALE_EPS).sum()) >= 1

	theirs = _theirs(config, F64, "mutated", True).attributions
	ours = _ours_raw(config, F64, "mutated").m
	rel = _rows_rel_l2(ours.flatten(0, 1), theirs.flatten(0, 1))
	assert float(rel.max()) <= RAW_FP64_MAX


@pytest.mark.parametrize("kind", ["shuffled", "mutated"])
@pytest.mark.parametrize("stats", ["cpu_mirror", "two_pass"])
@pytest.mark.parametrize("config", ["A", "B"])
def test_results_do_not_depend_on_the_step_size(config, stats, kind):
	"""Steps of 3 and 8 sequences against steps of 1, for 13 sequences, the
	last step padded with copies of the last sequence. Rows are independent
	in every operation, so only reassociation can differ."""

	start, end = _window(config)
	base_raw = _ours_raw(config, F64, kind, stats, 1)
	base_run = _ours_run(config, F64, kind, start, end, stats, 1)
	assert base_raw.debug["steps"] == N_SEQUENCES
	assert base_raw.debug["padded_peaks"] == 0

	for step in (3, 8):
		raw = _ours_raw(config, F64, kind, stats, step)
		run = _ours_run(config, F64, kind, start, end, stats, step)
		n_steps = -(-N_SEQUENCES // step)
		assert raw.debug["seqs_per_step"] == step
		assert raw.debug["steps"] == n_steps
		assert raw.debug["padded_peaks"] == n_steps * step - N_SEQUENCES
		assert (run.meta["seqs_per_step"], run.meta["steps"]) == (step, n_steps)

		y_ref = torch.cat([base_raw.y_x[:, None], base_raw.y_r], dim=1).double()
		y = torch.cat([raw.y_x[:, None], raw.y_r], dim=1).double()
		errs = {
			"m": float(_rows_rel_l2(raw.m.flatten(0, 1),
				base_raw.m.flatten(0, 1)).max()),
			"attr": float(_rows_rel_l2(run.attr, base_run.attr).max()),
			"deltas": float(((raw.deltas - base_raw.deltas).abs()
				/ _dy(base_raw).clamp_min(1)).max()),
			"y": float(((y - y_ref).abs() / y_ref.abs().clamp_min(1)).max()),
		}

		assert max(errs.values()) <= STEP_TOL, (step, errs)


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_the_rules_change_the_multipliers(config):
	"""With shuffled references every pair's multipliers differ from the
	gradient at its sequence by at least 10%, so the tests above exercise
	the rules."""

	ours = _ours_raw(config, F64, "shuffled").m.flatten(0, 1)
	rel = _rows_rel_l2(ours, _gradient(config).repeat_interleave(K, 0))
	assert float(rel.min()) >= RULES_MIN


##


@pytest.mark.parametrize("workers", [0, 2])
def test_drawn_references_are_the_injected_shuffles(workers):
	"""With no references passed, the engine draws them, in this process or
	in a spawn pool, bit for bit as the helper's shuffles, which are
	deep_lift_shap's own draw. Config B, in steps of 4: 13 sequences make 4
	steps, the last padded."""

	rs, k = 1234, 3
	X = tiny.sequences("B")
	R = tiny.shuffled_references(X, k, random_state=rs)
	engine = _engine("B", torch.float32, n_shuffles=k, random_state=rs,
		ref_workers=workers, seqs_per_step=4)
	drawn, injected = engine.raw(X), engine.raw(X, R)
	for name in ("m", "y_x", "y_r", "deltas"):
		assert torch.equal(getattr(drawn, name), getattr(injected, name)), name

	start, end = _window("B")
	drawn_run = engine.run(X, start, end)
	injected_run = engine.run(X, start, end, references=R)
	assert torch.equal(drawn_run.attr, injected_run.attr)
	assert torch.equal(drawn_run.deltas, injected_run.deltas)
	assert drawn_run.meta["references"] == "produced"
	assert injected_run.meta["references"] == "injected"
	assert drawn_run.meta["random_state"] == rs
	assert drawn_run.meta["n_shuffles"] == k


@pytest.mark.parametrize("workers", [0, 2])
def test_a_run_keeps_the_references_it_used(workers):
	"""keep_references returns the references of the kept sequences as the
	run used them, drawn or injected; the kept sequences straddle the steps,
	and the last sits in the padded step."""

	rs, k = 7, 3
	X = tiny.sequences("B")
	R = tiny.shuffled_references(X, k, random_state=rs)
	keep = [0, 3, 4, 9, 12]
	start, end = _window("B")
	engine = _engine("B", torch.float32, n_shuffles=k, random_state=rs,
		ref_workers=workers, seqs_per_step=4)
	drawn = engine.run(X, start, end, keep_references=keep)
	injected = engine.run(X, start, end, references=R,
		keep_references=numpy.array(keep))
	expected = references.onehot_to_tokens(R[keep])
	for result in (drawn, injected):
		assert result.references.dtype == numpy.uint8
		assert result.references.shape == (len(keep), k, X.shape[-1])
		numpy.testing.assert_array_equal(result.references, expected)
		assert result.meta["kept_references"] == len(keep)

	assert torch.equal(drawn.attr, injected.attr)
	assert engine.run(X, start, end, references=R).references is None
	kept = engine.run(X, start, end, references=R, keep_references=[])
	assert kept.references.shape == (0, k, X.shape[-1])

	for bad in ([3, 1], [0, 0], [-1], [13], [[1, 2]], [0.5],
			numpy.array([True, False])):
		with pytest.raises(ValueError, match="keep_references"):
			engine.run(X, start, end, references=R, keep_references=bad)


def test_the_reference_pool_starts_before_the_count_gradient(monkeypatch):
	"""The pool's first chunks are submitted before the count gradient is
	computed (and, on a GPU, before the warm-up compiles)."""

	calls = []
	real_chunks = references.RefProducer.chunks
	real_cotangent = fdls.Engine._counts_cotangent

	def chunks(self, *args, **kwargs):
		calls.append("chunks")
		return real_chunks(self, *args, **kwargs)

	def cotangent(self, length):
		calls.append("cotangent")
		return real_cotangent(self, length)

	monkeypatch.setattr(references.RefProducer, "chunks", chunks)
	monkeypatch.setattr(fdls.Engine, "_counts_cotangent", cotangent)
	X = tiny.sequences("A")[:5]
	engine = _engine("A", torch.float32, n_shuffles=2, ref_workers=0,
		seqs_per_step=2)
	result = engine.run(X, *_window("A"))
	assert calls == ["chunks", "cotangent"]
	assert result.meta["references"] == "produced"
	assert result.meta["steps"] == 3


def test_recut_cuts_consecutive_chunks_into_steps_of_the_new_size():
	"""What the engine does to the pool's chunks when its warm-up halves the
	step size, from 5 to 2 here."""

	rng = numpy.random.default_rng(0)
	refs = rng.integers(0, 4, size=(13, 3, 7), dtype=numpy.uint8)
	chunks = [(a, min(a + 5, 13), refs[a:a + 5]) for a in range(0, 13, 5)]
	got = list(fdls._recut(iter(chunks), 2))
	assert [(a, b) for a, b, _ in got] == [(a, min(a + 2, 13))
		for a in range(0, 13, 2)]
	for a, b, r in got:
		numpy.testing.assert_array_equal(r, refs[a:b])

	assert [(a, b) for a, b, _ in fdls._recut(iter(chunks), 5)] == [(0, 5),
		(5, 10), (10, 13)]
	assert [(a, b) for a, b, _ in fdls._recut(iter(chunks), 1)] == [(a, a + 1)
		for a in range(13)]
	assert list(fdls._recut(iter([]), 2)) == []
	with pytest.raises(RuntimeError, match="consecutive"):
		list(fdls._recut(iter([chunks[0], chunks[2]]), 2))


def test_engine_arguments():
	model = tiny.tiny_model("B")
	X = tiny.sequences("B")[:3]
	L = X.shape[-1]
	R = tiny.shuffled_references(X, 2)

	with pytest.raises(NotImplementedError, match="count head only"):
		fdls.Engine(model, output="profile", group=1)

	with pytest.raises(ValueError, match="output"):
		fdls.Engine(model, output="logits", group=1)

	with pytest.raises(ValueError, match="attributes one output"):
		fdls.Engine(model, group=None)

	with pytest.raises(ValueError, match="contract"):
		fdls.Engine(ControlWrapper(model), group=1)

	for bad in (2, True):
		with pytest.raises(ValueError, match="group must be"):
			fdls.Engine(model, group=bad)

	for bad in (0, -1, "big", 2.5, True):
		with pytest.raises(ValueError, match="seqs_per_step"):
			fdls.Engine(model, group=1, seqs_per_step=bad)

	for name in ("n_shuffles", "head_rows"):
		for bad in (0, 1.5, True):
			with pytest.raises(ValueError, match=name):
				fdls.Engine(model, group=1, **{name: bad})

	with pytest.raises(ValueError, match="forward_stats"):
		fdls.Engine(model, group=1, forward_stats="three_pass")

	for name, bad in (("precision", "reference"), ("precision", None),
			("compile", "default"), ("compile", "max-autotune"),
			("mem_budget_gb", 0), ("mem_budget_gb", -1.0),
			("mem_budget_gb", True), ("mem_budget_gb", "8"),
			("deterministic", 1), ("deterministic", None)):
		with pytest.raises(ValueError, match=name):
			fdls.Engine(model, group=1, **{name: bad})

	with pytest.raises(ValueError, match="random_state"):
		fdls.Engine(model, group=1, random_state=1.5)

	with pytest.raises(ValueError, match="warning_threshold"):
		fdls.Engine(model, group=1, warning_threshold="high")

	for bad in (-1, 1.5, True):
		with pytest.raises(ValueError, match="ref_workers"):
			fdls.Engine(model, group=1, ref_workers=bad)

	engine = fdls.Engine(model, group=1, n_shuffles=2, ref_workers=0)
	assert (engine.forward_stats, engine.random_state, engine.head_rows) == (
		"cpu_mirror", 0, 128)
	assert (engine.precision, engine.compile, engine.deterministic,
		engine.mem_budget_gb) == ("tf32", "none", False, 12.0)

	with_n = X.clone()
	with_n[1, :, 7] = 0
	bad_inputs = [
		(X * 2, None, "one-hot"),
		(with_n, R, "unknown"),
		(with_n, None, "unknown"),
		(X[:, :3], None, "shape"),
		(X[:0], None, "at least one"),
		(X, R[:2], "references"),
		(X, R[..., :-1], "references"),
		(X, R[:, :0], "references"),
		(X, R * 2, "one-hot"),
		(X[..., :2 * model.trimming], None, "trimming"),
	]

	for x, r, match in bad_inputs:
		with pytest.raises(ValueError, match=match):
			engine.raw(x, r)

		with pytest.raises(ValueError, match=match):
			engine.run(x, 0, x.shape[-1] if x.dim() == 3 else 1, references=r)

	for start, end in ((10, 10), (-1, 20), (0, L + 1), (50, 40)):
		with pytest.raises(ValueError, match="window"):
			engine.run(X, start, end, references=R)

	# Injected references set K, whatever n_shuffles says, and may hold N.
	R_n = R.clone()
	R_n[0, 1, :, 5] = 0
	raw = fdls.Engine(model, group=1, n_shuffles=20).raw(X, R_n)
	assert raw.m.shape == (3, 2, 4, L) and bool(torch.isfinite(raw.m).all())

	# random_state None draws a base seed once, records it, and uses it.
	drawn = fdls.Engine(model, group=1, n_shuffles=2, random_state=None,
		ref_workers=0)
	rs = drawn.random_state
	assert isinstance(rs, int) and 0 <= rs < fdls.SEED_RANGE
	result = drawn.run(X, 0, L)
	assert result.meta["random_state"] == rs
	expected = drawn.run(X, 0, L, references=tiny.shuffled_references(X, 2,
		random_state=rs))
	assert torch.equal(result.attr, expected.attr)


def test_engine_takes_numpy_inputs_and_group_none():
	"""numpy one-hots work like tensors; group None on a one-group model
	attributes that group."""

	model = tiny.tiny_model("A")
	X = tiny.sequences("A")[:3]
	R = tiny.shuffled_references(X, 2)
	expected = fdls.Engine(model, group=0).raw(X, R)
	for group, x, r in ((0, X.numpy(), R.numpy()), (None, X, R)):
		got = fdls.Engine(model, group=group).raw(x, r)
		for name in ("m", "y_x", "y_r", "deltas"):
			assert torch.equal(getattr(got, name), getattr(expected, name))


def test_run_warns_once_for_high_deltas():
	"""One summary RuntimeWarning per run when a delta exceeds the threshold,
	none for a threshold of None. A delta exceeds it only if strictly
	greater, as in tangermeme."""

	model = tiny.tiny_model("A")
	X = tiny.sequences("A")[:4]
	R = tiny.shuffled_references(X, 3)
	# float32 deltas depend on the step size through the matrix products'
	# row counts, so the largest one is taken at the step size used below.
	largest = float(fdls.Engine(model, warning_threshold=None,
		seqs_per_step=3).run(X, 0, X.shape[-1], references=R).deltas.max())

	for threshold, warns in ((0.0, True), (None, False), (1.0, False),
			(largest, False)):
		engine = fdls.Engine(model, warning_threshold=threshold,
			seqs_per_step=3)
		with warnings.catch_warnings(record=True) as caught:
			warnings.simplefilter("always")
			result = engine.run(X, 0, X.shape[-1], references=R)

		messages = [str(w.message) for w in caught
			if issubclass(w.category, RuntimeWarning)]
		summary = result.meta["deltas"]
		assert summary["pairs"] == 12 and summary["threshold"] == threshold
		assert summary["max"] == float(result.deltas.max()) == largest
		if warns:
			assert len(messages) == 1, messages
			assert re.fullmatch(r"Convergence deltas too high: \d+ of 12 pairs "
				r"> 0 \(max \S+, p50/p99/p99\.9 \S+/\S+/\S+\)", messages[0])
			assert summary["above"] == int((result.deltas > 0).sum())
		else:
			assert messages == []
			assert summary["above"] == (None if threshold is None else 0)


##


_FLAGS = (
	"torch.get_float32_matmul_precision()",
	"torch.backends.cuda.matmul.allow_tf32",
	"torch.backends.cudnn.allow_tf32",
	"torch.backends.cudnn.benchmark",
	"torch.backends.cudnn.deterministic",
)


def test_precision_mode_sets_logs_and_restores_the_flags(tmp_path,
	monkeypatch, caplog):
	"""Each mode's flags, every flag in the log, the previous flags restored
	on exit, and one Inductor cache directory per mode."""

	root = tmp_path / "inductor"
	monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(root))
	before = fdls.precision_flags()
	expected = {
		("tf32", False): ("high", True, True, True, False),
		("fp32", False): ("highest", False, False, True, False),
		("tf32", True): ("high", True, True, False, True),
		("fp32", True): ("highest", False, False, False, True),
	}

	for (mode, deterministic), values in expected.items():
		with caplog.at_level(logging.INFO, logger=fdls.logger.name):
			caplog.clear()
			with fdls.precision_mode(mode, deterministic=deterministic) as flags:
				inside = fdls.precision_flags()
				cache_dir = os.environ["TORCHINDUCTOR_CACHE_DIR"]

			logged = caplog.text

		assert tuple(flags[name] for name in _FLAGS) == values
		assert inside == flags
		expected_dir = str(root / ("fast_deep_lift_shap_" + mode))
		assert cache_dir == expected_dir == fdls.inductor_cache_dir(mode, root)
		assert flags["$TORCHINDUCTOR_CACHE_DIR"] == cache_dir
		for name in _FLAGS:
			assert name in logged

		assert "precision mode " + mode in logged and "restored" in logged
		assert fdls.precision_flags() == before

	assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(root)

	with pytest.raises(RuntimeError, match="body"), fdls.precision_mode("fp32",
			inductor_cache=False) as flags:
		assert flags["$TORCHINDUCTOR_CACHE_DIR"] == str(root)
		raise RuntimeError("body")

	assert fdls.precision_flags() == before
	with pytest.raises(ValueError, match="precision"), fdls.precision_mode(
			"bf16"):
		pass

	nested = fdls.inductor_cache_dir("fp32", root / "fast_deep_lift_shap_tf32")
	assert nested == str(root / "fast_deep_lift_shap_fp32")


def test_step_bytes_and_automatic_step_size():
	"""0.93 GB per sequence for the default architecture at K = 20, and S =
	clamp(floor(min(budget, 0.6 x available) / per sequence), 1, 32)."""

	real = dict(K=20, L=2114, C=128, H=256, n_blocks=9)
	per_peak = fdls.step_bytes(1, **real)
	L, C, H = 2114, 128, 256
	saved = 4 * L * (C + 9 * (C + H))
	fwd, bwd = 4 * L * (4 * C + H), 4 * L * (3 * C + 2 * H)
	assert per_peak == 21 * (saved + fwd + 16 * L + L) + 20 * (bwd + 16 * L)
	assert round(per_peak / 1e9, 2) == 0.93
	for S in (1, 4, 8, 32):
		assert fdls.step_bytes(S, **real) == S * per_peak

	free = 80e9
	assert fdls.auto_seqs_per_step(free, 8.0, per_peak) == 8
	assert fdls.auto_seqs_per_step(free, 12.0, per_peak) == 12
	assert fdls.auto_seqs_per_step(10e9, 12.0, per_peak) == 6
	assert fdls.auto_seqs_per_step(free, 0.1, per_peak) == 1
	assert fdls.auto_seqs_per_step(1e12, 1e3, per_peak) == 32

	model = tiny.tiny_model("A")
	engine = fdls.Engine(model, seqs_per_step=5)
	assert engine.step_size(13, 256, 5) == 5 and engine.step_size(3, 256, 5) == 3
	assert fdls.Engine(model).step_size(100, 256, 5) == fdls.AUTO_SEQS_PER_STEP
	w = engine.weights
	assert engine.step_estimate(4, 5, 256) == fdls.step_bytes(4, 5, 256, w.C,
		w.H, 3)


def test_compile_kwargs_follow_what_torch_supports(monkeypatch):
	"""Each engine keeps its compiled code apart where torch.compile takes
	isolate_recompiles, and compiles without it elsewhere."""

	calls = []
	monkeypatch.setattr(torch, "compile", lambda fn, **kwargs:
		calls.append(kwargs) or fn)
	model = tiny.tiny_model("A")
	for supported in (True, False):
		calls.clear()
		monkeypatch.setattr(fdls, "_compile_isolates_recompiles",
			lambda: supported)
		engine = fdls.Engine(model, compile="none")
		engine._compiled(fdls.forward_save)
		assert calls[-1]["fullgraph"] is True and calls[-1]["dynamic"] is False
		assert calls[-1]["mode"] == "default"
		assert ("isolate_recompiles" in calls[-1]) is supported
		assert ("recompile_limit" in calls[-1]) is supported


##


def test_audit_peaks_are_evenly_spaced_and_distinct():
	from cherimoya.fast_deep_lift_shap.checks import audit_peaks

	assert audit_peaks(10, 8).tolist() == [0, 1, 3, 4, 5, 6, 8, 9]
	assert audit_peaks(4999, 8).tolist() == [0, 714, 1428, 2142, 2856, 3570,
		4284, 4998]
	assert audit_peaks(5, 8).tolist() == [0, 1, 2, 3, 4]
	assert audit_peaks(5, 0).tolist() == [] and audit_peaks(0, 8).tolist() == []
	assert audit_peaks(1, 1).tolist() == [0]
	for n in range(1, 40):
		for k in range(1, n + 1):
			peaks = audit_peaks(n, k)
			assert len(numpy.unique(peaks)) == k and peaks[0] == 0
			assert peaks[-1] == (n - 1 if k > 1 else 0)


def test_the_audit_comparison_bounds_and_warning():
	from cherimoya.fast_deep_lift_shap.checks import compare_attributions

	rng = numpy.random.default_rng(0)
	stock = rng.normal(size=(6, 4, 50))
	same = compare_attributions(stock, stock)
	assert same["pass"] and not same["warn"] and same["max_rel_l2"] == 0.0
	assert same["min_pearson"] == pytest.approx(1)

	# A median relative L2 of 5e-4: within the bounds, but it warns.
	noise = rng.normal(size=stock.shape)
	noise *= 5e-4 * (numpy.linalg.norm(stock.reshape(6, -1), axis=1)
		/ numpy.linalg.norm(noise.reshape(6, -1), axis=1))[:, None, None]
	near = compare_attributions(stock + noise, stock)
	assert near["pass"] and near["warn"]
	assert near["median_rel_l2"] == pytest.approx(5e-4, rel=1e-6)

	# A large constant offset: the centred values decorrelate (Pearson below
	# 0.9999) while the relative L2 stays under 1e-2, so Pearson alone fails.
	offset = stock.copy()
	offset[2] = 100.0 + 0.1 * rng.normal(size=offset[2].shape)
	ours = offset.copy()
	ours[2] += 0.02 * rng.normal(size=ours[2].shape)
	pearson_only = compare_attributions(ours, offset)
	assert pearson_only["rel_l2"][2] <= 1e-2
	assert pearson_only["pearson"][2] < 0.9999
	assert not pearson_only["pass"]
	assert pearson_only["failed_positions"] == [2]

	nan = stock.copy()
	nan[1, 0, 0] = numpy.nan
	assert compare_attributions(nan, stock)["failed_positions"] == [1]
	with pytest.raises(ValueError, match="equal, non-empty"):
		compare_attributions(stock[:2], stock)
