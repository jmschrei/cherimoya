"""Tests of the fast engine's backward rules, one at a time, against the
tangermeme and cherimoya functions they replace, on the CPU.

Each rule gets the same inputs as the function it replaces, in float64:

* `gelu_rescale` against tangermeme's `_nonlinear` at |dz| = 0, 5e-7,
  9.999e-7, exactly 1e-6, 1.0001e-6, 1e-3 and 1: the same branch (the
  gradient where |dz| < 1e-6, strictly; the secant otherwise), within 1 ulp.
* `ln_rule` against tangermeme's `_layer_normalization_helper` on
  cherimoya's `_ConvNormView`, as `conv_norm_op` calls it: <= 1e-13.
* `dw3` and its transpose against cherimoya's `_cheri_conv` and autograd
  through it: <= 1e-14.
* `counts_cotangent` against autograd through the real
  ``LogCountWrapper(ControlWrapper(model))``: equal.
* The window of hypothetical attributions against tangermeme's
  `hypothetical_attributions`, sliced: equal.
* The mean over references against tangermeme's stack-and-mean of the
  full-length attributions, sliced: equal.

tangermeme's rules read the activations off the module they correct, so
each reference call gets a stub holding tangermeme's joint layout: one copy
of the sequence per pair, then the references. Only the sequence half of a
rule's output is compared, since that is all the engine computes.

GELU on the CPU. The CPU kernels for GELU and its gradient compute a
vectorized body and a scalar tail, which round differently in the last bit,
and which elements fall in the tail depends on the tensor's shape. So the
GELU rule's test keeps every contiguous run a multiple of 16 elements and
below ATen's parallel grain size, where every element takes the vectorized
path on both sides. The composite tests run on the tiny models' real shapes
and are bounded by relative error.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from types import SimpleNamespace

from tangermeme._deep_lift_utils import _layer_normalization_helper
from tangermeme._deep_lift_utils import _nonlinear

from cherimoya import ControlWrapper
from cherimoya import LogCountWrapper
from cherimoya.cheri import CONV_NORM_EPS
from cherimoya.cheri import _cheri_conv
from cherimoya.deep_lift_shap import _ConvNormView
from cherimoya.deep_lift_shap import conv_norm_op
from cherimoya.fast_deep_lift_shap import engine as fdls

from . import fast_deep_lift_shap_helpers as tiny


F64 = torch.float64
DTYPES = [torch.float32, torch.float64]
CONFIG_NAMES = list(tiny.CONFIGS)

# Sequences and references per sequence in the composite tests.
S, K = 2, 3

LN_TOL = 1e-13
DW3_TOL = 1e-14
COMPOSITE_TOL = 1e-13


@contextlib.contextmanager
def _threads(n):
	previous = torch.get_num_threads()
	torch.set_num_threads(n)
	try:
		yield
	finally:
		torch.set_num_threads(previous)


def _ulps(a, b):
	"""The elementwise distance in units in the last place between two
	float64 tensors, with -0.0 equal to +0.0."""

	assert a.dtype == b.dtype == F64
	lowest = torch.iinfo(torch.int64).min

	def line(t):
		bits = t.contiguous().view(torch.int64)
		return torch.where(bits < 0, lowest - bits, bits).to(F64)

	return (line(a) - line(b)).abs()


def _rel_l2(a, b):
	"""The largest relative L2 distance over the rows of a and b."""

	a, b = a.flatten(1), b.flatten(1)
	return float(((a - b).norm(dim=1) / b.norm(dim=1)).max())


def _joint(t):
	"""Rows [x_0..x_{S-1}, r_00..r_{S-1,K-1}] in tangermeme's joint layout:
	one copy of each sequence per pair, then the references."""

	return torch.cat([t[:S].repeat_interleave(K, dim=0), t[S:]])


def _joint_batch(config, dtype):
	"""S sequences (one random, the AT-rich one), then their K dinucleotide
	shuffles: (S + S*K, 4, L)."""

	X = tiny.sequences(config)[[0, 12]]
	refs = tiny.shuffled_references(X, K)
	return torch.cat([X.float(), refs.flatten(0, 1)]).to(dtype)


def _gelu_reference(z_joint, g_joint):
	"""tangermeme's `_nonlinear` on a GELU(tanh) stub over the joint rows:
	the result, the gradient nn.GELU's autograd gives, and its output."""

	leaf = z_joint.detach().clone().requires_grad_()
	with torch.enable_grad():
		out = torch.nn.GELU(approximate="tanh")(leaf)
		grad_input, = torch.autograd.grad(out, leaf, g_joint)

	stub = SimpleNamespace(input=z_joint, output=out.detach())
	result, = _nonlinear(stub, (grad_input,), (g_joint,))
	return result, grad_input, out.detach()


##


DZ = [0.0, 5e-7, 9.999e-7, 1e-6, 1.0001e-6, 1e-3, 1.0]


@pytest.mark.parametrize("dz", DZ)
def test_gelu_rescale_matches_tangermeme_nonlinear(dz):
	"""|z_x - z_r| = dz everywhere, with random signs. The gradient branch is
	the strict |dz| < 1e-6.

	For |dz| exactly 1e-6, z_x - z_r is exact only for base points below
	about 1.9e-6, so those are multiples of 2**-40 below 0.85e-6; elsewhere
	they are uniform in [-4, 4]. The expected branch's value must be within
	1 ulp of ours and of tangermeme's, and ours must be more than 1 ulp from
	the other branch wherever the two branches are more than 2 ulps apart.
	"""

	n_s, n_k, inner = 3, 4, (16, 8)
	P = n_s * n_k
	gen = torch.Generator().manual_seed(int(dz * 1e7) + 7)
	if dz == 1e-6:
		zx = torch.randint(-900_000, 900_001, (n_s, *inner),
			generator=gen).to(F64) * 2.0**-40
	else:
		zx = torch.rand((n_s, *inner), generator=gen, dtype=F64) * 8 - 4

	sign = torch.randint(0, 2, (n_s, n_k, *inner), generator=gen).to(F64)
	zr = zx.unsqueeze(1) - (sign * 2 - 1) * dz
	din = zx.unsqueeze(1) - zr
	small = dz < fdls.RESCALE_EPS
	if dz == 1e-6:
		assert torch.equal(din.abs(), torch.full_like(din, 1e-6))

	assert bool(((din.abs() < 1e-6) == small).all())

	G = torch.randn((n_s, n_k, *inner), generator=gen, dtype=F64)
	G_ref_half = torch.randn((P, *inner), generator=gen, dtype=F64)
	z_joint = torch.cat([zx.repeat_interleave(n_k, dim=0), zr.flatten(0, 1)])
	g_joint = torch.cat([G.flatten(0, 1), G_ref_half])
	ref, grad_input, out = _gelu_reference(z_joint, g_joint)
	ref = ref[:P]

	z = torch.cat([zx, zr.flatten(0, 1)])
	ours = fdls.gelu_rescale(G, z, n_s, n_k).reshape(P, *inner)
	assert ours.dtype == F64 and bool(torch.isfinite(ours).all())

	tangent = grad_input[:P]
	secant = g_joint[:P] * ((out[:P] - out[P:]) / (z_joint[:P] - z_joint[P:]))
	expected, other = (tangent, secant) if small else (secant, tangent)
	assert float(_ulps(ref, expected).max()) <= 1
	assert float(_ulps(ours, expected).max()) <= 1
	assert float(_ulps(ours, ref).max()) <= 1
	if dz == 0:
		assert bool(torch.isnan(other).all())  # 0/0: only the gradient
	else:
		far = _ulps(expected, other) > 2
		assert float(far.to(F64).mean()) >= 0.9
		assert bool((_ulps(ours, other)[far] > 1).all())

	# The sequence half reads only the sequence half's gradients.
	again, _, _ = _gelu_reference(z_joint, torch.cat([G.flatten(0, 1),
		-3 * G_ref_half]))
	assert torch.equal(again[:P], ref)


def _ln_reference(y, Gn, G_ref_half):
	"""`conv_norm_op`'s call: `_layer_normalization_helper` on a
	`_ConvNormView` of the joint rows, sequence half returned."""

	view = _ConvNormView(_joint(y))
	assert view.eps == CONV_NORM_EPS and view.weight is None
	assert view.normalized_shape == list(y.shape[1:])
	result, = _layer_normalization_helper(view, None,
		(torch.cat([Gn.flatten(0, 1), G_ref_half]),))
	return result[:S * K]


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_ln_rule_matches_layer_normalization_helper(config):
	"""Every block's real convolution output from a float64 forward, plus a
	synthetic y with means up to 50 and standard deviations down to 0.01,
	so that the centring and the variance term matter."""

	cfg = tiny.CONFIGS[config]
	w = fdls.CheriWeights.from_model(tiny.tiny_model(config, dtype=F64))
	with torch.no_grad():
		fwd = fdls.model_forward(_joint_batch(config, F64), w,
			forward_stats="cpu_mirror", group=cfg.group)

	gen = torch.Generator().manual_seed(11)
	L, C = cfg.length, w.C
	R = S + S * K
	ys = [saved.y for saved in fwd.saved]
	offset = torch.rand((R, 1, 1), generator=gen, dtype=F64) * 100 - 50
	scale = 10 ** (torch.rand((R, 1, 1), generator=gen, dtype=F64) * 3 - 2)
	ys.append(offset + scale * torch.randn((R, L, C), generator=gen,
		dtype=F64))

	errs = []
	for y in ys:
		mu, _, v = fdls.rule_stats(y, w.eps)
		Gn = torch.randn((S, K, L, C), generator=gen, dtype=F64)
		G_ref_half = torch.randn((S * K, L, C), generator=gen, dtype=F64)
		ours = fdls.ln_rule(Gn, y, mu.view(R), v.view(R), S, K)
		assert ours.shape == (S, K, L, C) and ours.dtype == F64

		ref = _ln_reference(y, Gn, G_ref_half)
		assert torch.equal(_ln_reference(y, Gn, 2 * G_ref_half + 1), ref)
		errs.append(_rel_l2(ours.flatten(0, 1), ref))

	assert max(errs) <= LN_TOL


@pytest.mark.parametrize("shape", [(2114, 128), (100, 8)])
@pytest.mark.parametrize("d", [2**i for i in range(9)])
def test_dw3_and_its_transpose_match_cheri_conv(d, shape):
	"""`dw3` against `_cheri_conv`, and `dw3_t` against autograd through
	`_cheri_conv`, which is how `conv_norm_op` takes the rule back to the
	block's input. L=100 with d >= 128 puts both outer taps outside the
	sequence everywhere."""

	L, C = shape
	gen = torch.Generator().manual_seed(d)
	cw = torch.randn((3, C), generator=gen, dtype=F64)
	h = torch.randn((S + S * K, L, C), generator=gen, dtype=F64)
	assert _rel_l2(fdls.dw3(h, cw, d), _cheri_conv(h, cw, d)) <= DW3_TOL

	G = torch.randn((S, K, L, C), generator=gen, dtype=F64)
	leaf = torch.randn((S * K, L, C), generator=gen, dtype=F64,
		requires_grad=True)
	with torch.enable_grad():
		expected, = torch.autograd.grad(_cheri_conv(leaf, cw, d), leaf,
			G.flatten(0, 1))

	ours = fdls.dw3_t(G, cw, d)
	assert ours.shape == G.shape
	assert _rel_l2(ours.flatten(0, 1), expected) <= DW3_TOL


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("config, group", [("A", 0), ("B", 0), ("B", 1),
	("B", None), ("C", 0)])
def test_counts_cotangent_is_autograd_through_the_real_head(config, group,
	dtype):
	"""The gradient of the attributed count with respect to the last block's
	output, from the real module run as `deep_lift_shap` runs it. Config B
	has a control track and two groups; group None attributes target 0."""

	model = tiny.tiny_model(config, dtype=dtype)
	w = fdls.CheriWeights.from_model(model)
	X1h = _joint_batch(config, dtype)
	L, T = X1h.shape[-1], w.T
	captured = {}
	handle = model.blocks[-1].register_forward_hook(lambda m, args, out:
		captured.update(hL=out))
	try:
		with torch.enable_grad():
			y = LogCountWrapper(ControlWrapper(model), group=group)(
				X1h.clone().requires_grad_())[:, 0]
			expected, = torch.autograd.grad(y.sum(), captured["hL"])
	finally:
		handle.remove()

	assert expected.shape == (len(X1h), L, w.C)
	for rows in (len(X1h), fdls.DEFAULT_HEAD_ROWS):
		ours = fdls.counts_cotangent(w, group=group, length=L, rows=rows)
		assert ours.shape == (L, w.C) and ours.dtype == dtype
		assert ours.is_contiguous()
		assert torch.equal(expected, ours.expand_as(expected))

	# The count Linear's row over the trimmed mean, zero outside [T, L - T).
	row = w.Wl[0 if group is None else group, :w.C]
	assert torch.equal(ours[T:L - T], (row / (L - 2 * T)).expand(L - 2 * T,
		w.C))
	assert not bool(ours[:T].any()) and not bool(ours[L - T:].any())


def test_counts_cotangent_arguments():
	w = fdls.CheriWeights.from_model(tiny.tiny_model("B"))
	with torch.no_grad():
		# The engine runs under no_grad; the cotangent enables grad itself.
		assert fdls.counts_cotangent(w, group=1, length=200).shape == (200, 8)

	with torch.inference_mode(), pytest.raises(RuntimeError, match="inference"):
		fdls.counts_cotangent(w, group=1, length=200)

	with pytest.raises(ValueError, match="rows"):
		fdls.counts_cotangent(w, group=1, length=200, rows=0)

	with pytest.raises(ValueError, match="group"):
		fdls.counts_cotangent(w, group=2, length=200)

	with pytest.raises(ValueError, match="trimming"):
		fdls.counts_cotangent(w, group=1, length=2 * w.T)


##


def _window(L, W):
	"""The attribution window `cherimoya attribute` takes."""

	start = L // 2 - W // 2
	return start, start + W


def _pairs(L, n_s, n_k, dtype, seed):
	"""Joint one-hot rows with a few N columns in the references, and
	multipliers with exact zeros of both signs, for n_s * n_k pairs."""

	gen = torch.Generator().manual_seed(seed)
	X = tiny.random_onehot(n_s, L, seed=seed).to(dtype)
	refs = tiny.random_references(X, n_k, seed=seed + 1).to(dtype).flatten(0,
		1)
	refs[:, :, torch.randint(0, L, (5,), generator=gen)] = 0
	m = torch.randn((n_s * n_k, 4, L), generator=gen, dtype=F64).to(dtype)
	m[m.abs() < 0.05] = 0.0
	m[:, :, ::7] *= -1
	y = torch.randn(n_s + n_s * n_k, generator=gen, dtype=F64).to(dtype)
	return torch.cat([X, refs]), m, y


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("L, W", [(2114, 400), (256, 64), (200, 50),
	(2114, 100)])
def test_window_hyp_is_hypothetical_attributions_sliced(L, W, dtype):
	"""For one-hot (or N) references both sides round once, to fl(m_i -
	m_b), so the window equals tangermeme's full-length projection, sliced
	afterwards."""

	n_s, n_k = 2, 3
	X1h, m, y = _pairs(L, n_s, n_k, dtype, seed=L + W)
	start, end = _window(L, W)
	hyp, _ = fdls.epilogue(m, X1h, y[:n_s], y[n_s:].view(n_s, n_k), start,
		end, n_s, n_k)
	assert hyp.shape == (n_s, n_k, 4, W) and hyp.dtype == dtype
	assert hyp.is_contiguous()

	Xp = X1h[:n_s].repeat_interleave(n_k, dim=0)
	full, = tiny.TDLS.hypothetical_attributions((m,), (Xp,), (X1h[n_s:],))
	assert torch.equal(hyp.flatten(0, 1), full[..., start:end])


def test_epilogue_deltas_are_tangermemes_expression():
	"""|(y_x - y_r) - sum((x - r) * m)| over the full length, bitwise."""

	for dtype in DTYPES:
		n_s, n_k, L = 3, 4, 256
		X1h, m, y = _pairs(L, n_s, n_k, dtype, seed=3)
		_, delta = fdls.epilogue(m, X1h, y[:n_s], y[n_s:].view(n_s, n_k), 96,
			160, n_s, n_k)
		assert delta.shape == (n_s, n_k) and delta.dtype == dtype

		y_joint = torch.cat([y[:n_s].repeat_interleave(n_k), y[n_s:]])
		Xp, r = X1h[:n_s].repeat_interleave(n_k, dim=0), X1h[n_s:]
		expected = abs(torch.sub(*torch.chunk(y_joint, 2))
			- torch.sum((Xp - r) * m, dim=(1, 2)))
		assert torch.equal(delta.flatten(), expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_k, L, W", [(20, 2114, 400), (4, 256, 64),
	(5, 200, 50), (20, 2114, 100), (5, 256, 256), (3, 2114, 2100)])
def test_mean_over_references_is_tangermemes_stack_mean_sliced(n_k, L, W,
	dtype):
	"""tangermeme averages a sequence's full-length hypothetical
	attributions, ``torch.stack(...).mean(dim=0)``, on the CPU; the engine
	averages the windows. Equal at 1, 4 and 8 threads, for the default
	shape, windows that are not a multiple of 16 wide, the full length, and
	a window ending 7 positions before the end."""

	n_s = 2
	X1h, m, y = _pairs(L, n_s, n_k, dtype, seed=n_k * L + W)
	start, end = _window(L, W)
	hyp, _ = fdls.epilogue(m, X1h, y[:n_s], y[n_s:].view(n_s, n_k), start,
		end, n_s, n_k)
	Xp = X1h[:n_s].repeat_interleave(n_k, dim=0)
	full, = tiny.TDLS.hypothetical_attributions((m,), (Xp,), (X1h[n_s:],))
	attr_ = list(full)

	ours = fdls.mean_over_references(hyp, start, L)
	assert ours.shape == (n_s, 4, W) and ours.dtype == dtype
	for threads in (1, 4, 8):
		with _threads(threads):
			ref = torch.stack([torch.stack(attr_[s * n_k:(s + 1) * n_k]).mean(
				dim=0)[:, start:end] for s in range(n_s)])

		assert torch.equal(ours, ref), "{} threads".format(threads)


def test_epilogue_and_mean_arguments():
	X1h, m, y = _pairs(64, 2, 3, F64, seed=0)
	for start, end in ((10, 10), (-1, 20), (0, 65), (40, 30)):
		with pytest.raises(ValueError, match="window"):
			fdls.epilogue(m, X1h, y[:2], y[2:].view(2, 3), start, end, 2, 3)

	hyp, _ = fdls.epilogue(m, X1h, y[:2], y[2:].view(2, 3), 20, 40, 2, 3)
	with pytest.raises(ValueError, match="window"):
		fdls.mean_over_references(hyp, 50, 64)

	with pytest.raises(ValueError, match="CPU"):
		fdls.mean_over_references(hyp.to("meta"), 20, 64)

	with pytest.raises(ValueError, match="rows"):
		fdls.split_rows(X1h[:-1], 2, 3)


##


def _reference_block_backward(G, G_ref_half, h, u, block, rs):
	"""One CheriBlock's backward as tangermeme runs it, on the joint rows,
	sequence half returned: ordinary gradients through the residual, the
	scale and both Linears, tangermeme's `_nonlinear` at the GELU, and
	cherimoya's own `conv_norm_op` at the fused convolution and norm."""

	P = S * K
	g_out = torch.cat([G.flatten(0, 1), G_ref_half])
	u_joint = _joint(u)
	with torch.enable_grad():
		q = torch.zeros_like(u_joint, requires_grad=True)
		g_q, = torch.autograd.grad(F.linear(q, block.W2) * rs, q, g_out)

	g_u, _, _ = _gelu_reference(u_joint, g_q)
	with torch.enable_grad():
		n = torch.zeros(u_joint.shape[:-1] + (block.W1.shape[1],),
			dtype=u.dtype, requires_grad=True)
		g_n, = torch.autograd.grad(F.linear(n, block.W1), n, g_u)

	stub = SimpleNamespace(input=_joint(h), conv_weight=block.cw,
		dilation=block.d)
	g_h, = conv_norm_op(stub, None, (g_n,))
	return (g_out + g_h)[:P]


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_bwd_block_composes_the_reference_rules(config):
	"""Each block of a float64 tiny model, on its real inputs, against the
	block's backward as tangermeme runs it. The count gradient reaches the
	last block broadcast over the pairs, never copied K-fold."""

	cfg = tiny.CONFIGS[config]
	w = fdls.CheriWeights.from_model(tiny.tiny_model(config, dtype=F64))
	gen = torch.Generator().manual_seed(5)
	L, C = cfg.length, w.C
	errs = []
	with torch.no_grad():
		z0 = fdls.stem_forward(_joint_batch(config, F64), w.W0, w.b0)
		_, saved = fdls.forward_save(z0, w.blocks, w.rs, w.eps, "cpu_mirror")
		for i, block in enumerate(w.blocks):
			h, _ = fdls.forward_save(z0, w.blocks[:i], w.rs, w.eps,
				"cpu_mirror")
			G = torch.randn((S, K, L, C), generator=gen, dtype=F64)
			ours = fdls.bwd_block(G, saved[i], block, w.rs, S, K)
			assert ours.shape == (S, K, L, C) and ours.dtype == F64

			G_ref_half = torch.randn((S * K, L, C), generator=gen, dtype=F64)
			ref = _reference_block_backward(G, G_ref_half, h, saved[i].u,
				block, w.rs)
			errs.append(_rel_l2(ours.flatten(0, 1), ref))

			if i == len(w.blocks) - 1:
				G0 = fdls.counts_cotangent(w, group=cfg.group, length=L)
				broadcast = fdls.bwd_block(G0.expand(S, K, L, C), saved[i],
					block, w.rs, S, K)
				copied = fdls.bwd_block(G0.repeat(S, K, 1, 1), saved[i],
					block, w.rs, S, K)
				assert torch.equal(broadcast, copied)

	assert max(errs) <= COMPOSITE_TOL


@pytest.mark.parametrize("config", CONFIG_NAMES)
def test_stem_rule_and_input_multipliers(config):
	"""`stem_rule` against `_nonlinear` on the stem GELU, and
	`input_multipliers` against autograd through iconv. Shuffled references
	take the secant almost everywhere; r = x takes the gradient
	everywhere."""

	cfg = tiny.CONFIGS[config]
	w = fdls.CheriWeights.from_model(tiny.tiny_model(config, dtype=F64))
	gen = torch.Generator().manual_seed(9)
	L, C = cfg.length, w.C
	X1h = _joint_batch(config, F64)
	same = torch.cat([X1h[:S], X1h[:S].repeat_interleave(K, dim=0)])
	for rows in (X1h, same):
		with torch.no_grad():
			z0 = fdls.stem_forward(rows, w.W0, w.b0)
			G = torch.randn((S, K, L, C), generator=gen, dtype=F64)
			ours = fdls.stem_rule(G, z0, S, K)

		assert ours.shape == (S * K, C, L) and ours.is_contiguous()
		g_joint = torch.cat([G.flatten(0, 1), torch.randn((S * K, L, C),
			generator=gen, dtype=F64)]).transpose(1, 2)
		ref, _, _ = _gelu_reference(_joint(z0), g_joint)
		assert _rel_l2(ours, ref[:S * K]) <= COMPOSITE_TOL

		with torch.no_grad():
			m = fdls.input_multipliers(ours, rows[S:], w.W0)

		leaf = rows[S:].clone().requires_grad_()
		with torch.enable_grad():
			expected, = torch.autograd.grad(F.conv1d(leaf, w.W0, w.b0,
				padding=10), leaf, ours)

		assert m.shape == (S * K, 4, L) and torch.equal(m, expected)
