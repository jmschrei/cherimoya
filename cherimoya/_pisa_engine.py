# _pisa_engine.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
A fast PISA engine for the profile logits of a Cherimoya model.

`tangermeme.pisa.pisa` runs DeepLIFT/SHAP once for every output position.
For each sequence and each of its references, it forwards `batch_size`
copies of the sequence and `batch_size` copies of the reference, and then,
for each block of `batch_size` output positions, runs a backward over both
halves of that batch through backward hooks. This engine computes the same
multipliers with the forward and the rules of
`cherimoya._deep_lift_shap_engine`:

* The forward runs once per sequence, over the sequence and its K
  references: K + 1 rows, where tangermeme forwards 2 * batch_size rows for
  each reference.
* The terms of each rule that depend only on the activations, the rescale
  rule's multiplier at each GELU and the normalization rule's terms, are
  computed once per sequence-reference pair and shared by every output
  position.
* The backward runs over the sequence half of each pair only, for a chunk
  of B output positions at once: K * B rows. The gradient of an output
  position at the final residual stream, which every pair shares, is the
  profile convolution's taps for the attributed channel, placed around that
  position.
* The rules are linear in the gradient, so each normalization rule's
  gradient splits into a local part and a dense part, a coefficient times
  1 and a coefficient times its variance term. The local part of each
  output position runs over the positions it reaches, its seed grown by
  each block's dilation; the dense parts' fields run through the blocks
  below once per sequence, and each output position adds them, times its
  coefficients.
* The averaged hypothetical attributions are linear in each pair's
  multipliers but for the reference's own base at each position, so iconv's
  input gradient runs once on the gradient summed over the pairs, and one
  kernel computes, for each pair, only the multiplier of its reference's
  base and its delta (`stem_backward`).

The rules are tangermeme's, called as tensor functions: `_rescale` at every
GELU, and `_layer_normalization_terms` and `_layer_normalization_grad`, the
rule `cherimoya.deep_lift_shap.conv_norm_op` applies, at every normalization.
The references are tangermeme's `dinucleotide_shuffle`, drawn as `pisa`
draws them.

This module is private. `cherimoya.pisa.pisa` runs it as its fast path, for
float32 Cherimoya models only, and compares it with tangermeme's `pisa` on
every call that uses it.
"""

import contextlib
import gc
import inspect
import logging
import math
import time
import warnings

from typing import NamedTuple

import torch
import torch.nn.functional as F

from tangermeme._deep_lift_utils import _layer_normalization_grad
from tangermeme._deep_lift_utils import _layer_normalization_terms
from tangermeme._deep_lift_utils import _rescale
from tangermeme.deep_lift_shap import hypothetical_attributions
from tangermeme.ersatz import dinucleotide_shuffle
from tqdm import tqdm

from ._deep_lift_shap_engine import BlockWeights
from ._deep_lift_shap_engine import FREE_MEMORY_FRACTION
from ._deep_lift_shap_engine import ICONV_PADDING
from ._deep_lift_shap_engine import MEM_BUDGET
from ._deep_lift_shap_engine import RECOMPILE_LIMIT
from ._deep_lift_shap_engine import dw3
from ._deep_lift_shap_engine import forward_save
from .cheri import CONV_NORM_EPS
from .cheri import HAS_TRITON

if HAS_TRITON:
	import triton
	import triton.language as tl


logger = logging.getLogger(__name__)

FCONV_PADDING = 37

CPU_OUTPUTS_PER_CHUNK = 16
MAX_OUTPUTS_PER_CHUNK = 128

FUSED_MAX_FILTERS = 128
"""The most filters `backward_pass` runs `block_backward` for. At 256 the
kernel's hidden tile leaves too little shared memory and it ran slower than
the torch ops it replaces."""

POSITION_FILTERS = (32, 256)
"""The fewest and the most filters for which `Engine` runs the blocks below
the top one by position (`position_backward`), one pair at a time. Above 256
the tiles are untested."""

MAX_POSITION_OUTPUTS = 4096

POSITION_MIN_OUTPUTS = 2
"""The fewest output positions per chunk, per filter, for which `Engine`
runs the blocks by position: each position's matrix costs 4C^3 and is
shared by the chunk's rows that cover it."""

POSITION_MIN_CHUNK = ((32, 768), (64, 256))
"""(C, n): for models of at most C filters, the fewest output positions per
chunk for which `Engine` runs the blocks by position, when that is more
than `POSITION_MIN_OUTPUTS` per filter. One pair per call launches every
kernel of the backward once per pair, and a row saves less at fewer
filters: by position was slower at 512 outputs and faster at 768 with 32
filters, slower at 192 and faster at 256 with 64."""

FOLD_MAX_FILTERS = 64
"""The most filters `block_backward` builds the gradient at a block's output
in its kernel for, from the block above's terms. At 128 the kernel already
sits at 255 registers, and the folded loads took as long as the separate
shift kernel they replace."""


class BlockTerms(NamedTuple):
	"""What one block's backward reads for each of a sequence's K pairs,
	with an axis of length one for the output positions."""

	slope: torch.Tensor  # (K, 1, L, H) the rescale rule's multiplier
	v_avg: torch.Tensor  # (K, 1, 1, 1) the normalization rule's terms
	a_sum: torch.Tensor  # (K, 1, L, C)
	variance_term: torch.Tensor  # (K, 1, L, C)


class Result(NamedTuple):
	"""`Engine.run`'s output: CPU tensors in the model's dtype."""

	attr: torch.Tensor  # (N, n_outputs, 4, L) hypothetical, averaged
	deltas: torch.Tensor  # (N, K, n_outputs) convergence deltas
	y_x: torch.Tensor  # (N, n_outputs) the attributed outputs
	y_r: torch.Tensor  # (N, K, n_outputs) the references' outputs


def rescale_slope(z):
	"""The rescale rule's multiplier through GELU(tanh) for each pair.

	z (K + 1, ...) is the GELU's input over a sequence, then its K
	references. The result, (K, ...), is the secant (g(z_x) - g(z_r)) /
	(z_x - z_r), or the derivative at z_x where |z_x - z_r| < 1e-6: what
	tangermeme's `_rescale` returns for a gradient of one. Multiplying a
	gradient by it is `_rescale` applied to that gradient, as
	`cherimoya._deep_lift_shap_engine.gelu_rescale` applies it.
	"""

	g = F.gelu(z, approximate="tanh")
	derivative = torch.ops.aten.gelu_backward(torch.ones_like(z[:1]), z[:1],
		approximate="tanh")
	return _rescale(z[:1] - z[1:], g[:1] - g[1:], derivative, 1.0)


def rule_terms(saved, z0):
	"""Each block's `BlockTerms` and the stem GELU's multiplier, (K, 1, C,
	L), from what `forward_save` kept for a sequence and its K references.

	The normalization terms are tangermeme's `_layer_normalization_terms`
	over each row's (L, C) plane, with the sequence row broadcast against
	its K reference rows, as
	`cherimoya._deep_lift_shap_engine.ln_rule` computes them. On CUDA this
	function runs compiled.
	"""

	terms = []
	for s in saved:
		mu, v = s.mu[:, None, None], s.v[:, None, None]
		ln = _layer_normalization_terms(s.y[:1], mu[:1], v[:1], s.y[1:],
			mu[1:], v[1:], 2)
		terms.append(BlockTerms(rescale_slope(s.u).unsqueeze(1),
			*(t.unsqueeze(1) for t in ln)))

	return tuple(terms), rescale_slope(z0).unsqueeze(1)


def shift(x, offset, W, dim=-2):
	"""x moved along `dim` and cut to W positions: out[p] = x[p + offset]
	where 0 <= p + offset < x.shape[dim], and zero elsewhere. offset is a
	0-d int64 tensor, so a compiled caller does not recompile for a new
	one."""

	n = x.shape[dim]
	index = torch.arange(W, device=x.device) + offset
	inside = ((index >= 0) & (index < n)).view(W, *[1] * (-1 - dim))
	return torch.where(inside, x.index_select(dim, index.clamp(0, n - 1)), 0)


def windows(lo, W, dilations):
	"""The windows of the local part of the backward, (lo, W) each, lo a
	0-d int64 tensor: output b's window is [lo + b, lo + b + W). The seed's
	first, then the one after each block in `dilations` (top first), the
	one before grown by the block's dilation on either side. They are not
	cut to [0, L): what lies outside is zeroed where it is made."""

	spans, grown = [(lo, W)], 0
	for d in dilations:
		grown += d
		spans.append((lo - grown, W + 2 * grown))

	return spans


def positions(lo, B, W):
	"""The sequence positions of B output positions' windows, (B, W) int64:
	lo + b + l, lo a 0-d int64 tensor."""

	return (lo + torch.arange(B, device=lo.device)[:, None]
		+ torch.arange(W, device=lo.device))


if HAS_TRITON:
	@triton.jit
	def _tf32_round(x):
		# fp32 -> tf32 with round-to-nearest-even on the 13 dropped bits, as
		# cuBLAS's TF32 GEMMs round their operands; the tensor cores
		# truncate what they are given.
		bits = x.to(tl.uint32, bitcast=True)
		bits = (bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000
		return bits.to(tl.float32, bitcast=True)


	@triton.jit
	def _tap(base, at, Wp, mask_l, offs_c, mask_c, stride_l, stride_c, v):
		# v * g at the positions `at` of the block above's window, zero
		# outside it, as `shift` and the product order of `shift_add` give it.
		inside = (at >= 0) & (at < Wp) & mask_l
		g = tl.load(base + at[:, None] * stride_l + offs_c[None, :] * stride_c,
			mask=inside[:, None] & mask_c[None, :], other=0.0)
		return tl.where(inside[:, None], v * g, 0.0)


	@triton.jit
	def _block_backward_kernel(G_ptr, P_ptr, V_ptr, CW_ptr, W2_ptr, W1_ptr,
		S_ptr, A_ptr, Res_ptr, Out_ptr, Sum_ptr, Dot_ptr, Off_ptr, rs, d, Wp,
		B, L, n_rows, n_tiles,
		stride_gk, stride_gb, stride_gl, stride_gc,
		stride_pk, stride_pb, stride_pl, stride_pc,
		stride_vk, stride_cwt, stride_cwc,
		stride_w2c, stride_w2h, stride_w1h, stride_w1c,
		stride_sk, stride_sl, stride_sh,
		stride_ak, stride_al, stride_ac,
		C: tl.constexpr, H: tl.constexpr, BLOCK_C: tl.constexpr,
		BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
		PRECISION: tl.constexpr, FOLD: tl.constexpr):
		# b varies fastest, so the programs that read one (k, tile) of the
		# slope and a_sum run together and share those reads in L2.
		pid = tl.program_id(0)
		b = pid % B
		t = (pid // B) % n_tiles
		k = pid // (B * n_tiles)
		kb = k.to(tl.int64) * B + b

		offs_l = t * BLOCK_M + tl.arange(0, BLOCK_M)
		# The slope and a_sum span the sequence's n_rows positions; the
		# outputs, row b's window, which starts at Off_ptr's position plus b.
		# With FOLD, G and P are row b's window of the block above, which
		# starts d later; without, G is this block's. G is zeroed at rows
		# outside the sequence before the products.
		rows = tl.load(Off_ptr) + b + offs_l
		valid = (rows >= 0) & (rows < n_rows)
		rows = tl.minimum(tl.maximum(rows, 0), n_rows - 1)
		offs_c = tl.arange(0, BLOCK_C)
		offs_h = tl.arange(0, BLOCK_H)
		mask_l = offs_l < L
		mask_c = offs_c < C
		mask_lc = mask_l[:, None] & mask_c[None, :]

		if FOLD:
			# `shift_add`'s sum, in its order: CW_ptr holds the taps as
			# stored, so the reversed tap j is row 2 - j.
			src = offs_l - d
			v = tl.load(V_ptr + k.to(tl.int64) * stride_vk)
			p_base = (P_ptr + k.to(tl.int64) * stride_pk + b.to(tl.int64)
				* stride_pb)
			cw = CW_ptr + offs_c * stride_cwc
			x = (_tap(p_base, src - d, Wp, mask_l, offs_c, mask_c, stride_pl,
				stride_pc, v) * tl.load(cw + 2 * stride_cwt, mask=mask_c,
				other=0.0)[None, :])
			x = x + (_tap(p_base, src, Wp, mask_l, offs_c, mask_c, stride_pl,
				stride_pc, v) * tl.load(cw + stride_cwt, mask=mask_c,
				other=0.0)[None, :])
			x = x + (_tap(p_base, src + d, Wp, mask_l, offs_c, mask_c,
				stride_pl, stride_pc, v) * tl.load(cw, mask=mask_c,
				other=0.0)[None, :])
			inside = (src >= 0) & (src < Wp) & mask_l
			g = tl.load(G_ptr + k.to(tl.int64) * stride_gk + b.to(tl.int64)
				* stride_gb + src[:, None] * stride_gl + offs_c[None, :]
				* stride_gc, mask=inside[:, None] & mask_c[None, :], other=0.0)
			g = g + x
			tl.store(Res_ptr + (kb * L + offs_l[:, None]) * C
				+ offs_c[None, :], g, mask=mask_lc)
			g = tl.where(valid[:, None], g, 0.0)
		else:
			g = tl.load(G_ptr + k.to(tl.int64) * stride_gk + b.to(tl.int64)
				* stride_gb + offs_l[:, None] * stride_gl + offs_c[None, :]
				* stride_gc, mask=mask_lc & valid[:, None], other=0.0)

		# Inductor passes rs as a float64 scalar.
		g = g * tl.cast(rs, tl.float32)
		if PRECISION == "tf32":
			g = _tf32_round(g)

		s_base = S_ptr + k.to(tl.int64) * stride_sk + rows[:, None] * stride_sl
		acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
		for h0 in range(0, H, BLOCK_H):
			hk = h0 + offs_h
			mask_h = hk < H
			w2 = tl.load(W2_ptr + offs_c[:, None] * stride_w2c + hk[None, :]
				* stride_w2h, mask=mask_c[:, None] & mask_h[None, :], other=0.0)
			gu = tl.dot(g, w2, input_precision=PRECISION)
			slope = tl.load(s_base + hk[None, :] * stride_sh,
				mask=mask_l[:, None] & mask_h[None, :], other=0.0)
			gu = gu * slope
			if PRECISION == "tf32":
				gu = _tf32_round(gu)

			w1 = tl.load(W1_ptr + hk[:, None] * stride_w1h + offs_c[None, :]
				* stride_w1c, mask=mask_h[:, None] & mask_c[None, :], other=0.0)
			acc = tl.dot(gu, w1, acc, input_precision=PRECISION)

		tl.store(Out_ptr + (kb * L + offs_l[:, None]) * C + offs_c[None, :],
			acc, mask=mask_lc)

		a = tl.load(A_ptr + k.to(tl.int64) * stride_ak + rows[:, None]
			* stride_al + offs_c[None, :] * stride_ac, mask=mask_lc, other=0.0)
		tl.store(Sum_ptr + kb * n_tiles + t, tl.sum(acc))
		tl.store(Dot_ptr + kb * n_tiles + t, tl.sum(acc * a))


	@triton.jit
	def _stem_backward_kernel(Gz_ptr, W_ptr, J_ptr, Ex_ptr, Lo_ptr, S_ptr,
		Rs_ptr, Dl_ptr, K, B, Wm, L, n_tiles, stride_gk, stride_gb, stride_jk,
		C: tl.constexpr, TAPS: tl.constexpr, PAD: tl.constexpr,
		BLOCK_Q: tl.constexpr, BLOCK_C: tl.constexpr, EVEN_C: tl.constexpr,
		ACC: tl.constexpr, S_LAST: tl.constexpr):
		# One program takes BLOCK_Q positions of row b's multipliers and loops
		# over the channel blocks, then the K pairs. Gz holds TAPS - 1 zero
		# rows before each row's window and zero rows after it, so that input
		# row q - t of output q is padded row q + TAPS - 1 - t, unmasked.
		pid = tl.program_id(0)
		b = pid % B
		tile = pid // B
		n_cb: tl.constexpr = (C + BLOCK_C - 1) // BLOCK_C
		offs_q = tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
		offs_c = tl.arange(0, BLOCK_C)
		at = tl.load(Lo_ptr) + b + offs_q
		inside = (at >= 0) & (at < L) & (offs_q < Wm)
		at = tl.minimum(tl.maximum(at, 0), L - 1)
		rows = (Gz_ptr + b.to(tl.int64) * stride_gb + (offs_q[:, None] + TAPS
			- 1) * C)
		total = tl.zeros((BLOCK_Q, BLOCK_C), dtype=tl.float32)
		for cb in range(n_cb):
			cc = cb * BLOCK_C + offs_c
			mask_c = (cc < C)[None, :] | EVEN_C
			# Input row q - PAD sits at the position of output q.
			ex = tl.load(Ex_ptr + at[:, None] * C + cc[None, :],
				mask=inside[:, None] & mask_c, other=0.0)
			s_acc = tl.zeros((BLOCK_Q, BLOCK_C), dtype=tl.float32)
			for k in range(K):
				# Row 4 of W is zero: the reference's base, or none.
				j = tl.load(J_ptr + k.to(tl.int64) * stride_jk + at, mask=inside,
					other=4)
				ptrs = rows + k.to(tl.int64) * stride_gk + cc[None, :]
				w_rows = W_ptr + j[:, None] * (TAPS * C) + cc[None, :]
				acc = tl.zeros((BLOCK_Q, BLOCK_C), dtype=tl.float32)
				for t in tl.static_range(TAPS):
					if EVEN_C:
						g = tl.load(ptrs - t * C)
						w = tl.load(w_rows + t * C)
					else:
						g = tl.load(ptrs - t * C, mask=mask_c, other=0.0)
						w = tl.load(w_rows + t * C, mask=mask_c, other=0.0)
					acc += g * w
					if t == PAD:
						s_acc += g
						gx = g * ex

				total += acc
				tl.store(Dl_ptr + ((k.to(tl.int64) * B + b) * n_tiles + tile)
					* n_cb + cb, tl.sum(gx - acc))

			# With ACC, S and Rs hold the sums over the pairs before these.
			if S_LAST:
				s_ptrs = S_ptr + (b.to(tl.int64) * Wm + offs_q[:, None]) * C + cc[
					None, :]
			else:
				s_ptrs = S_ptr + (b.to(tl.int64) * C + cc[None, :]) * Wm + offs_q[
					:, None]
			s_mask = (offs_q < Wm)[:, None] & mask_c
			if ACC:
				s_acc += tl.load(s_ptrs, mask=s_mask, other=0.0)

			tl.store(s_ptrs, s_acc, mask=s_mask)

		rs_sum = tl.sum(total, axis=1)
		if ACC:
			rs_sum += tl.load(Rs_ptr + b.to(tl.int64) * Wm + offs_q,
				mask=offs_q < Wm, other=0.0)

		tl.store(Rs_ptr + b.to(tl.int64) * Wm + offs_q, rs_sum,
			mask=offs_q < Wm)


	@triton.jit
	def _position_matrix_kernel(W1T_ptr, W2_ptr, S_ptr, Off_ptr, M_ptr, P,
		n_rows, stride_w2c, stride_w2h, stride_sk, stride_sl, stride_sh,
		C: tl.constexpr, H: tl.constexpr, BLOCK_O: tl.constexpr,
		BLOCK_I: tl.constexpr, BLOCK_H: tl.constexpr, PRECISION: tl.constexpr):
		# M[k, p] (C out, C in) = W1^T diag(slope_k[lo + p]) W2^T, the slope
		# times W2's columns in float32, then rounded as a product operand.
		pid = tl.program_id(0)
		n_o: tl.constexpr = (C + BLOCK_O - 1) // BLOCK_O
		n_i: tl.constexpr = (C + BLOCK_I - 1) // BLOCK_I
		ti = pid % n_i
		to = (pid // n_i) % n_o
		kp = pid // (n_i * n_o)
		k = kp // P
		row = tl.load(Off_ptr) + kp % P
		row = tl.minimum(tl.maximum(row, 0), n_rows - 1)
		offs_o = to * BLOCK_O + tl.arange(0, BLOCK_O)
		offs_i = ti * BLOCK_I + tl.arange(0, BLOCK_I)
		offs_h = tl.arange(0, BLOCK_H)
		mask_o, mask_i = offs_o < C, offs_i < C
		s_base = S_ptr + k.to(tl.int64) * stride_sk + row * stride_sl
		acc = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
		for h0 in range(0, H, BLOCK_H):
			hk = h0 + offs_h
			mask_h = hk < H
			w1 = tl.load(W1T_ptr + offs_o[:, None] * H + hk[None, :],
				mask=mask_o[:, None] & mask_h[None, :], other=0.0)
			w2 = tl.load(W2_ptr + offs_i[None, :] * stride_w2c + hk[:, None]
				* stride_w2h, mask=mask_h[:, None] & mask_i[None, :], other=0.0)
			s = tl.load(s_base + hk * stride_sh, mask=mask_h, other=0.0)
			w2 = w2 * s[:, None]
			if PRECISION == "tf32":
				w2 = _tf32_round(w2)

			acc = tl.dot(w1, w2, acc, input_precision=PRECISION)

		# M is only ever a product operand, so it is stored rounded.
		if PRECISION == "tf32":
			acc = _tf32_round(acc)

		tl.store(M_ptr + (kp.to(tl.int64) * C + offs_o[:, None]) * C
			+ offs_i[None, :], acc, mask=mask_o[:, None] & mask_i[None, :])


	@triton.jit
	def _position_block_kernel(G_ptr, M_ptr, A_ptr, Out_ptr, Sum_ptr, Dot_ptr,
		Off_ptr, Tab_ptr, rs, n, W, P, n_rows, n_programs,
		stride_gk, stride_gb, stride_gl, stride_gc,
		stride_ak, stride_al, stride_ac,
		C: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr,
		BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
		# One program takes one position p of the level's P = n + W - 1, up
		# to BLOCK_B of the rows whose window covers it, rows b with position
		# p at offset p - b of row b's window, and BLOCK_N output channels:
		# one matrix M[k, p] serves all of them. The programs are numbered
		# position by position (`position_table`), so those that read one
		# M[k, p] run together.
		pid = tl.program_id(0)
		n_n: tl.constexpr = (C + BLOCK_N - 1) // BLOCK_N
		tn = pid % n_n
		q = pid // n_n
		k = q // n_programs
		q = q % n_programs
		p = tl.load(Tab_ptr + q)
		b_end = tl.minimum(p + 1, n)
		offs_b = tl.load(Tab_ptr + n_programs + q) + tl.arange(0, BLOCK_B)
		mask_b = offs_b < b_end
		offs_l = p - offs_b
		at = tl.load(Off_ptr) + p
		inside = (at >= 0) & (at < n_rows)
		at = tl.minimum(tl.maximum(at, 0), n_rows - 1)

		g_rows = (G_ptr + k.to(tl.int64) * stride_gk + offs_b.to(tl.int64)
			* stride_gb + offs_l.to(tl.int64) * stride_gl)
		offs_n = tn * BLOCK_N + tl.arange(0, BLOCK_N)
		mask_n = offs_n < C
		m_base = M_ptr + ((k.to(tl.int64) * P + p) * C + offs_n[None, :]) * C
		offs_k = tl.arange(0, BLOCK_K)
		acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
		for c0 in range(0, C, BLOCK_K):
			ck = c0 + offs_k
			mask_k = ck < C
			g = tl.load(g_rows[:, None] + ck[None, :] * stride_gc,
				mask=(mask_b & inside)[:, None] & mask_k[None, :], other=0.0)
			# Inductor passes rs as a float64 scalar.
			g = g * tl.cast(rs, tl.float32)
			if PRECISION == "tf32":
				g = _tf32_round(g)

			w = tl.load(m_base + ck[:, None], mask=mask_k[:, None]
				& mask_n[None, :], other=0.0)
			acc = tl.dot(g, w, acc, input_precision=PRECISION)

		out = ((k.to(tl.int64) * n + offs_b) * W + offs_l)
		tl.store(Out_ptr + out[:, None] * C + offs_n[None, :], acc,
			mask=mask_b[:, None] & mask_n[None, :])
		a = tl.load(A_ptr + k.to(tl.int64) * stride_ak + at * stride_al
			+ offs_n * stride_ac, mask=mask_n, other=0.0)
		tl.store(Sum_ptr + out * n_n + tn, tl.sum(acc, axis=1), mask=mask_b)
		tl.store(Dot_ptr + out * n_n + tn, tl.sum(acc * a[None, :], axis=1),
			mask=mask_b)


def tf32_round(x):
	"""x (float32) rounded to TF32 as `_tf32_round` rounds it."""

	bits = x.view(torch.int32)
	bits = (bits + 0xFFF + ((bits >> 13) & 1)) & -0x2000
	return bits.view(torch.float32)


_DEVICES = {}
_SETTINGS = {}
_POINTER_TYPES = {"Off_ptr": "*i64", "Lo_ptr": "*i64", "Tab_ptr": "*i32",
	"J_ptr": "*i32"}
_UNIT_ARGUMENTS = {
	"_block_backward_kernel": ("stride_gc", "stride_pc", "stride_vk",
		"stride_cwc", "stride_w2c", "stride_w1h", "stride_sh", "stride_ac"),
	"_position_matrix_kernel": ("stride_w2h", "stride_sh"),
	"_position_block_kernel": ("stride_gc", "stride_ac"),
	"_stem_backward_kernel": (),
}
MAX_TENSOR_MEMORY = 512


def device_properties(device):
	"""(max_shared_mem, capability) of a CUDA device: the shared memory per
	block Triton allows, in bytes, and the compute capability (major,
	minor), read once per device. The kernels' settings and their precision
	are chosen from these alone, so replacing this function gives the engine
	another GPU type's settings."""

	device = torch.device(device)
	index = (torch.cuda.current_device() if device.index is None
		else device.index)
	if index not in _DEVICES:
		shared = triton.runtime.driver.active.utils.get_device_properties(
			index)["max_shared_mem"]
		_DEVICES[index] = (int(shared),
			tuple(torch.cuda.get_device_capability(index)))

	return _DEVICES[index]


def use_tf32(device):
	"""Whether the kernels' products run in TF32 on device: where
	``torch.backends.cuda.matmul.allow_tf32`` is on and the GPU has TF32
	(compute capability 8.0 or more). Below 8.0 cuBLAS runs float32 whatever
	the flag says."""

	device = torch.device(device)
	return bool(torch.backends.cuda.matmul.allow_tf32 and HAS_TRITON
		and device.type == "cuda" and device_properties(device)[1] >= (8, 0))


def _resources(kernel, constexprs, capability, num_warps, num_stages,
	ones=()):
	"""The shared memory per block, in bytes, and the tensor memory, in
	columns or None, of `kernel` compiled for a GPU of `capability` with
	these constexprs, warps and stages, without launching it, specialized
	as the engine's launches are: each argument named ``*_ptr`` a 16-byte
	aligned pointer, to float32 unless `_POINTER_TYPES` names another type;
	those in `ones` 1; the other strides int32 multiples of 16; rs a float32
	and every other argument an int32."""

	from triton.backends.compiler import GPUTarget
	from triton.compiler import ASTSource

	constexprs = {**constexprs, **{name: 1 for name in ones}}
	signature, attrs = {}, {}
	for i, name in enumerate(kernel.arg_names):
		if name in constexprs:
			signature[name] = "constexpr"
		elif name.endswith("_ptr"):
			signature[name] = _POINTER_TYPES.get(name, "*fp32")
			attrs[(i,)] = [["tt.divisibility", 16]]
		else:
			signature[name] = "fp32" if name == "rs" else "i32"
			if name.startswith("stride_"):
				attrs[(i,)] = [["tt.divisibility", 16]]

	target = GPUTarget("cuda", capability[0] * 10 + capability[1], 32)
	compiled = triton.compile(ASTSource(kernel, signature, constexprs, attrs),
		target=target, options={"num_warps": num_warps,
		"num_stages": num_stages})
	return compiled.metadata.shared, getattr(compiled.metadata, "tmem_size",
		None)


def _lowered(config, stages, tiles):
	"""config, then config with each smaller number of stages, then with the
	tiles halved one after another, each tile down to its smallest, each
	again with every number of stages: config's index of num_stages (None if
	it has none) and tiles (index, smallest) in the order to halve them."""

	settings = []

	def add(c):
		counts = [c] if stages is None else [c[:stages] + (s,) + c[stages
			+ 1:] for s in range(config[stages], 0, -1)]
		for s in counts:
			if s not in settings:
				settings.append(s)

	add(config)
	for i, least in tiles:
		while config[i] // 2 >= least:
			config = config[:i] + (config[i] // 2,) + config[i + 1:]
			add(config)

	return settings


def _fitting(name, kernel, settings, launch, device, ones=None):
	"""The first of `settings` whose kernel, compiled for device's compute
	capability with the constexprs, warps and stages launch(setting) gives,
	fits its shared memory per block, or None if none does or Triton
	compiles none for it. Made once per setting and GPU type."""

	limit, capability = device_properties(device)
	key = (name, tuple(settings), limit, capability, tuple(sorted(launch(
		settings[0])[0].items())))
	if key not in _SETTINGS:
		chosen, reasons = None, []
		for setting in settings:
			constexprs, warps, stages = launch(setting)
			try:
				shared, tmem = _resources(kernel, constexprs, capability, warps,
					stages, _UNIT_ARGUMENTS[name] if ones is None else ones)
			except Exception as exc:
				reasons.append("{}: {}".format(type(exc).__name__,
					str(exc).strip().split("\n")[0]))
				continue

			if shared <= limit and (tmem is None or tmem <= MAX_TENSOR_MEMORY):
				chosen = setting
				break

			reasons.append("{} needs {} B of shared memory".format(setting,
				shared))

		if chosen is None:
			logger.warning("%s: no setting runs on this GPU (compute capability "
				"%d.%d, %d B of shared memory per block; %s); its part runs in "
				"torch", name, *capability, limit, "; ".join(reasons[:2]))
		elif chosen != settings[0]:
			logger.info("%s: %s on this GPU (compute capability %d.%d), in place "
				"of %s", name, chosen, *capability, settings[0])

		_SETTINGS[key] = chosen

	return _SETTINGS[key]


def block_settings(C, H, device, tf32):
	"""`block_backward`'s kernel settings for C filters and H hidden ones on
	device, in TF32 or not: `_block_backward_config`'s where they fit, with
	fewer stages and then smaller tiles where they do not, and None where
	none fits or Triton cannot compile one."""

	def launch(s):
		return dict(C=C, H=H, BLOCK_C=max(16, triton.next_power_of_2(C)),
			BLOCK_M=s[0], BLOCK_H=s[1], PRECISION="tf32" if tf32 else "ieee",
			FOLD=C <= FOLD_MAX_FILTERS), s[2], s[3]

	return _fitting("_block_backward_kernel", _block_backward_kernel,
		_lowered(_block_backward_config(C), 3, ((0, 16), (1, 16))), launch,
		device)


def position_settings(C, H, device, tf32):
	"""`position_backward`'s kernel settings for C filters and H hidden ones
	on device, in TF32 or not, as `block_settings` chooses them from
	`_position_config`'s, or None where either kernel has none."""

	precision = "tf32" if tf32 else "ieee"
	matrix, products = _position_config(C)
	matrix = _fitting("_position_matrix_kernel", _position_matrix_kernel,
		_lowered(matrix, 4, ((0, 16), (1, 16), (2, 16))), lambda s: (dict(C=C,
		H=H, BLOCK_O=s[0], BLOCK_I=s[1], BLOCK_H=s[2], PRECISION=precision),
		s[3], s[4]), device)
	products = _fitting("_position_block_kernel", _position_block_kernel,
		_lowered(products, 4, ((1, 16), (0, 16), (2, 16))), lambda s: (dict(
		C=C, BLOCK_B=s[0], BLOCK_N=s[1], BLOCK_K=s[2], PRECISION=precision),
		s[3], s[4]), device)
	return None if matrix is None or products is None else (matrix, products)


def stem_settings(C, taps, device, s_last=False):
	"""`stem_backward`'s kernel settings for C filters and iconv's taps on
	device, with S channels-last when `s_last` is true, as `block_settings`
	chooses them from `_stem_backward_config`'s, or None."""

	def launch(s):
		return dict(C=C, TAPS=taps, PAD=ICONV_PADDING, BLOCK_Q=s[0],
			BLOCK_C=s[1], EVEN_C=C % s[1] == 0, ACC=s_last, S_LAST=s_last), s[2], 1

	# With S channels-last the engine runs one pair per call: K is 1.
	return _fitting("_stem_backward_kernel", _stem_backward_kernel, _lowered(
		_stem_backward_config(C, s_last), None, ((1, 16), (0, 2))), launch,
		device, ("K",) if s_last else ())


def _block_backward_config(C):
	"""BLOCK_M, BLOCK_H, num_warps and num_stages of `block_backward`'s
	kernel for C filters, fixed so that every process sums in one order, on
	every GPU where they fit (`block_settings`)."""

	if C <= 32:
		return 64, 64, 4, 2
	if C <= 64:
		return 32, 64, 4, 2
	return 64, 32, 4, 2


def shift_add(G, g, v_avg, cw, d, W, before=0):
	"""The gradient at a block's input, over W positions of each row's
	window, which starts d + before positions before the block's own: G, the
	gradient at the block's output, plus dw3 of v_avg * g, g its product at
	the normalization, with the taps cw (3, C, as stored) reversed. g is
	zero outside [0, L), where G holds values that every reader of G masks
	out."""

	h, cw = v_avg * g, cw.flip(0)
	return shift(G, -d - before, W) + (shift(h, -2 * d - before, W) * cw[0]
		+ shift(h, -d - before, W) * cw[1] + shift(h, -before, W) * cw[2])


def block_backward(above, W2, slope, W1, a_sum, lo, W, rs, tf32,
	config=None):
	"""One block's backward over row b's W consecutive positions from lo +
	b, lo a 0-d int64 device tensor, in one Triton kernel: G, the gradient
	at the block's output, `shift_add` of the block above's terms (in the
	kernel for at most `FOLD_MAX_FILTERS` filters, before it otherwise), then
	``torch.matmul(torch.matmul(G * rs, W2) * slope[:, 0][:, at], W1)``, at
	= `positions`(lo, B, W), and the normalization rule's two reductions of
	it over each row's (W, C) window.

	above = (G, g, v_avg, cw, d) is the block above's: the gradient at its
	output, (K, B, Wp, C) or (B, Wp, C) for every pair, its product g (K, B,
	Wp, C), its v_avg (K, 1, 1, 1), its taps cw (3, C) and its dilation d;
	its window starts d positions after this one. The kernel reads lo, so
	that a new window neither recompiles nor waits on the host. G is zeroed
	at the positions outside [0, L) before the products, where the product
	is zero. slope (K, 1, L, H) and a_sum (K, 1, L, C) are each pair's terms
	over the whole sequence, read at the positions `at`. Each program takes
	up to BLOCK_M positions of one (k, b) row and keeps its hidden gradient
	on chip, BLOCK_H columns at a time. Returns G and the product, each (K,
	B, W, C), and the product's sum and its dot product with a_sum[:, 0][:,
	at] per row, (K, B, 1, 1), each the sum of per-program partial sums in a
	fixed order. The products run in TF32 when `tf32` is true, as cuBLAS
	runs them under ``torch.backends.cuda.matmul.allow_tf32``, with the
	operands rounded to nearest as cuBLAS rounds them, and in IEEE float32
	otherwise.
	"""

	Ga, g, v_avg, cw, d = above
	K, B, Wp, C = g.shape
	H = W2.shape[1]
	fold = C <= FOLD_MAX_FILTERS
	if not fold:
		Ga = shift_add(Ga, g, v_avg, cw, d, W)

	if tf32:
		W2, W1 = tf32_round(W2), tf32_round(W1)

	# TF32 wgmma takes both operands K-major only; so laid out, the weights
	# go straight from memory to shared memory.
	W2, W1 = W2.t().contiguous().t(), W1.t().contiguous().t()
	BLOCK_M, BLOCK_H, num_warps, num_stages = (config
		or _block_backward_config(C))
	n_tiles = triton.cdiv(W, BLOCK_M)

	stride_gk = Ga.stride(0) if Ga.dim() == 4 else 0
	res = (torch.empty((K, B, W, C), dtype=g.dtype, device=g.device) if fold
		else Ga)
	out = torch.empty((K, B, W, C), dtype=g.dtype, device=g.device)
	sums = torch.empty((K, B, n_tiles), dtype=g.dtype, device=g.device)
	dots = torch.empty((K, B, n_tiles), dtype=g.dtype, device=g.device)
	_block_backward_kernel[(K * n_tiles * B,)](Ga, g, v_avg, cw, W2, W1,
		slope, a_sum, res, out, sums, dots, lo, rs, d, Wp, B, W,
		slope.shape[2], n_tiles, stride_gk, *Ga.stride()[-3:], *g.stride(),
		v_avg.stride(0), cw.stride(0), cw.stride(1),
		W2.stride(0), W2.stride(1), W1.stride(0), W1.stride(1),
		slope.stride(0), slope.stride(2), slope.stride(3),
		a_sum.stride(0), a_sum.stride(2), a_sum.stride(3),
		C=C, H=H, BLOCK_C=max(16, triton.next_power_of_2(C)),
		BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H, PRECISION="tf32" if tf32 else "ieee",
		FOLD=fold, num_warps=num_warps, num_stages=num_stages)

	return (res, out, sums.sum(dim=-1)[..., None, None],
		dots.sum(dim=-1)[..., None, None])


def _position_config(C):
	"""The tiles of `position_backward`'s two kernels for C filters: the
	matrices' BLOCK_O, BLOCK_I, BLOCK_H, num_warps and num_stages, then the
	products' BLOCK_B, BLOCK_N, BLOCK_K, num_warps and num_stages, fixed so
	that every process sums in one order, on every GPU where they fit
	(`position_settings`)."""

	if C <= 32:
		return (32, 32, 64, 1, 2), (64, 32, 32, 4, 2)
	if C <= 64:
		# At 3 stages the products tile left one position wrong in some calls
		# at C = 64 (Triton 3.7.1).
		return (64, 64, 32, 4, 1), (128, 64, 32, 4, 1)
	if C <= 128:
		return (128, 128, 32, 4, 3), (128, 128, 32, 4, 3)
	return (128, 128, 32, 4, 3), (128, 256, 32, 8, 3)


def position_table(n, W, BLOCK_B, device):
	"""The programs of `_position_block_kernel` for n rows of W positions,
	(2, n_programs) int32 on the device: at each of the n + W - 1 positions
	in turn, the rows that cover it in tiles of BLOCK_B, each program's
	position and its first row."""

	m, P = min(n, W), n + W - 1
	p = torch.arange(P)
	count = (torch.minimum(p + 1, P - p).clamp(max=m) + BLOCK_B - 1) // BLOCK_B
	at = torch.repeat_interleave(p, count)
	start = torch.cumsum(count, 0) - count
	first = (at - W + 1).clamp(min=0) + (torch.arange(len(at)) - start[at]) \
		* BLOCK_B
	return torch.stack([at, first]).to(device=device, dtype=torch.int32)


def position_backward(G, W2, slope, W1, a_sum, lo, rs, tf32, table,
	config=None):
	"""One block's backward over row b's W consecutive positions from lo +
	b, lo a 0-d int64 device tensor, by position: what `block_backward`
	computes from G, ``torch.matmul(torch.matmul(G * rs, W2) * slope[:,
	0][:, at], W1)`` and the normalization rule's two reductions of it, with
	the two products and the slope between them folded into one matrix per
	pair and position, M = W2 diag(slope) W1.

	G (K, B, W, C) is the gradient at the block's output, at the positions
	`positions`(lo, B, W); it is zeroed at the positions outside [0, L).
	slope (K, 1, L, H) and a_sum (K, 1, L, C) are each pair's terms over
	the whole sequence, and table is `position_table`(B, W, BLOCK_B) for
	the configuration's BLOCK_B. One kernel forms M at the B + W - 1 positions the
	windows cover, (K, B + W - 1, C, C); a second takes each position and
	the rows whose window covers it, which share its M. The slope multiplies
	W2's columns in float32; both products run in TF32 when `tf32` is true,
	with their operands rounded to nearest as cuBLAS rounds them, and in
	IEEE float32 otherwise. Per row this costs one C x C product where
	`block_backward` costs two C x H ones, H = 2C, and the matrices cost 4C^3
	per position, so it pays when B is large next to C. Returns the product
	(K, B, W, C) and its sum and its dot product with a_sum[:, 0][:, at] per
	row, (K, B, 1, 1), each the sum of per-program partial sums in a fixed
	order.
	"""

	K, B, W, C = G.shape
	H = W2.shape[1]
	P = B + W - 1
	(BLOCK_O, BLOCK_I, BLOCK_H, warps_m, stages_m), (BLOCK_B, BLOCK_N,
		BLOCK_K, warps_p, stages_p) = config or _position_config(C)
	precision = "tf32" if tf32 else "ieee"

	W1T = (tf32_round(W1) if tf32 else W1).t().contiguous()
	M = torch.empty((K, P, C, C), dtype=G.dtype, device=G.device)
	_position_matrix_kernel[(K * P * triton.cdiv(C, BLOCK_O) * triton.cdiv(C,
		BLOCK_I),)](W1T, W2, slope, lo, M, P, slope.shape[2], W2.stride(0),
		W2.stride(1), slope.stride(0), slope.stride(2), slope.stride(3), C=C,
		H=H, BLOCK_O=BLOCK_O, BLOCK_I=BLOCK_I, BLOCK_H=BLOCK_H,
		PRECISION=precision, num_warps=warps_m, num_stages=stages_m)

	n_n = triton.cdiv(C, BLOCK_N)
	n_programs = table.shape[1]
	out = torch.empty((K, B, W, C), dtype=G.dtype, device=G.device)
	sums = torch.empty((K, B, W, n_n), dtype=G.dtype, device=G.device)
	dots = torch.empty((K, B, W, n_n), dtype=G.dtype, device=G.device)
	_position_block_kernel[(K * n_programs * n_n,)](G, M, a_sum, out, sums,
		dots, lo, table, rs, B, W, P, slope.shape[2], n_programs, *G.stride(),
		a_sum.stride(0), a_sum.stride(2), a_sum.stride(3), C=C,
		BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
		PRECISION=precision, num_warps=warps_p, num_stages=stages_p)

	return (out, sums.sum(dim=(-2, -1))[..., None, None],
		dots.sum(dim=(-2, -1))[..., None, None])


def _stem_backward_config(C, s_last=False):
	"""BLOCK_Q, BLOCK_C and num_warps of `stem_backward`'s kernel for C
	filters, with S channels-last, one pair per call, when `s_last` is true,
	on every GPU where they fit (`stem_settings`). Other tiles of these sizes
	spilled registers, some taking 20 to 60 times as long."""

	if C <= 32:
		return (32 if s_last else 16), max(16, triton.next_power_of_2(C)), (
			4 if s_last else 2)
	if C <= 64:
		return (16, 64, 4) if s_last else (32, 64, 8)
	if C <= 128:
		return 8, 128, 8
	return 8, 256, 8


def stem_padding(W, taps, config):
	"""The zero rows `stem_backward` needs before and after each row's W
	positions of Gz, for iconv's taps and the kernel's settings config."""

	BLOCK_Q = config[0]
	n_tiles = triton.cdiv(W + 2 * ICONV_PADDING, BLOCK_Q)
	return taps - 1, n_tiles * BLOCK_Q - W


def stem_partials(Wm, C, config):
	"""The partial sums `stem_backward` gives per pair and row for each
	delta, for Wm positions, C filters and the kernel's settings config."""

	BLOCK_Q, BLOCK_C, _ = config
	return triton.cdiv(Wm, BLOCK_Q) * triton.cdiv(C, BLOCK_C)


def stem_backward(Gz, W5, J, Ex, lo, Wm, config=None, S=None, Rs=None,
	Dl=None, s_last=False):
	"""What `epilogue` keeps of iconv's input gradient for every pair, in
	IEEE float32, without the multipliers of each pair.

	Each pair's multipliers m_k (4, Wm), Wm = W + 2p and p =
	`ICONV_PADDING`, are those of ``conv1d_input`` with padding p of its
	local gradient at iconv's output, row b over the positions [lo + b, lo
	+ b + Wm), lo a 0-d int64 device tensor. The hypothetical attributions
	averaged over the pairs are linear in m_k except for the term
	m_k[r_k(i), i], with r_k(i) the reference's base at position i, so the
	sum over the pairs of the local gradients and that term's sum suffice;
	sum((x - r_k) * m_k) is the gradient's dot product with Ex less the
	term's sum over the positions.

	Gz (K, B, Wp, C) is the local gradient, channels-last, zero outside [0,
	L), with the rows of `stem_padding` around each row's W positions. W5
	(5, taps, C) is iconv's weight permuted with a fifth, zero, row, J (K,
	L) int32 each reference's base at each position, 4 where it has none,
	and Ex (L, C) iconv's convolution of the sequence's one-hot, channels-
	last, without bias. Returns S (B, C, Wm), the sum over the pairs of the
	gradient padded by p, whose ``conv1d_input`` is the sum of the m_k; Rs
	(B, Wm), the sum over the pairs of m_k[r_k(i), i], zero outside [0, L);
	and each pair's sum((x - r_k) * m_k) over the positions in [0, L) in
	partial sums, (K, B, n), to be summed over the last axis. Given S and
	Rs, the sums over these pairs are added to theirs, so that groups of
	pairs can run one after another; given Dl, the partial sums are written
	into it. With `s_last`, S is (B, Wm, C) instead, whose reads and writes
	coalesce when it is added to once per pair.
	"""

	K, B, Wp, C = Gz.shape
	taps = W5.shape[1]
	L = Ex.shape[0]
	BLOCK_Q, BLOCK_C, num_warps = config or _stem_backward_config(C, s_last)
	n_tiles = triton.cdiv(Wm, BLOCK_Q)
	assert Wp == taps - 1 + n_tiles * BLOCK_Q and Gz.stride(-1) == 1 \
		and Gz.stride(-2) == C

	accumulate = S is not None
	if not accumulate:
		S = torch.empty((B, Wm, C) if s_last else (B, C, Wm), dtype=Gz.dtype,
			device=Gz.device)
		Rs = torch.empty((B, Wm), dtype=Gz.dtype, device=Gz.device)

	if Dl is None:
		Dl = torch.empty((K, B, n_tiles * triton.cdiv(C, BLOCK_C)),
			dtype=Gz.dtype, device=Gz.device)

	assert Dl.shape == (K, B, n_tiles * triton.cdiv(C, BLOCK_C)) \
		and Dl.is_contiguous()
	_stem_backward_kernel[(B * n_tiles,)](Gz, W5, J, Ex, lo, S, Rs, Dl, K, B,
		Wm, L, n_tiles, Gz.stride(0), Gz.stride(1), J.stride(0), C=C,
		TAPS=taps, PAD=ICONV_PADDING, BLOCK_Q=BLOCK_Q, BLOCK_C=BLOCK_C,
		EVEN_C=C % BLOCK_C == 0, ACC=accumulate, S_LAST=s_last,
		num_warps=num_warps, num_stages=1)
	return S, Rs, Dl


def stem_fused(W0):
	"""Whether `Engine` can run iconv's input gradient through
	`stem_backward` on references whose every position holds at most one
	base: on CUDA with Triton, for float32 models. It does where the GPU runs
	the kernel (`stem_settings`)."""

	return HAS_TRITON and W0.is_cuda and W0.dtype == torch.float32


def one_hot(X):
	"""Whether every position of the int8 CPU tensor X (..., 4, L) holds at
	most one base, each a one."""

	return bool(((X == 0) | (X == 1)).all()) and bool((X.sum(dim=-2)
		<= 1).all())


def position_min_outputs(C):
	"""The fewest output positions per chunk for which `Engine` runs the
	blocks of a model of C filters by position: `POSITION_MIN_OUTPUTS` per
	filter, and at least `POSITION_MIN_CHUNK`'s."""

	n = POSITION_MIN_OUTPUTS * C
	for most, chunk in POSITION_MIN_CHUNK:
		if C <= most:
			return max(n, chunk)

	return n


def by_position(W0):
	"""Whether `Engine` can run the blocks below the top one in
	`position_backward`, one pair at a time: where `stem_fused` holds, for
	models of `POSITION_FILTERS` filters. It does where the GPU runs both
	kernels (`position_settings`, `stem_settings`)."""

	lo, hi = POSITION_FILTERS
	return stem_fused(W0) and lo <= W0.shape[0] <= hi


def fused(taps):
	"""Whether `backward_pass` can run the blocks below the top one in
	`block_backward`: on CUDA with Triton, for float32 models of at most
	`FUSED_MAX_FILTERS` filters. It does where the GPU runs the kernel
	(`block_settings`)."""

	return (HAS_TRITON and taps.is_cuda and taps.dtype == torch.float32
		and taps.shape[-1] <= FUSED_MAX_FILTERS)


def backward_pass(taps, lo, B, terms, stem, blocks, rs, tf32=None,
	stem_last=None, rows=(0, 0), tables=None, config=None):
	"""The local part of the backward over K pairs and B output positions,
	from the final residual stream to iconv's output. taps (width, C) is
	every output position's gradient at the final residual stream, which
	every pair shares: the profile convolution's taps for the attributed
	channel, over the positions [lo + b, lo + b + width) for output b of the
	chunk, lo a 0-d int64 tensor; the positions outside [0, L) get zeros.

	The operations are those of
	`cherimoya._deep_lift_shap_engine.backward_pass`, with each rule's terms
	read from `rule_terms`, split by linearity. Each normalization rule's
	gradient is v_avg * g, local, plus the dense part (-v_avg * mean(g)) *
	1 + dot(g, a_sum) * variance_term. The local part of each output
	position is carried over its own `windows`, so that one offset moves
	every output's window to the next; the top block's first product runs
	once, on the taps, for every pair and output. The two coefficients of
	each dense part are returned and `basis_fields` and `basis_multipliers`
	give what each one adds. Where `fused` holds and the GPU runs its kernel,
	the blocks below the top one run in `block_backward`, which also builds
	the gradient at each one's output from the block above's terms, in TF32
	when `tf32` is true (None: `use_tf32`). Given tables, one
	`position_table` for each block below the top one, top first, those
	blocks run in `position_backward` instead. config is the kernels'
	settings, (`block_settings`, `position_settings`), a None for a kernel
	the GPU does not run; None reads them for taps' device, which only an
	eager caller can do.

	Returns Gz (K * B, C, Ws + 2 * ICONV_PADDING), pair-major, the local
	gradient at iconv's output, row b over the positions [a + b, a + b + Ws
	+ 2 * ICONV_PADDING); coef (K, B, 2 * len(blocks)), each block's two
	coefficients, top first; and a, 0-d. With the stem's multiplier
	channels-last, stem_last (K, L, C), Gz is (K, B, rows[0] + Ws + rows[1],
	C) instead, row b's Ws positions from [a + b + ICONV_PADDING] after
	rows[0] zero rows and before rows[1], for `stem_backward`. On CUDA this
	function runs compiled.
	"""

	K = stem.shape[0]
	L = stem.shape[-1]
	width, C = taps.shape
	spans = windows(lo, width, [b.d for b in reversed(blocks)])
	if tf32 is None:
		tf32 = use_tf32(taps.device)
	if config is None:
		config = ((block_settings(C, blocks[0].W2.shape[1], taps.device, tf32)
			if fused(taps) and blocks else None), (position_settings(C,
			blocks[0].W2.shape[1], taps.device, tf32) if tables is not None
			else None))

	block_config, position_config = config
	use_fused = tables is not None or (fused(taps) and block_config is not None)
	# With stem_last, the gradient at iconv's output is computed over its
	# window with rows[0] positions before it and rows[1] after it, where it
	# is zero, so that the stem's multiplier fuses into the last shift.
	before = rows[0] if stem_last is not None else 0
	spans[-1] = (spans[-1][0] - before, spans[-1][1] + sum(rows)
		if stem_last is not None else spans[-1][1])

	def inside(a, W):
		at = positions(a, B, W)
		return ((at >= 0) & (at < L))[..., None], at.clamp(0, L - 1)

	# The top block's G has no pair axis.
	seed, at = inside(lo, width)
	G, coef, above = torch.where(seed, taps, 0), [], None
	for i, ((a, Wa), (b, Wb), t, block) in enumerate(zip(spans, spans[1:],
		reversed(terms), reversed(blocks))):
		if above is not None and tables is not None:
			G = shift_add(*above, Wa)
			g, total, dot = position_backward(G, block.W2, t.slope, block.W1,
				t.a_sum, a, rs, tf32, tables[i - 1], position_config)
		elif above is not None:
			G, g, total, dot = block_backward(above, block.W2, t.slope,
				block.W1, t.a_sum, a, Wa, rs, tf32, block_config)
		else:
			if G.dim() == 3:
				Gu = torch.where(seed, torch.matmul(taps * rs, block.W2), 0)
			else:
				valid, at = inside(a, Wa)
				Gu = torch.matmul(torch.where(valid, G, 0) * rs, block.W2)

			g = torch.matmul(Gu * t.slope[:, 0][:, at], block.W1)
			total = g.sum(dim=(-2, -1), keepdim=True)
			dot = (g * t.a_sum[:, 0][:, at]).sum(dim=(-2, -1), keepdim=True)

		coef.append(-t.v_avg * (total / (L * C)))
		coef.append(dot)

		if use_fused:
			above = (G, g, t.v_avg, block.cw, block.d)
		else:
			G = shift_add(G, g, t.v_avg, block.cw, block.d, Wb, before
				if i == len(blocks) - 1 else 0)

	a, Wa = spans[-1]
	if above is not None:
		G = shift_add(*above, Wa, before)
	elif not blocks and before:
		G = shift(G, -before, Wa)

	valid, at = inside(a, Wa)
	coef = (torch.cat([c.view(K, -1, 1) for c in coef], dim=-1) if coef
		else taps.new_zeros(K, B, 0))
	if stem_last is not None:
		Gz = torch.where(valid, G, 0) * stem_last[:, at]
		return Gz, coef, a + before - ICONV_PADDING

	Gz = F.pad(torch.where(valid, G, 0).transpose(-1, -2) * stem[:, 0][:, :,
		at].transpose(1, 2), (ICONV_PADDING, ICONV_PADDING))
	return (Gz.reshape(-1, *Gz.shape[-2:]).contiguous(), coef,
		a - ICONV_PADDING)


def basis_fields(terms, stem, blocks, rs):
	"""The gradients at iconv's output of the dense parts `backward_pass`
	leaves out, (K * 2 * len(blocks), C, L), pair-major: for each block, top
	first, its gradient at the block's input for the fields 1 and
	variance_term, each run through the blocks below with their whole rules
	and the stem. On CUDA this function runs compiled."""

	K, _, C, L = stem.shape
	ones, fields = stem.new_ones(L, C), None
	for t, block in zip(reversed(terms), reversed(blocks)):
		cw = block.cw.flip(0)
		if fields is not None:
			Gu = torch.matmul(fields * rs, block.W2) * t.slope
			Gy = _layer_normalization_grad(torch.matmul(Gu, block.W1),
				t.v_avg, t.a_sum, t.variance_term, 2)
			del Gu
			fields = fields + dw3(Gy, cw, block.d)
			del Gy

		new = dw3(torch.stack([ones.expand(K, L, C), t.variance_term[:, 0]],
			dim=1), cw, block.d)
		fields = new if fields is None else torch.cat([fields, new], dim=1)

	if fields is None:
		return stem.new_zeros(0, C, L)

	return (fields.transpose(-1, -2) * stem).flatten(0, 1)


def basis_multipliers(Gz, W0, K):
	"""The multipliers of `basis_fields`' Gz (K * n, C, L), iconv's input
	gradient, (K, n, 4, L). Every output position's multipliers are its
	local ones plus these, times its coefficients."""

	n, _, L = Gz.shape
	if n == 0:
		return Gz.new_zeros(K, 0, 4, L)

	return torch.nn.grad.conv1d_input((n, 4, L), W0, Gz,
		padding=ICONV_PADDING).view(K, n // K, 4, L)


def basis_terms(M, x, r):
	"""What each of `basis_multipliers`' M (K, n, 4, L) adds, in float64:
	its hypothetical attributions with its pair's reference, over K, (K *
	n, 4 * L), and its sum((x - r) * M), (K, n), in M's dtype. x (1, 4, L)
	is the sequence's one-hot and r (K, 4, L) its references'."""

	K, n, _, L = M.shape
	dtype, M = M.dtype, M.double()
	x, r = x.double()[:, None].expand_as(M), r.double()[:, None].expand_as(M)
	hyp, = hypothetical_attributions((M.flatten(0, 1),), (x.flatten(0, 1),),
		(r.flatten(0, 1),))
	dot = ((x - r) * M).sum(dim=(-2, -1))
	return hyp.view(K * n, 4 * L) / K, dot.to(dtype)


def epilogue(m, lo, x, r, y_x, y_r, coef, D):
	"""Each pair's convergence delta at each output position, and each output
	position's hypothetical attributions averaged over the K references,
	those of the local multipliers only.

	m (K * B, 4, W) holds the local multipliers, row b over the positions
	[lo + b, lo + b + W), which may reach outside [0, L), x (1, 4, L) the
	sequence's one-hot and r (K, 4, L) its references', y_x (B,) and y_r
	(K, B) the attributed outputs, and coef (K, B, n) and D (K, n) the
	dense parts' coefficients and sums (`basis_terms`). The delta is
	`pisa`'s, |(y_x - y_r) - sum((x - r) * m)|, over the whole multipliers,
	and each pair's attributions are tangermeme's
	`hypothetical_attributions`. m may instead be `stem_backward`'s S, Rs
	and Dl with S's ``conv1d_input`` in its place, (M (B, 4, W), Rs, Dl),
	which give the same averages and deltas. Returns the attributions (B, 4,
	L) and the deltas (K, B). On CUDA this function runs compiled.
	"""

	L, (K, B) = x.shape[-1], y_r.shape
	W = (m[0] if isinstance(m, tuple) else m).shape[-1]
	at = positions(lo, B, W)
	inside = ((at >= 0) & (at < L))[:, None]
	at = at.clamp(0, L - 1)
	if isinstance(m, tuple):
		M, Rs, Dl = m
		delta = ((y_x - y_r) - Dl.sum(dim=-1) - (coef * D[:,
			None]).sum(dim=-1)).abs()
		hyp = torch.where(inside, M - Rs[:, None], 0) / K
	else:
		m = torch.where(inside, m.view(K, B, *m.shape[1:]), 0)
		x = x[0][:, at].transpose(0, 1).expand_as(m)
		r = r[:, :, at].transpose(1, 2)
		delta = ((y_x - y_r) - ((x - r) * m).sum(dim=(-2, -1))
			- (coef * D[:, None]).sum(dim=-1)).abs()
		hyp, = hypothetical_attributions((m.flatten(0, 1),),
			(x.flatten(0, 1),), (r.flatten(0, 1),))
		hyp = hyp.view(m.shape).mean(dim=0)

	# Each output's window back to its place in [0, L).
	index = torch.arange(L, device=hyp.device) - (lo + torch.arange(B,
		device=hyp.device)[:, None])
	keep = ((index >= 0) & (index < W))[:, None]
	hyp = hyp.gather(-1, index.clamp(0, W - 1)[:, None].expand(B, 4, L))
	return torch.where(keep, hyp, 0), delta


def dynamo_compiles():
	"""A count of Dynamo's work in this process, the frames handed to it
	(not counted under fullgraph) plus the graphs it compiled, which moves
	only when a compiled function traces; None when this torch does not
	keep it."""

	try:
		from torch._dynamo.utils import counters
		return int(counters["frames"]["total"]
			+ counters["stats"]["unique_graphs"])
	except Exception:
		return None


def sequence_bytes(K, L, C, H, n_blocks, itemsize=4):
	"""The bytes the forward and the rule terms of one sequence with K
	references allocate on the device, estimated analytically.

	The forward is `cherimoya._deep_lift_shap_engine.step_bytes`'s, for its
	K + 1 rows. The rule terms are, per pair, a slope (L x H) and two
	normalization terms (L x C) per block and the stem's slope (C x L). The
	forward's saved activations are freed once the terms are computed, so
	this overestimates by those during the chunks.
	"""

	per_row = (itemsize * L * (C + n_blocks * (C + H))
		+ itemsize * L * (4 * C + H) + itemsize * 4 * L + 4 * L)
	per_pair = itemsize * L * (n_blocks * (H + 2 * C) + C)
	return (K + 1) * per_row + K * per_pair


def output_bytes(K, L, C, H, itemsize=4):
	"""The bytes one output position of a chunk allocates on the device,
	estimated analytically: the backward's transients for its K rows, counted
	as in `step_bytes` (3 (L x C) + 2 (L x H) each), the multipliers and the
	epilogue's three (4 x L) tensors for each row, and its gradient at the
	final residual stream, built padded (2 (L x C)). The backward's local
	part runs over fewer than L positions, so this overestimates."""

	per_row = itemsize * L * (3 * C + 2 * H) + 4 * itemsize * 4 * L
	return K * per_row + 2 * itemsize * L * C


def pair_output_bytes(L, C, W, itemsize=4):
	"""The bytes one output position of a chunk allocates on the device when
	the blocks run by position, one pair at a time, and the bytes such a
	chunk allocates whatever its size, estimated analytically: for its row
	of the backward, about four (W x C) tensors over the widest window W;
	one (C x C) matrix per position; the gradient summed over the pairs (C x
	W); and the multipliers and the epilogue's (4 x L) tensors. The matrices
	cover W - 1 positions besides one per output. At 128 and 256 filters
	this overestimates by about a third."""

	per_output = itemsize * (5 * W * C + C * C + 4 * 4 * L)
	return per_output, itemsize * W * C * C


class Engine:
	"""PISA attributions of a Cherimoya model's profile logits for one
	channel: what ``pisa(model, X, hypothetical=True, ...)`` computes for a
	module that returns ``ControlWrapper(model)(X)[0][:, channel]``.

	The engine works one sequence at a time. The sequence and its K
	references are forwarded together, the rule terms of the K pairs are
	computed, and the backward runs over the pairs for chunks of B output
	positions. The hypothetical attributions are averaged over the
	references on the device. A last chunk with fewer than B positions is
	padded with the positions after it, so that every chunk has the same
	shapes, and the padded positions are dropped.

	On the CPU, B is at most 16. On CUDA:

	* B is at most the largest that `sequence_bytes` and `output_bytes`
	  estimate fits both `MEM_BUDGET` and `FREE_MEMORY_FRACTION` of the
	  device memory the process can use, from 1 to 128. In either case the
	  output positions are then split into chunks of equal size, B or one
	  fewer, before the last chunk is padded.
	* The forward, the rule terms, the backward, the epilogue and the dense
	  parts' fields (`basis_fields`) run under torch.compile, fullgraph and
	  with static shapes, as in
	  `cherimoya._deep_lift_shap_engine.Engine`, and a gc.collect() follows
	  every sequence that compiled: one during which `dynamo_compiles` moved,
	  or, when it is None, the first of each call and of each chunk size.
	* Sequences are pipelined. A sequence's results are copied back into
	  pinned memory without waiting; the host then waits for the previous
	  sequence, once, on its event, and draws the next sequence's
	  references while the device runs this one.
	* An out-of-memory error empties the allocator's cache, halves B and
	  runs the sequence again. At B = 1 the error propagates.
	* With Triton and a float32 model, when every reference position holds
	  at most one base, iconv's input gradient runs through `stem_backward`
	  and cuDNN on the gradient summed over the pairs; otherwise cuDNN runs
	  it for every pair.
	* Where iconv's input gradient so runs and the model has
	  `POSITION_FILTERS` filters, the backward runs one pair at a time over
	  chunks of up to `MAX_POSITION_OUTPUTS` output positions, sized by
	  `pair_output_bytes`, with the blocks below the top one in
	  `position_backward`, when a chunk holds at least
	  `position_min_outputs` outputs; `stem_backward` adds each pair's sums
	  to those before it.
	* Each Triton kernel's tiles, warps and stages are its fixed ones where
	  they fit the GPU's shared memory per block, and fewer stages, then
	  smaller tiles, where they do not (`block_settings`,
	  `position_settings`, `stem_settings`), so that one GPU type always
	  sums in one order. Where none fits, or Triton compiles none, that part
	  runs in torch: the blocks as without `block_backward` or by position,
	  and iconv's input gradient in cuDNN for every pair. The products run
	  in TF32 where `use_tf32` holds.


	Parameters
	----------
	model: cherimoya.Cherimoya
		The model, with float32 parameters, or float64 ones for tests. It is
		not changed.

	channel: int, optional
		The profile channel to attribute, an index into the model's
		``sum(signal_groups)`` profile channels. Default is 0.

	device: str or torch.device or None, optional
		Where the passes run. None means the model's device. Default is None.

	n_shuffles: int, optional
		The number of dinucleotide-shuffled references drawn per sequence
		when none are passed to `run`. Default is 20.

	random_state: int, optional
		The seed of those references. Default is 0.

	warning_threshold: float or None, optional
		`run` warns once if a convergence delta exceeds it. None means no
		such warning. Default is 1e-3.

	outputs_per_chunk: int or None, optional
		B, or None to choose it as described above. Default is None.
	"""

	def __init__(self, model, channel=0, device=None, n_shuffles=20,
		random_state=0, warning_threshold=1e-3, outputs_per_chunk=None):

		if device is None:
			device = model.iconv.weight.device

		self.device = torch.device(device)

		def get(t):
			return t.detach().to(self.device)

		self.W0, self.b0 = get(model.iconv.weight), get(model.iconv.bias)
		self.blocks = tuple(BlockWeights(get(b.conv.conv_weight),
			get(b.linear1.weight), get(b.linear2.weight), int(b.dilation))
			for b in model.blocks)
		self.Wf, self.bf = get(model.fconv.weight), get(model.fconv.bias)
		# Every output position's gradient at the final residual stream, over
		# the positions it reads from FCONV_PADDING before it.
		self.taps = self.Wf[channel, :model.n_filters].T.contiguous()
		self.rs = float(model.residual_scale)
		self.T = int(model.trimming)
		self.C, self.H = model.n_filters, model.expansion * model.n_filters
		self.n_ctl = model.n_control_tracks
		self.channel = channel

		self.n_shuffles, self.random_state = n_shuffles, random_state
		self.warning_threshold = warning_threshold
		self.outputs_per_chunk = outputs_per_chunk

		# `stem_backward`'s weight, and the rows it needs around the local
		# gradient at iconv's output, whose window is the seed's grown by
		# every block's dilation.
		self.W5, self._gathered, self._by_pair = None, False, False
		self._stem = (None, None)
		if stem_fused(self.W0):
			self._stem = tuple(stem_settings(self.C, self.W0.shape[-1],
				self.device, s_last) for s_last in (False, True))

		self._position = by_position(self.W0) and None not in self._stem
		self._tables = {}
		if self._stem[0] is not None:
			self.W5 = torch.cat([self.W0.permute(1, 2, 0),
				self.W0.new_zeros(1, self.W0.shape[-1], self.C)]).contiguous()
			self.Ws = self.taps.shape[0] + 2 * sum(b.d for b in self.blocks)
			self.stem_rows, self.pair_rows = (None if config is None else
				stem_padding(self.Ws, self.W0.shape[-1], config) for config in
				self._stem)

		passes = (forward_save, rule_terms, backward_pass, epilogue,
			basis_fields)
		if self.device.type == "cuda":
			kwargs = {"fullgraph": True, "dynamic": False, "mode": "default"}
			parameters = inspect.signature(torch.compile).parameters
			if {"recompile_limit", "isolate_recompiles"} <= set(parameters):
				kwargs.update(recompile_limit=RECOMPILE_LIMIT,
					isolate_recompiles=True)

			passes = tuple(torch.compile(fn, **kwargs) for fn in passes)

		(self.forward, self.rule_terms, self.backward, self.epilogue,
			self.basis) = passes

	def run(self, X, references=None, verbose=False):
		"""The hypothetical attributions of X at every output position,
		averaged over each sequence's references, as ``pisa(...,
		hypothetical=True)`` returns them, with each convergence delta and
		output.

		One warning is given if a delta exceeds `warning_threshold`, where
		tangermeme warns for each chunk of output positions.


		Parameters
		----------
		X: torch.Tensor, shape=(N, 4, L)
			The one-hot sequences, without unknown characters.

		references: torch.Tensor or None, optional
			(N, K, 4, L) one-hot references, or None to draw `n_shuffles`
			per sequence, as `pisa` draws them: those of every sequence are
			``dinucleotide_shuffle(X[i:i+1], n=n_shuffles,
			random_state=random_state)``. Default is None.

		verbose: bool, optional
			Whether to show a progress bar. Default is False.


		Returns
		-------
		result: Result
			The attributions (N, n_outputs, 4, L), deltas and outputs.
		"""

		X = X.to(device="cpu", dtype=torch.int8)
		if references is not None:
			references = references.to(device="cpu", dtype=torch.int8)

		K = self.n_shuffles if references is None else references.shape[1]
		N, _, L = X.shape
		n_outputs = L - 2 * self.T
		dtype = self.W0.dtype
		attr = torch.empty(N, n_outputs, 4, L, dtype=dtype)
		y_x = torch.empty(N, n_outputs, dtype=dtype)
		y_r = torch.empty(N, K, n_outputs, dtype=dtype)
		deltas = torch.empty(N, K, n_outputs, dtype=dtype)
		progress = tqdm(total=N, unit="seq", disable=not verbose)

		def consume(i, out):
			for kept, values in zip((y_x, y_r, deltas, attr), out):
				kept[i] = values

			progress.update(1)

		# Whether `_sequence` runs iconv's input gradient in `stem_backward`,
		# and the blocks by position where B allows.
		self._gathered = self.W5 is not None and one_hot(X if references is
			None else references)
		self._by_pair = (self._gathered and self._position
			and self._settings()[1] is not None)
		began = time.perf_counter()
		device = (torch.cuda.device(self.device) if self.device.type == "cuda"
			else contextlib.nullcontext())
		try:
			with device:
				self._sequences(X, references, K, n_outputs, consume)
		finally:
			progress.close()

		seconds = time.perf_counter() - began
		logger.info("fast PISA: %d sequences x %d references x %d outputs in "
			"%.1f s (%.0f pair-outputs/s)", N, K, n_outputs, seconds,
			N * K * n_outputs / seconds)

		threshold = self.warning_threshold
		if threshold is not None and bool((deltas > threshold).any()):
			warnings.warn("Convergence deltas too high: {} of {} pairs and "
				"outputs > {:g}, the largest {:.4g}".format(int((deltas
				> threshold).sum()), deltas.numel(), threshold,
				float(deltas.max())), RuntimeWarning, stacklevel=2)

		return Result(attr, deltas, y_x, y_r)

	def _chunk_size(self, n_outputs, K, L):
		"""B: the most output positions per chunk, `outputs_per_chunk` if
		given, `CPU_OUTPUTS_PER_CHUNK` on the CPU, and on CUDA the most whose
		`output_bytes` fit, after `sequence_bytes`, in the memory this
		process can use (what is free plus what its caching allocator holds
		unused), within the limits of the class docstring; then the size of
		the equal chunks that cover the positions with that many chunks."""

		if self.outputs_per_chunk is not None:
			B = self.outputs_per_chunk
		elif self.device.type == "cuda":
			free, _ = torch.cuda.mem_get_info(self.device)
			available = (free + torch.cuda.memory_reserved(self.device)
				- torch.cuda.memory_allocated(self.device))
			itemsize = self.W0.dtype.itemsize
			budget = min(MEM_BUDGET, FREE_MEMORY_FRACTION * available)
			fixed = sequence_bytes(K, L, self.C, self.H, len(self.blocks),
				itemsize)
			per_output = output_bytes(K, L, self.C, self.H, itemsize)
			B = int((budget - fixed) // per_output)
			B = max(1, min(MAX_OUTPUTS_PER_CHUNK, B))
			if self._by_pair:
				pair, chunk = pair_output_bytes(L, self.C, self.taps.shape[0]
					+ 2 * sum(b.d for b in self.blocks), itemsize)
				pair_B = min(MAX_POSITION_OUTPUTS, int((budget - fixed - chunk)
					// pair))
				if min(pair_B, n_outputs) >= position_min_outputs(self.C):
					B, per_output = pair_B, pair

			logger.info("at most %d outputs per chunk (%.2f GB each, %.2f GB "
				"per sequence, %.1f GB available)", B, per_output / 1e9,
				fixed / 1e9, available / 1e9)
		else:
			B = CPU_OUTPUTS_PER_CHUNK

		return math.ceil(n_outputs / math.ceil(n_outputs / min(B, n_outputs)))

	def _position_tables(self, B):
		"""`position_table` for each block below the top one, top first, for
		chunks of B outputs, made once per B."""

		BLOCK_B = self._settings()[1][1][0]
		if (B, BLOCK_B) not in self._tables:
			width, grown, tables = self.taps.shape[0], 0, []
			for block in tuple(reversed(self.blocks))[:-1]:
				grown += block.d
				tables.append(position_table(B, width + 2 * grown, BLOCK_B,
					self.device))

			self._tables[B, BLOCK_B] = tuple(tables)

		return self._tables[B, BLOCK_B]

	def _settings(self):
		"""The kernels' settings for the blocks on the engine's device at the
		precision `use_tf32` gives, (`block_settings`, `position_settings`),
		None for a kernel that does not run there or that the engine does
		not use."""

		if self.device.type != "cuda" or not HAS_TRITON:
			return None, None

		tf32 = use_tf32(self.device)
		return ((block_settings(self.C, self.H, self.device, tf32)
			if fused(self.taps) else None), (position_settings(self.C, self.H,
			self.device, tf32) if self._position else None))

	def _references(self, X, references, i, K):
		"""The references of sequence i, (K, 4, L) int8: those passed in, or
		those `pisa` draws, ``dinucleotide_shuffle(X[i:i+1], n=K,
		random_state=random_state)``."""

		if references is not None:
			return references[i]

		return dinucleotide_shuffle(X[i:i + 1], n=K,
			random_state=self.random_state)[0]

	def _profile(self, hL):
		"""The attributed channel's profile logits of each row, (R,
		n_outputs), from the final residual stream hL (R, L, C): the model's
		profile convolution, over the stream and the all-zero control tracks
		`ControlWrapper` feeds, at positions [T, L - T).

		The convolution runs in float64 and its result is cast back. The
		seeds are the convolution's weights as they are, so a TF32
		convolution here would round the outputs the deltas compare the
		attributions with, and not the attributions.
		"""

		H = hL.transpose(1, 2).contiguous()
		L = H.shape[-1]
		if self.n_ctl:
			H = torch.cat([H, H.new_zeros(len(H), self.n_ctl, L)], dim=1)

		c = slice(self.channel, self.channel + 1)
		y = F.conv1d(H.double(), self.Wf[c].double(), self.bf[c].double(),
			padding=FCONV_PADDING)
		return y[:, 0, self.T:L - self.T].to(hL.dtype)

	def _sequence(self, inputs, K, B, n_outputs):
		"""One sequence on the engine's device: its rows and its references',
		(K + 1, 4, L), an int8 one-hot, -> (y_x, y_r, deltas, attr), on the
		device. Each large intermediate is released as soon as the next stage
		no longer needs it. When `run` found that every reference position
		holds at most one base, iconv's input gradient runs in
		`stem_backward`, where the engine has it, and, with B large enough,
		the backward runs one pair at a time (the class docstring)."""

		X1h = inputs.to(self.W0.dtype)
		L = X1h.shape[-1]
		z0 = F.conv1d(X1h, self.W0, self.b0, padding=ICONV_PADDING)
		hL, saved = self.forward(z0, self.blocks, self.rs, CONV_NORM_EPS)
		y = self._profile(hL)
		del hL
		terms, stem = self.rule_terms(saved, z0)
		del saved, z0
		x, r = X1h[:1], X1h[1:]
		hyp, D = basis_terms(basis_multipliers(self.basis(terms, stem,
			self.blocks, self.rs), self.W0, K), x, r)

		n_chunks = math.ceil(n_outputs / B)
		y = F.pad(y, (0, n_chunks * B - n_outputs))
		attr = torch.empty(n_chunks * B, 4, L, dtype=y.dtype, device=y.device)
		deltas = torch.empty(K, n_chunks * B, dtype=y.dtype, device=y.device)
		coefs = torch.empty(K, n_chunks * B, D.shape[-1], dtype=y.dtype,
			device=y.device)
		tf32 = use_tf32(self.device)
		config = self._settings()
		gathered = self._gathered
		stem_last = rows = None
		if gathered:
			stem_last, rows = stem[:, 0].transpose(1, 2).contiguous(), \
				self.stem_rows
			J = torch.where(r.sum(dim=1) > 0, r.argmax(dim=1), 4).to(
				torch.int32)
			# In float64: cuDNN may run a float32 convolution in TF32, and
			# the deltas' sum((x - r) * m) is float32 otherwise.
			Ex = F.conv1d(x.double(), self.W0.double(), padding=ICONV_PADDING)[
				0].T.to(x.dtype).contiguous()
			Wm = self.Ws + 2 * ICONV_PADDING

		by_pair = self._by_pair and B >= position_min_outputs(self.C)
		if by_pair:
			tables = self._position_tables(B)
			pairs = [(tuple(BlockTerms(*(f[k:k + 1] for f in t)) for t in
				terms), stem[k:k + 1], stem_last[k:k + 1], J[k:k + 1]) for k in
				range(K)]

		for a in range(0, n_outputs, B):
			# The first position output a reads, on the device, so that a new
			# chunk neither recompiles nor waits on the host.
			lo = torch.full((), self.T + a - FCONV_PADDING, dtype=torch.int64,
				device=self.device)
			if by_pair:
				coef = coefs.new_empty(K, B, D.shape[-1])
				Dl = coefs.new_empty(K, B, stem_partials(Wm, self.C,
					self._stem[1]))
				S = Rs = None
				for k, (terms_k, stem_k, last_k, J_k) in enumerate(pairs):
					Gz, coef_k, lo_k = self.backward(self.taps, lo, B, terms_k,
						stem_k, self.blocks, self.rs, tf32, last_k, self.pair_rows,
						tables, config)
					coef[k] = coef_k[0]
					S, Rs, _ = stem_backward(Gz, self.W5, J_k, Ex, lo_k, Wm,
						self._stem[1], S=S, Rs=Rs, Dl=Dl[k:k + 1], s_last=True)
					del Gz

				lo = lo_k
				m = (torch.nn.grad.conv1d_input((B, 4, Wm), self.W0,
					S.transpose(1, 2), padding=ICONV_PADDING), Rs, Dl)
				del S
			elif gathered:
				Gz, coef, lo = self.backward(self.taps, lo, B, terms, stem,
					self.blocks, self.rs, tf32, stem_last, rows, None, config)
				S, Rs, Dl = stem_backward(Gz, self.W5, J, Ex, lo, Wm,
					self._stem[0])
				del Gz
				m = (torch.nn.grad.conv1d_input((B, 4, Wm), self.W0, S,
					padding=ICONV_PADDING), Rs, Dl)
				del S
			else:
				Gz, coef, lo = self.backward(self.taps, lo, B, terms, stem,
					self.blocks, self.rs, tf32, config=config)

				# iconv's input gradient, the call autograd makes for F.conv1d
				# when only the input needs a gradient.
				m = torch.nn.grad.conv1d_input((K * B, 4, Gz.shape[-1]),
					self.W0, Gz, padding=ICONV_PADDING)
				del Gz

			attr[a:a + B], deltas[:, a:a + B] = self.epilogue(m, lo, x, r,
				y[0, a:a + B], y[1:, a:a + B], coef, D)
			coefs[:, a:a + B] = coef

		# The dense parts' attributions, every output position's
		# coefficients times the basis's, in float64.
		flat = attr.view(n_chunks * B, -1)
		flat.copy_(torch.addmm(flat.double(), coefs.transpose(0, 1).reshape(
			n_chunks * B, K * D.shape[-1]).double(), hyp))
		return (y[0, :n_outputs], y[1:, :n_outputs], deltas[:, :n_outputs],
			attr[:n_outputs])

	def _sequences(self, X, references, K, n_outputs, consume):
		"""Every sequence, in order; consume(i, out) receives the results of
		sequence i, (y_x, y_r, deltas, attr), once they are on the host. On
		CUDA a sequence is queued before the host waits for the one before it
		(the class docstring)."""

		N, _, L = X.shape
		cuda = self.device.type == "cuda"
		B = self._chunk_size(n_outputs, K, L)

		def finish(i, out, done):
			if done is not None:
				done.synchronize()

			consume(i, out)

		i, pending, compiles = 0, None, cuda
		while i < N:
			count = dynamo_compiles() if cuda else None
			rows = torch.cat([X[i:i + 1], self._references(X, references, i,
				K)])
			if cuda:
				rows = rows.pin_memory()

			# The error is handled outside the except clause: its traceback
			# holds the failed sequence's tensors, which must be freed before
			# the cache is emptied.
			failed, done = None, None
			try:
				out = self._sequence(rows.to(self.device, non_blocking=True), K,
					B, n_outputs)
				if cuda:
					out = [t.to("cpu", non_blocking=True) for t in out]
					done = torch.cuda.Event(blocking=True)
					done.record()
			except torch.cuda.OutOfMemoryError as exc:
				if B == 1:
					raise

				failed = str(exc).strip().split("\n")[0]

			if failed is not None:
				if pending is not None:
					finish(*pending)
					pending = None

				torch.cuda.synchronize(self.device)
				gc.collect()
				torch.cuda.empty_cache()
				logger.warning("out of GPU memory with %d outputs per chunk; "
					"retrying with %d (%s)", B, B // 2, failed)
				B //= 2
				compiles = True
				continue

			# A call that compiles leaves Dynamo's tracing state in reference
			# cycles that hold a whole sequence's activations. Only the cyclic
			# garbage collector frees them. A full collection scans every
			# object in the process, so it runs only after a sequence that
			# traced.
			traced = compiles if count is None else dynamo_compiles() != count
			if traced:
				gc.collect()

			compiles = False

			if pending is not None:
				finish(*pending)

			pending = (i, out, done)
			i += 1

		if pending is not None:
			finish(*pending)
