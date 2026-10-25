# _deep_lift_shap_engine.py
# Author: Eugenio Mattei

"""
A fast DeepLIFT/DeepSHAP engine for the log-counts of a Cherimoya model.

`tangermeme.deep_lift_shap.deep_lift_shap` attributes a model through
backward hooks, over batches that hold each sequence-reference pair as two
rows: every sequence is forwarded once per reference, and the backward runs
over both halves of the batch. This engine computes the same multipliers
from a forward and a backward written out by hand:

* The forward runs over a joint batch per step: each sequence once, then its
  K references. Every operation is row-wise, so the K copies of a sequence
  that tangermeme forwards are the same row.
* The backward runs over the sequence half of each pair only. tangermeme's
  rule for a sequence row reads the reference row's activations but never
  its gradient, so the reference half is never needed.
* The rules are tangermeme's: the rescale rule (`_rescale`) for every
  GELU, and the closed-form normalization rule that
  `cherimoya.deep_lift_shap.conv_norm_op` applies to `FusedDilatedConvNorm`
  (`_layer_normalization_stats`, `_terms` and `_grad`), called as tensor
  functions. The references are tangermeme's `dinucleotide_shuffle`, drawn
  with the seeds `deep_lift_shap` uses.

Rows are laid out as ``[x_0, ..., x_{S-1}, r_00, ..., r_{S-1,K-1}]``: S
sequences, then the K references of each. `split_rows` views them as
sequences (S, 1, ...) and references (S, K, ...), so that a sequence row
broadcasts over its K pairs without being copied.

This module is private. `cherimoya.deep_lift_shap.deep_lift_shap` runs it
as its fast path, for float32 Cherimoya models only, and compares it with
tangermeme's `deep_lift_shap` on every call that uses it.
"""

import collections
import contextlib
import gc
import inspect
import logging
import time
import warnings

from typing import NamedTuple

import numpy
import torch
import torch.nn.functional as F

from tangermeme._deep_lift_utils import _layer_normalization_grad
from tangermeme._deep_lift_utils import _layer_normalization_stats
from tangermeme._deep_lift_utils import _layer_normalization_terms
from tangermeme._deep_lift_utils import _rescale
from tangermeme.deep_lift_shap import hypothetical_attributions
from tangermeme.ersatz import dinucleotide_shuffle
from tqdm import tqdm

from .cheri import CONV_NORM_EPS
from .cheri import _cheri_conv
from .cheri import _cheri_conv_norm_cpu


logger = logging.getLogger(__name__)

ICONV_PADDING = 10

CPU_SEQS_PER_STEP = 8
MAX_SEQS_PER_STEP = 32

MEM_BUDGET = 12e9
FREE_MEMORY_FRACTION = 0.6
"""On CUDA a step is planned for at most `MEM_BUDGET` bytes and this fraction
of the device memory the process can use."""

RECOMPILE_LIMIT = 8
"""Compiles allowed per compiled pass of one engine: one per step size, and
the step size halves at most 5 times from 32."""

RELATIVE_DELTA_MAX = 0.01
"""`Engine.run` warns when the mean convergence delta exceeds this fraction
of the mean |y_x - y_r|, the bound `deep_lift_shap` checks per batch."""


class BlockWeights(NamedTuple):
	"""One CheriBlock's tensors."""

	cw: torch.Tensor  # (3, C) depthwise taps; row 0 reads l - d, row 2 l + d
	W1: torch.Tensor  # (H, C) linear1, no bias
	W2: torch.Tensor  # (C, H) linear2, no bias
	d: int  # dilation, 2**i


class BlockSaved(NamedTuple):
	"""What one block's backward rules read. Nothing else is kept: the
	normalized stream and the GELU's output are recomputed."""

	y: torch.Tensor  # (R, L, C) depthwise conv output, the norm's input
	u: torch.Tensor  # (R, L, H) linear1 output, the GELU's input
	mu: torch.Tensor  # (R,) the rule's mean of y over the (L, C) plane
	v: torch.Tensor  # (R,) the rule's inverse std, (var + eps) ** -0.5


class Result(NamedTuple):
	"""`Engine.run`'s output: CPU tensors in the model's dtype."""

	attr: torch.Tensor  # (N, 4, L) hypothetical attributions, averaged
	deltas: torch.Tensor  # (N, K) convergence deltas
	y_x: torch.Tensor  # (N,) the attributed output of each sequence
	y_r: torch.Tensor  # (N, K) the attributed output of each reference


def dw3(h, cw, d):
	"""The 3-tap dilated depthwise convolution as a shift-add.

	h has shape (..., L, C), channels last, and cw has shape (3, C). The
	output is ``y[l] = cw[0] * h[l - d] + cw[1] * h[l] + cw[2] * h[l + d]``,
	with zeros outside [0, L). With the taps reversed it is the
	convolution's transpose.
	"""

	L = h.shape[-2]
	hp = F.pad(h, (0, 0, d, d))
	return (hp[..., :L, :] * cw[0] + hp[..., d:d + L, :] * cw[1]
		+ hp[..., 2 * d:2 * d + L, :] * cw[2])


def forward_save(z0, blocks, rs, eps):
	"""Run the stem GELU and the blocks on z0 (R, C, L), iconv's output,
	keeping what the backward rules read.

	The function is row-wise: a batch can hold sequences and references in
	any order. On the CPU it is the model's CPU path, operation for
	operation. On CUDA, where it runs compiled, the convolution is `dw3` and
	the stream is normalized with the rule's own statistics, where the
	model's Triton kernel computes the variance as E[y^2] - mean^2. The
	saved mu and v are the rule's statistics on either device.


	Returns
	-------
	hL: torch.Tensor, shape=(R, L, C)
		The final residual stream, channels last.

	saved: list of BlockSaved
		One per block.
	"""

	h = F.gelu(z0, approximate="tanh").transpose(1, 2).contiguous()
	saved = []
	for cw, W1, W2, d in blocks:
		y = dw3(h, cw, d) if h.is_cuda else _cheri_conv(h, cw, d)
		mu, v = _layer_normalization_stats(y, 2, eps)
		yhat = ((y - mu) * v if h.is_cuda
			else _cheri_conv_norm_cpu(h, cw, d, eps))

		# The block's residual, in the model's operation order.
		u = F.linear(yhat, W1)
		h = h + F.linear(F.gelu(u, approximate="tanh"), W2) * rs

		# Saved flat. Saved as (R, 1, 1), they change how Inductor fuses the
		# rule's reductions, and the attributions of a trained 128-filter
		# model moved by up to 1.6e-4 (relative L2) with the rounding.
		saved.append(BlockSaved(y, u, mu.view(-1), v.view(-1)))

	return h, saved


def split_rows(t, S, K):
	"""View rows ``[x_0..x_{S-1}, r_00..r_{S-1,K-1}]`` as sequences
	(S, 1, ...) and references (S, K, ...). Both are views, so `t` must be
	contiguous along its rows."""

	return t[:S].unsqueeze(1), t[S:].view(S, K, *t.shape[1:])


def gelu_rescale(G, z, S, K):
	"""The rescale rule through GELU(tanh) on the sequence half: tangermeme's
	`_rescale`, as its `_nonlinear` applies it.

	G (S, K, ...) is each pair's gradient at the GELU's output, and z
	(S + S*K, ...) the GELU's input over the joint rows. The result,
	(S, K, ...), is the gradient at the GELU's input: G times the secant,
	or, where the input barely moves, the plain gradient that nn.GELU's
	autograd hands tangermeme. The GELU's outputs are recomputed from z over
	all rows at once, as the forward computed them.
	"""

	zx, zr = split_rows(z, S, K)
	gx, gr = split_rows(F.gelu(z, approximate="tanh"), S, K)
	gradient = torch.ops.aten.gelu_backward(G, zx.expand_as(G),
		approximate="tanh")
	return _rescale(zx - zr, gx - gr, gradient, G)


def ln_rule(Gn, y, mu, v, S, K):
	"""`conv_norm_op`'s normalization rule on the sequence half: tangermeme's
	`_layer_normalization_terms` and `_layer_normalization_grad` over each
	row's (L, C) plane, with each sequence row broadcast against its K
	reference rows.

	Gn (S, K, L, C) is each pair's gradient at the normalization's output, y
	(S + S*K, L, C) its input over the joint rows, and mu and v (S + S*K,)
	its statistics, saved by the forward. Returns the gradient at y,
	(S, K, L, C).
	"""

	yx, yr = split_rows(y, S, K)
	mx, mr = (t[..., None, None] for t in split_rows(mu, S, K))
	vx, vr = (t[..., None, None] for t in split_rows(v, S, K))
	terms = _layer_normalization_terms(yx, mx, vx, yr, mr, vr, 2)
	return _layer_normalization_grad(Gn, *terms, 2)


def backward_pass(G0, saved, z0, blocks, rs, S, K):
	"""The backward over the S*K pairs, from the final residual stream to
	iconv's output: G0 (L, C), the count gradient every pair shares, ->
	(S*K, C, L).

	Each block computes ``h + linear2(GELU(linear1(conv_norm(h)))) * rs``.
	Backwards through each, last block first: the ordinary gradient through
	the scale and linear2, the rescale rule at the GELU, the ordinary
	gradient through linear1, the normalization rule pushed back through the
	depthwise convolution, and the residual. Then the stem GELU's rescale
	rule, in its (C, L) layout. G0 enters broadcast over the pairs, without
	a copy. On CUDA this function runs compiled.
	"""

	L, C = G0.shape
	G = G0.expand(S, K, L, C)
	for s, block in zip(reversed(saved), reversed(blocks)):
		Gu = gelu_rescale(torch.matmul(G * rs, block.W2), s.u, S, K)
		Gy = ln_rule(torch.matmul(Gu, block.W1), s.y, s.mu, s.v, S, K)
		G = G + dw3(Gy, block.cw.flip(0), block.d)

	Gz = gelu_rescale(G.transpose(-1, -2), z0, S, K)
	return Gz.reshape(S * K, *Gz.shape[-2:]).contiguous()


def epilogue(m, X1h, y_x, y_r, S, K):
	"""Each pair's convergence delta, and each sequence's hypothetical
	attributions averaged over its K references.

	m (S*K, 4, L) holds the multipliers, X1h (S + S*K, 4, L) the joint
	one-hot rows, and y_x (S,) and y_r (S, K) the attributed outputs. The
	delta is `deep_lift_shap`'s, |(y_x - y_r) - sum((x - r) * m)|, and each
	pair's attributions are tangermeme's `hypothetical_attributions`.
	Returns the attributions (S, 4, L) and the deltas (S, K). On CUDA this
	function runs compiled.
	"""

	x, r = split_rows(X1h, S, K)
	m = m.view(S, K, *m.shape[1:])
	delta = ((y_x[:, None] - y_r) - ((x - r) * m).sum(dim=(-2, -1))).abs()
	hyp, = hypothetical_attributions((m.flatten(0, 1),),
		(x.expand_as(r).flatten(0, 1),), (r.flatten(0, 1),))
	return hyp.view(S, K, *hyp.shape[1:]).mean(dim=1), delta


def step_bytes(S, K, L, C, H, n_blocks, itemsize=4):
	"""The bytes one step of S sequences with K references each allocates on
	the device, estimated analytically.

	A step forwards R = S(K + 1) rows and runs the backward over P = SK
	pairs, all of length L. Per row: kept from the forward to the backward,
	z0 (C x L) and, per block, y (L x C) and u (L x H); the forward's
	transients, counted as if none were reused (the residual stream, yhat,
	the GELU's output, linear2's output and the new stream: 4 (L x C) +
	(L x H)); the one-hot (4 x L) and its int8 copy (4 x L bytes). Per pair:
	the backward's transients, counted the same way (3 (L x C) + 2 (L x H)),
	and the multipliers (4 x L). The forward's and the backward's transients
	never coexist, so the sum overestimates by the smaller of the two.
	cuDNN's workspace and the weights are not counted.
	"""

	rows, pairs = S * (K + 1), S * K
	per_row = (itemsize * L * (C + n_blocks * (C + H))
		+ itemsize * L * (4 * C + H) + itemsize * 4 * L + 4 * L)
	per_pair = itemsize * L * (3 * C + 2 * H) + itemsize * 4 * L
	return rows * per_row + pairs * per_pair


def _joint_rows(X, references, S, pin_memory):
	"""A step's rows ``[x_0..x_{S-1}, r_00..r_{S-1,K-1}]`` as one int8
	one-hot (S + S*K, 4, L), padded with the last sequence and its
	references, in pinned memory if `pin_memory`.

	X (n, 4, L) and references (n, K, 4, L), n <= S, are int8 one-hots.
	"""

	n, K = references.shape[:2]
	rows = torch.empty(S + S * K, *X.shape[1:], dtype=torch.int8,
		pin_memory=pin_memory)
	rows[:n] = X
	rows[n:S] = X[-1]
	refs = rows[S:].view(S, K, *X.shape[1:])
	refs[:n] = references
	refs[n:] = references[-1]
	return rows


class Engine:
	"""DeepLIFT/DeepSHAP attributions of a Cherimoya model's count head:
	what ``deep_lift_shap(LogCountWrapper(ControlWrapper(model), group), X,
	hypothetical=True, ...)`` computes.

	The engine works in steps of S whole sequences, each with its K
	references. A step's joint rows are forwarded once, the backward runs
	over the S*K pairs, and the hypothetical attributions are averaged over
	each sequence's references on the device. A last step with fewer than S
	sequences is padded by repeating its last sequence and references, so
	that every step has the same shapes, and the padded rows are dropped.

	On the CPU, S is 8. On CUDA:

	* S is the largest that `step_bytes` estimates fits both `MEM_BUDGET`
	  and `FREE_MEMORY_FRACTION` of the device memory the process can use,
	  from 1 to 32.
	* The forward, the backward and the epilogue run under torch.compile,
	  fullgraph and with static shapes. The weights are arguments, so the
	  blocks unroll into one graph each way and every matrix product stays
	  an extern cuBLAS call. iconv, its input gradient and the count head run
	  eagerly, with the default path's operations. Where torch.compile takes
	  `isolate_recompiles` (torch 2.13 and later), each engine keeps its
	  compiled code apart from other engines'. Compiling leaves the traced
	  step's tensors in reference cycles, so a gc.collect() follows every
	  step that compiled.
	* Steps are pipelined. A step's inputs are copied from pinned memory,
	  the step is queued, and its results are copied back into pinned
	  memory, all without waiting; the host then waits for the previous step,
	  once, on its event, and draws the next step's references while the
	  device runs this one. PyTorch's pinned-memory allocator hands a buffer
	  out again only after the copies queued on it have finished.
	* An out-of-memory error in a step empties the allocator's cache, halves
	  S and runs the step's sequences again. At S = 1 the error propagates.


	Parameters
	----------
	model: cherimoya.Cherimoya
		The model, with float32 parameters, or float64 ones for tests. It is
		not changed.

	group: int or None, optional
		The signal group to attribute, as `LogCountWrapper` takes it. None
		attributes the first group. Default is 0.

	device: str or torch.device or None, optional
		Where the steps run. None means the model's device. Default is None.

	n_shuffles: int, optional
		The number of dinucleotide-shuffled references drawn per sequence
		when none are passed to `run`. Default is 20.

	random_state: int, optional
		The base seed of those references. Default is 0.

	warning_threshold: float or None, optional
		`run` warns once if a convergence delta exceeds it. None means no
		such warning. Default is 1e-3.

	seqs_per_step: int or None, optional
		S, or None to choose it as described above. Default is None.
	"""

	def __init__(self, model, group=0, device=None, n_shuffles=20,
		random_state=0, warning_threshold=1e-3, seqs_per_step=None):

		if device is None:
			device = model.iconv.weight.device

		self.device = torch.device(device)

		def get(t):
			return t.detach().to(self.device)

		self.W0, self.b0 = get(model.iconv.weight), get(model.iconv.bias)
		self.blocks = tuple(BlockWeights(get(b.conv.conv_weight),
			get(b.linear1.weight), get(b.linear2.weight), int(b.dilation))
			for b in model.blocks)
		self.Wl, self.bl = get(model.linear.weight), get(model.linear.bias)
		self.rs = float(model.residual_scale)
		self.T = int(model.trimming)
		self.C, self.H = model.n_filters, model.expansion * model.n_filters
		self.n_ctl = model.n_control_tracks
		self.group = 0 if group is None else group

		self.n_shuffles, self.random_state = n_shuffles, random_state
		self.warning_threshold = warning_threshold
		self.seqs_per_step = seqs_per_step

		passes = (forward_save, backward_pass, epilogue)
		if self.device.type == "cuda":
			kwargs = {"fullgraph": True, "dynamic": False, "mode": "default"}
			parameters = inspect.signature(torch.compile).parameters
			if {"recompile_limit", "isolate_recompiles"} <= set(parameters):
				kwargs.update(recompile_limit=RECOMPILE_LIMIT,
					isolate_recompiles=True)

			passes = tuple(torch.compile(fn, **kwargs) for fn in passes)

		self.forward, self.backward, self.epilogue = passes

	def run(self, X, references=None, verbose=False):
		"""The hypothetical attributions of X averaged over each sequence's
		references, as ``deep_lift_shap(..., hypothetical=True)`` returns
		them, with each pair's convergence delta and outputs.

		One warning is given if a delta exceeds `warning_threshold`, and one
		if the mean delta exceeds `RELATIVE_DELTA_MAX` of the mean output
		difference, where tangermeme checks both per batch.


		Parameters
		----------
		X: torch.Tensor, shape=(N, 4, L)
			The one-hot sequences, without unknown characters.

		references: torch.Tensor or None, optional
			(N, K, 4, L) one-hot references, or None to draw `n_shuffles`
			per sequence: reference k of every sequence is its dinucleotide
			shuffle with the seed ``random_state + k``, as `deep_lift_shap`
			draws it. Default is None.

		verbose: bool, optional
			Whether to show a progress bar. Default is False.


		Returns
		-------
		result: Result
			The attributions (N, 4, L), deltas and outputs.
		"""

		X = X.to(device="cpu", dtype=torch.int8)
		if references is not None:
			references = references.to(device="cpu", dtype=torch.int8)

		K = self.n_shuffles if references is None else references.shape[1]
		N, _, L = X.shape
		dtype = self.W0.dtype
		attr = torch.empty(N, 4, L, dtype=dtype)
		y_x = torch.empty(N, dtype=dtype)
		y_r = torch.empty(N, K, dtype=dtype)
		deltas = torch.empty(N, K, dtype=dtype)
		progress = tqdm(total=N, unit="seq", disable=not verbose)

		def consume(a, b, out):
			for kept, values in zip((y_x, y_r, deltas, attr), out):
				kept[a:b] = values[:b - a]

			progress.update(b - a)

		# The gradient of the attributed count at the final residual stream,
		# which every pair shares: the head averages positions [T, L - T)
		# and applies the count Linear, and the control tracks' feature is a
		# constant.
		G0 = torch.zeros(L, self.C, dtype=dtype, device=self.device)
		G0[self.T:L - self.T] = self.Wl[self.group, :self.C] / (L - 2 * self.T)

		began = time.perf_counter()
		device = (torch.cuda.device(self.device) if self.device.type == "cuda"
			else contextlib.nullcontext())
		try:
			with device:
				self._steps(X, references, K, G0, consume)
		finally:
			progress.close()

		seconds = time.perf_counter() - began
		logger.info("fast DeepLIFT/SHAP: %d sequences x %d references in "
			"%.1f s (%.0f pairs/s)", N, K, seconds, N * K / seconds)

		threshold = self.warning_threshold
		if threshold is not None and bool((deltas > threshold).any()):
			warnings.warn("Convergence deltas too high: {} of {} pairs > {:g}, "
				"the largest {:.4g}".format(int((deltas > threshold).sum()),
				deltas.numel(), threshold, float(deltas.max())),
				RuntimeWarning, stacklevel=2)

		relative = float(deltas.double().mean() / (y_x[:, None]
			- y_r).double().abs().mean().clamp_min(1e-12))
		if relative > RELATIVE_DELTA_MAX:
			warnings.warn("Convergence deltas are {:.1%} of the output "
				"differences over {} pairs, above {:.0%}: the attributions do "
				"not sum to the change in the output".format(relative,
				deltas.numel(), RELATIVE_DELTA_MAX), RuntimeWarning,
				stacklevel=2)

		return Result(attr, deltas, y_x, y_r)

	def _step_size(self, N, K, L):
		"""S: `seqs_per_step` if given, `CPU_SEQS_PER_STEP` on the CPU, and on
		CUDA the most whose `step_bytes` fits the memory this process can use
		(what is free plus what its caching allocator holds unused), within
		the limits of the class docstring. Never more than N."""

		if self.seqs_per_step is not None:
			S = self.seqs_per_step
		elif self.device.type == "cuda":
			free, _ = torch.cuda.mem_get_info(self.device)
			available = (free + torch.cuda.memory_reserved(self.device)
				- torch.cuda.memory_allocated(self.device))
			per_sequence = step_bytes(1, K, L, self.C, self.H,
				len(self.blocks), self.W0.dtype.itemsize)
			S = int(min(MEM_BUDGET, FREE_MEMORY_FRACTION * available)
				// per_sequence)
			S = max(1, min(MAX_SEQS_PER_STEP, S))
			logger.info("%d sequences per step (%.2f GB each, %.1f GB "
				"available)", S, per_sequence / 1e9, available / 1e9)
		else:
			S = CPU_SEQS_PER_STEP

		return min(S, N)

	def _references(self, X, references, a, b, K):
		"""The references of the sequences [a, b), (b - a, K, 4, L) int8: those
		passed in, or drawn in one call to `dinucleotide_shuffle`, reference k
		of each sequence with the seed random_state + k."""

		if references is not None:
			return references[a:b]

		rows = X[a:b].repeat_interleave(K, dim=0)
		seeds = self.random_state + numpy.tile(numpy.arange(K), b - a)
		drawn = dinucleotide_shuffle(rows, n=1, random_state=seeds)[:, 0]
		return drawn.view(b - a, K, *X.shape[1:])

	def _head(self, hL):
		"""The attributed log-count of each row from the final residual stream
		hL (R, L, C): the model's count head, then `LogCountWrapper`'s group.

		The control tracks `ControlWrapper` feeds are zeros, so their count
		feature is log(0 + 1) = 0. The pooled stream is cast to the count
		Linear's dtype where the model calls ``.float()``: the same operation
		for float32 parameters, and float64 in a float64 model.
		"""

		L = hL.shape[1]
		H = hL.transpose(1, 2).contiguous()
		pooled = torch.mean(H[:, :, self.T:L - self.T].to(self.Wl.dtype), dim=2)
		if self.n_ctl:
			pooled = torch.cat([pooled, pooled.new_zeros(len(pooled), 1)],
				dim=-1)

		return F.linear(pooled, self.Wl, self.bl)[:, self.group]

	def _step(self, inputs, S, K, G0):
		"""One step on the engine's device: the joint rows (S + S*K, 4, L), an
		int8 one-hot, -> the step's results, on the device. Each large
		intermediate is released as soon as the next stage no longer needs
		it."""

		X1h = inputs.to(self.W0.dtype)
		z0 = F.conv1d(X1h, self.W0, self.b0, padding=ICONV_PADDING)
		hL, saved = self.forward(z0, self.blocks, self.rs, CONV_NORM_EPS)
		y = self._head(hL)
		del hL
		Gz = self.backward(G0, saved, z0, self.blocks, self.rs, S, K)
		del saved, z0

		# iconv's input gradient, the call autograd makes for F.conv1d when
		# only the input needs a gradient: the multipliers (S*K, 4, L).
		m = torch.nn.grad.conv1d_input((S * K, 4, X1h.shape[-1]), self.W0, Gz,
			padding=ICONV_PADDING)
		del Gz
		y_x, y_r = y[:S], y[S:].view(S, K)
		attr, delta = self.epilogue(m, X1h, y_x, y_r, S, K)
		return y_x, y_r, delta, attr

	def _steps(self, X, references, K, G0, consume):
		"""Every step, in order; consume(a, b, out) receives the results of the
		sequences [a, b) of each, (y_x, y_r, delta, attr), once they are on
		the host. On CUDA a step is queued before the host waits for the one
		before it (the class docstring)."""

		N, _, L = X.shape
		cuda = self.device.type == "cuda"
		S = self._step_size(N, K, L)

		def finish(a, b, out, done):
			if done is not None:
				done.synchronize()

			consume(a, b, out)

		starts = collections.deque(range(0, N, S))
		pending, compiles = None, cuda
		while starts:
			a = starts.popleft()
			b = min(a + S, N)
			rows = _joint_rows(X[a:b], self._references(X, references, a, b, K),
				S, pin_memory=cuda)

			# The error is handled outside the except clause: its traceback
			# holds the failed step's tensors, which must be freed before the
			# cache is emptied.
			failed, done = None, None
			try:
				out = self._step(rows.to(self.device, non_blocking=True), S, K,
					G0)
				if cuda:
					out = [t.to("cpu", non_blocking=True) for t in out]
					done = torch.cuda.Event(blocking=True)
					done.record()
			except torch.cuda.OutOfMemoryError as exc:
				if S == 1:
					raise

				failed = str(exc).strip().split("\n")[0]

			if failed is not None:
				if pending is not None:
					finish(*pending)
					pending = None

				torch.cuda.synchronize(self.device)
				gc.collect()
				torch.cuda.empty_cache()
				logger.warning("out of GPU memory with %d sequences per step; "
					"retrying with %d (%s)", S, S // 2, failed)
				S //= 2
				starts = collections.deque(range(a, N, S))
				compiles = True
				continue

			# A call that compiles, the first at a step size, leaves Dynamo's
			# tracing state in reference cycles that hold a whole step's
			# activations. Only the cyclic garbage collector frees them.
			if compiles:
				gc.collect()
				compiles = False

			if pending is not None:
				finish(*pending)

			pending = (a, b, out, done)

		if pending is not None:
			finish(*pending)
