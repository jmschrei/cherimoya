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


logger = logging.getLogger(__name__)

FCONV_PADDING = 37

CPU_OUTPUTS_PER_CHUNK = 16
MAX_OUTPUTS_PER_CHUNK = 128


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


def backward_pass(G0, terms, stem, blocks, rs):
	"""The backward over K pairs and B output positions, from the final
	residual stream to iconv's output: G0 (B, L, C), the gradient of each
	output position, which every pair shares, -> (K * B, C, L), pair-major.

	The operations are those of
	`cherimoya._deep_lift_shap_engine.backward_pass`, with each rule's terms
	read from `rule_terms` rather than computed from the activations. On
	CUDA this function runs compiled.
	"""

	K = stem.shape[0]
	G = G0.expand(K, *G0.shape)
	for t, block in zip(reversed(terms), reversed(blocks)):
		Gu = torch.matmul(G * rs, block.W2) * t.slope
		Gy = _layer_normalization_grad(torch.matmul(Gu, block.W1), t.v_avg,
			t.a_sum, t.variance_term, 2)
		G = G + dw3(Gy, block.cw.flip(0), block.d)

	Gz = G.transpose(-1, -2) * stem
	return Gz.reshape(-1, *Gz.shape[-2:]).contiguous()


def epilogue(m, x, r, y_x, y_r):
	"""Each pair's convergence delta at each output position, and each output
	position's hypothetical attributions averaged over the K references.

	m (K * B, 4, L) holds the multipliers, x (1, 4, L) the sequence's
	one-hot and r (K, 4, L) its references', and y_x (B,) and y_r (K, B) the
	attributed outputs. The delta is `pisa`'s, |(y_x - y_r) - sum((x - r) *
	m)|, and each pair's attributions are tangermeme's
	`hypothetical_attributions`. Returns the attributions (B, 4, L) and the
	deltas (K, B). On CUDA this function runs compiled.
	"""

	m = m.view(*y_r.shape, *m.shape[1:])
	x, r = x[:, None].expand_as(m), r[:, None].expand_as(m)
	delta = ((y_x - y_r) - ((x - r) * m).sum(dim=(-2, -1))).abs()
	hyp, = hypothetical_attributions((m.flatten(0, 1),), (x.flatten(0, 1),),
		(r.flatten(0, 1),))
	return hyp.view(m.shape).mean(dim=0), delta


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
	final residual stream, built padded (2 (L x C))."""

	per_row = itemsize * L * (3 * C + 2 * H) + 4 * itemsize * 4 * L
	return K * per_row + 2 * itemsize * L * C


class Engine:
	"""PISA attributions of a Cherimoya model's profile logits for one
	channel: what ``pisa(model, X, hypothetical=True, ...)`` computes for a
	module that returns ``ControlWrapper(model)(X)[0][:, channel]``.

	The engine works one sequence at a time. The sequence and its K
	references are forwarded together, the rule terms of the K pairs are
	computed, and the backward runs over the pairs for chunks of B output
	positions. The hypothetical attributions are averaged over the
	references on the device. A last chunk with fewer than B positions is
	padded with zero gradients, so that every chunk has the same shapes, and
	the padded positions are dropped.

	On the CPU, B is at most 16. On CUDA:

	* B is at most the largest that `sequence_bytes` and `output_bytes`
	  estimate fits both `MEM_BUDGET` and `FREE_MEMORY_FRACTION` of the
	  device memory the process can use, from 1 to 128. In either case the
	  output positions are then split into chunks of equal size, B or one
	  fewer, before the last chunk is padded.
	* The forward, the rule terms, the backward and the epilogue run under
	  torch.compile, fullgraph and with static shapes, as in
	  `cherimoya._deep_lift_shap_engine.Engine`, and a gc.collect() follows
	  every sequence that compiled.
	* Sequences are pipelined. A sequence's results are copied back into
	  pinned memory without waiting; the host then waits for the previous
	  sequence, once, on its event, and draws the next sequence's
	  references while the device runs this one.
	* An out-of-memory error empties the allocator's cache, halves B and
	  runs the sequence again. At B = 1 the error propagates.


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
		self.rs = float(model.residual_scale)
		self.T = int(model.trimming)
		self.C, self.H = model.n_filters, model.expansion * model.n_filters
		self.n_ctl = model.n_control_tracks
		self.channel = channel

		self.n_shuffles, self.random_state = n_shuffles, random_state
		self.warning_threshold = warning_threshold
		self.outputs_per_chunk = outputs_per_chunk

		passes = (forward_save, rule_terms, backward_pass, epilogue)
		if self.device.type == "cuda":
			kwargs = {"fullgraph": True, "dynamic": False, "mode": "default"}
			parameters = inspect.signature(torch.compile).parameters
			if {"recompile_limit", "isolate_recompiles"} <= set(parameters):
				kwargs.update(recompile_limit=RECOMPILE_LIMIT,
					isolate_recompiles=True)

			passes = tuple(torch.compile(fn, **kwargs) for fn in passes)

		self.forward, self.rule_terms, self.backward, self.epilogue = passes

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
			logger.info("at most %d outputs per chunk (%.2f GB each, %.2f GB "
				"per sequence, %.1f GB available)", B, per_output / 1e9,
				fixed / 1e9, available / 1e9)
		else:
			B = CPU_OUTPUTS_PER_CHUNK

		return math.ceil(n_outputs / math.ceil(n_outputs / min(B, n_outputs)))

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

	def _seeds(self, a, B, n_outputs, L):
		"""The gradient of the output positions [a, a + B) at the final
		residual stream, (B, L, C): the profile convolution's taps for the
		attributed channel, over the positions each output reads. Positions
		from n_outputs on get zeros."""

		taps = self.Wf[self.channel, :self.C].T
		width = taps.shape[0]
		G = torch.zeros(B, L + width - 1, self.C, dtype=taps.dtype,
			device=self.device)
		rows = torch.arange(min(B, n_outputs - a), device=self.device)[:, None]
		G[rows, self.T + a + rows + torch.arange(width,
			device=self.device)] = taps
		return G[:, FCONV_PADDING:FCONV_PADDING + L].contiguous()

	def _sequence(self, inputs, K, B, n_outputs):
		"""One sequence on the engine's device: its rows and its references',
		(K + 1, 4, L), an int8 one-hot, -> (y_x, y_r, deltas, attr), on the
		device. Each large intermediate is released as soon as the next stage
		no longer needs it."""

		X1h = inputs.to(self.W0.dtype)
		L = X1h.shape[-1]
		z0 = F.conv1d(X1h, self.W0, self.b0, padding=ICONV_PADDING)
		hL, saved = self.forward(z0, self.blocks, self.rs, CONV_NORM_EPS)
		y = self._profile(hL)
		del hL
		terms, stem = self.rule_terms(saved, z0)
		del saved, z0

		n_chunks = math.ceil(n_outputs / B)
		y = F.pad(y, (0, n_chunks * B - n_outputs))
		attr = torch.empty(n_chunks * B, 4, L, dtype=y.dtype, device=y.device)
		deltas = torch.empty(K, n_chunks * B, dtype=y.dtype, device=y.device)
		for a in range(0, n_outputs, B):
			Gz = self.backward(self._seeds(a, B, n_outputs, L), terms, stem,
				self.blocks, self.rs)

			# iconv's input gradient, the call autograd makes for F.conv1d
			# when only the input needs a gradient.
			m = torch.nn.grad.conv1d_input((K * B, 4, L), self.W0, Gz,
				padding=ICONV_PADDING)
			del Gz
			attr[a:a + B], deltas[:, a:a + B] = self.epilogue(m, X1h[:1],
				X1h[1:], y[0, a:a + B], y[1:, a:a + B])

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
			# garbage collector frees them.
			if compiles:
				gc.collect()
				compiles = False

			if pending is not None:
				finish(*pending)

			pending = (i, out, done)
			i += 1

		if pending is not None:
			finish(*pending)
