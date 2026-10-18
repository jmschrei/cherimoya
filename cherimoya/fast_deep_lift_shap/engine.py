# engine.py
# Author: Eugenio Mattei

"""
A fast DeepLIFT/DeepSHAP engine for the count head of a Cherimoya model.

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
  functions. The references are tangermeme's `dinucleotide_shuffle`, called with
  the seeds `deep_lift_shap` uses (`draw_references`).

Only the count head (`LogCountWrapper`) is implemented. The profile head
goes through a softmax rule that couples every position, and keeps the
default engine.

On a CUDA device the forward, the backward and the epilogue run under
`torch.compile`; iconv, the count head and iconv's input gradient run
eagerly, with exactly the operations the default engine runs. Steps are
pipelined: the host waits once per step, on the previous step's event.

Rows are laid out as ``[x_0, ..., x_{S-1}, r_00, ..., r_{S-1,K-1}]``: S
sequences, then the K references of each. `split_rows` views them as
sequences (S, 1, ...) and references (S, K, ...), so that a sequence row
broadcasts over its K pairs without being copied.

Deliberate differences from the default engine:

* The count head casts the pooled stream to the count Linear's dtype, where
  `Cherimoya` calls ``.float()``. Every model this engine accepts on a GPU
  has float32 parameters, so the two are the same operation; a float64 model
  on the CPU, which the tests use for their high-precision comparisons,
  keeps float64.
* The engine returns its results in the model's dtype and warns once per
  run, with a summary of the convergence deltas, where tangermeme warns once
  per batch. A `warning_threshold` of None means no warning.
"""

import collections
import contextlib
import gc
import inspect
import logging
import numbers
import operator
import os
import time
import warnings

from typing import NamedTuple
from dataclasses import dataclass

import numpy
import torch
import torch.nn.functional as F

from tangermeme._deep_lift_utils import _layer_normalization_grad
from tangermeme._deep_lift_utils import _layer_normalization_stats
from tangermeme._deep_lift_utils import _layer_normalization_terms
from tangermeme._deep_lift_utils import _rescale
from tangermeme.deep_lift_shap import hypothetical_attributions
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.utils import validate_input

from ..cheri import CONV_NORM_EPS
from ..cheri import _cheri_conv
from ..cheri import _cheri_conv_norm_cpu
from ..cherimoya import Cherimoya
from ..wrappers import ControlWrapper
from ..wrappers import LogCountWrapper
from ..wrappers import ProfileWrapper


logger = logging.getLogger(__name__)

VALIDATED_TANGERMEME_VERSIONS = ("1.5.1",)
"""The tangermeme releases whose DeepLIFT rules, hypothetical projection and
dinucleotide shuffle this engine was checked against. Another version runs,
with a warning."""

FORWARD_STATS = ("two_pass", "one_pass", "cpu_mirror")
OUTPUTS = ("counts", "profile")
PRECISIONS = ("tf32", "fp32")
COMPILE_MODES = ("auto", "default", "none")

GB = 10**9
"""`mem_budget_gb` counts gigabytes of 1e9 bytes."""

MAX_SEQS_PER_STEP = 32

FREE_MEMORY_FRACTION = 0.6
"""Automatic step sizing plans for at most this fraction of the device memory
available to the engine."""

MEMORY_CHECK_FACTOR = 1.3
"""A run whose steps allocate more than this times `step_bytes` logs a
warning."""

RECOMPILE_LIMIT = 8
"""Compiles allowed per compiled pass of one engine: one per step size, and
the step size halves at most 5 times from 32."""

DETERMINISTIC_CUBLAS = (":4096:8", ":16:8")
"""CUBLAS_WORKSPACE_CONFIG values under which cuBLAS is deterministic; the
engine sets the first."""

PRECISION_ENV = (
	"CUBLAS_WORKSPACE_CONFIG",
	"NVIDIA_TF32_OVERRIDE",
	"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
	"TORCHINDUCTOR_CACHE_DIR",
	"TRITON_CACHE_DIR",
)
"""Environment variables `precision_flags` records: they change the numerics
or where compiled code is cached."""

ICONV_KERNEL, ICONV_PADDING = 21, 10
FCONV_KERNEL, FCONV_PADDING = 75, 37

AUTO_SEQS_PER_STEP = 8
"""Sequences per step for `seqs_per_step="auto"` on the CPU. On CUDA "auto"
sizes steps from the memory the process can use."""

SEED_RANGE = 9_999_999
"""`random_state=None` draws a base seed from [0, SEED_RANGE), once per
Engine, as tangermeme draws one."""


def check_tangermeme_version():
	"""Warn if the installed tangermeme is not a validated release.

	The engine reproduces tangermeme's DeepLIFT rules, its hypothetical
	projection and the seeds its `deep_lift_shap` gives
	`dinucleotide_shuffle`, so a release that changes any of them makes the
	engine compute something else. `VALIDATED_TANGERMEME_VERSIONS` lists the
	releases it was checked against. The per-run audit (`audit`) compares
	the engine with the installed tangermeme whatever the version.


	Returns
	-------
	version: str
		The installed tangermeme version.
	"""

	import tangermeme

	version = tangermeme.__version__
	if version not in VALIDATED_TANGERMEME_VERSIONS:
		warnings.warn("cherimoya's fast DeepLIFT/SHAP engine reproduces "
			"tangermeme's DeepLIFT rules and was validated against "
			"tangermeme {}; tangermeme {} is installed. Its results are "
			"unchecked against this version; the audit compares them with "
			"this version's deep_lift_shap.".format(
			", ".join(VALIDATED_TANGERMEME_VERSIONS), version), UserWarning,
			stacklevel=3)

	return version


class BlockWeights(NamedTuple):
	"""One CheriBlock's tensors."""

	cw: torch.Tensor  # (3, C) depthwise taps; row 0 reads l - d, row 2 l + d
	W1: torch.Tensor  # (H, C) linear1, no bias
	W2: torch.Tensor  # (C, H) linear2, no bias
	d: int  # dilation, 2**i


@dataclass(frozen=True, eq=False)
class CheriWeights:
	"""The tensors of a Cherimoya model that the forward and backward read,
	detached, on one device.

	Build it with `CheriWeights.from_model`, which checks the architecture
	the hand-written passes implement.
	"""

	W0: torch.Tensor  # (C, 4, 21) iconv weight
	b0: torch.Tensor  # (C,) iconv bias
	blocks: tuple
	rs: float  # residual_scale, shared by every block
	Wf: torch.Tensor  # (n_out, C + n_ctl, 75) fconv weight
	bf: torch.Tensor  # (n_out,) fconv bias
	Wl: torch.Tensor  # (n_groups, C + (n_ctl > 0)) count Linear weight
	bl: torch.Tensor  # (n_groups,) count Linear bias
	T: int  # trimming: both heads read positions [T, L - T)
	eps: float  # CONV_NORM_EPS
	signal_groups: tuple
	n_ctl: int  # n_control_tracks

	@classmethod
	def from_model(cls, model, device=None):
		"""Read a model's weights after checking the architecture.

		The tensors are detached and copied to `device`, so changing the
		model afterwards changes nothing here. The parameters must all be
		float32; float64 is accepted on the CPU, for tests.


		Parameters
		----------
		model: cherimoya.Cherimoya
			The model.

		device: str or torch.device or None, optional
			Where to put the weights. None means the model's own device.
			Default is None.


		Returns
		-------
		weights: CheriWeights
			The weights.


		Raises
		------
		ValueError
			If the model breaks the architecture the engine implements; the
			message lists every violation (`contract_violations`).
		"""

		problems = contract_violations(model)
		if problems:
			raise ValueError("the model breaks the contract of the fast "
				"DeepLIFT/SHAP engine:\n  " + "\n  ".join(problems))

		if device is not None:
			dev = torch.device(device)
		else:
			dev = model.iconv.weight.device

		if model.iconv.weight.dtype == torch.float64 and dev.type != "cpu":
			raise ValueError("float64 models run on the CPU only (tests), "
				"not on {}".format(dev))

		def get(t):
			return t.detach().to(device=dev, copy=True)

		blocks = tuple(BlockWeights(get(b.conv.conv_weight),
			get(b.linear1.weight), get(b.linear2.weight), int(b.dilation))
			for b in model.blocks)

		return cls(
			W0=get(model.iconv.weight),
			b0=get(model.iconv.bias),
			blocks=blocks,
			rs=float(model.residual_scale),
			Wf=get(model.fconv.weight),
			bf=get(model.fconv.bias),
			Wl=get(model.linear.weight),
			bl=get(model.linear.bias),
			T=int(model.trimming),
			eps=float(CONV_NORM_EPS),
			signal_groups=tuple(int(g) for g in model.signal_groups),
			n_ctl=int(model.n_control_tracks),
		)

	@property
	def C(self):
		return self.W0.shape[0]

	@property
	def H(self):
		return self.blocks[0].W1.shape[0] if self.blocks else 0

	@property
	def n_out(self):
		return self.Wf.shape[0]

	@property
	def n_groups(self):
		return self.Wl.shape[0]

	@property
	def dtype(self):
		return self.W0.dtype

	@property
	def device(self):
		return self.W0.device


def contract_violations(model):
	"""Every way a model departs from the architecture the hand-written
	passes implement.


	Parameters
	----------
	model: object
		The model to check.


	Returns
	-------
	problems: list of str
		One line per violation; empty if there is none.
	"""

	if not isinstance(model, Cherimoya):
		return ["expected a cherimoya.Cherimoya, got {}".format(
			type(model).__name__)]

	problems = []
	C, n_layers, n_ctl = model.n_filters, model.n_layers, model.n_control_tracks
	H = model.expansion * C
	groups = list(model.signal_groups)
	n_out, n_groups = sum(groups), len(groups)

	gelus = [("igelu", model.igelu)] + [("blocks.{}.activation".format(i),
		b.activation) for i, b in enumerate(model.blocks)]
	for name, module in gelus:
		if not isinstance(module, torch.nn.GELU) or module.approximate != "tanh":
			problems.append("{} must be GELU(approximate='tanh'), got "
				"{!r}".format(name, module))

	dtypes = sorted({str(p.dtype) for p in model.parameters()})
	if dtypes not in (["torch.float32"], ["torch.float64"]):
		problems.append("parameters must all be float32 (float64 on the CPU "
			"for tests), got {}".format(dtypes))

	problems += _conv_violations("iconv", model.iconv, in_channels=4,
		out_channels=C, kernel=ICONV_KERNEL, padding=ICONV_PADDING)
	problems += _conv_violations("fconv", model.fconv, in_channels=C + n_ctl,
		out_channels=n_out, kernel=FCONV_KERNEL, padding=FCONV_PADDING)

	if len(model.blocks) != n_layers:
		problems.append("{} blocks for n_layers={}".format(len(model.blocks),
			n_layers))

	for i, block in enumerate(model.blocks):
		name = "blocks.{}".format(i)
		if block.dilation != 2**i or block.conv.dilation != 2**i:
			problems.append("{} dilation {}/{}, expected {}".format(name,
				block.dilation, block.conv.dilation, 2**i))

		if block.residual_scale != model.residual_scale:
			problems.append("{} residual_scale {} != model's {}".format(name,
				block.residual_scale, model.residual_scale))

		problems += _shape_violations(name + ".conv.conv_weight",
			block.conv.conv_weight, (3, C))
		for layer, shape in (("linear1", (H, C)), ("linear2", (C, H))):
			linear = getattr(block, layer)
			if not isinstance(linear, torch.nn.Linear) or linear.bias is not None:
				problems.append("{}.{} must be an nn.Linear without bias, "
					"got {!r}".format(name, layer, linear))
			else:
				problems += _shape_violations("{}.{}.weight".format(name,
					layer), linear.weight, shape)

	if not isinstance(model.linear, torch.nn.Linear) or model.linear.bias is None:
		problems.append("linear must be an nn.Linear with bias, got "
			"{!r}".format(model.linear))
	else:
		problems += _shape_violations("linear.weight", model.linear.weight,
			(n_groups, C + (n_ctl > 0)))
		problems += _shape_violations("linear.bias", model.linear.bias,
			(n_groups,))

	if not isinstance(model.trimming, int) or model.trimming < 0:
		problems.append("trimming must be a non-negative int, got "
			"{!r}".format(model.trimming))

	return problems


def _shape_violations(name, tensor, shape):
	if tuple(tensor.shape) == shape:
		return []

	return ["{} has shape {}, expected {}".format(name, tuple(tensor.shape),
		shape)]


def _conv_violations(name, conv, in_channels, out_channels, kernel, padding):
	"""The Conv1d geometry the engine hard-codes: F.conv1d(x, W, b,
	padding=padding) with stride and dilation 1."""

	if not isinstance(conv, torch.nn.Conv1d):
		return ["{} must be an nn.Conv1d, got {}".format(name,
			type(conv).__name__)]

	want = {
		"kernel_size": (kernel,),
		"padding": (padding,),
		"stride": (1,),
		"dilation": (1,),
		"groups": 1,
		"padding_mode": "zeros",
	}

	problems = ["{}.{} is {!r}, expected {!r}".format(name, key,
		getattr(conv, key), value) for key, value in want.items()
		if getattr(conv, key) != value]

	problems += _shape_violations(name + ".weight", conv.weight,
		(out_channels, in_channels, kernel))
	if conv.bias is None:
		problems.append("{} must have a bias".format(name))
	else:
		problems += _shape_violations(name + ".bias", conv.bias,
			(out_channels,))

	return problems


def make_wrapper(model, output="counts", group=0):
	"""The wrapper `cherimoya attribute` attributes, built from cherimoya's
	own classes, so that a bad group fails exactly as there.


	Parameters
	----------
	model: cherimoya.Cherimoya
		The model.

	output: str, optional
		"counts" for `LogCountWrapper`, "profile" for `ProfileWrapper`.
		Default is "counts".

	group: int or None, optional
		The signal group to attribute. A count head with several groups
		needs one, since DeepLIFT/SHAP attributes one output. Default is 0.


	Returns
	-------
	wrapper: torch.nn.Module
		The output wrapper over `ControlWrapper(model)`.
	"""

	if output not in OUTPUTS:
		raise ValueError("output must be either `counts` or `profile`.")

	if output == "counts" and group is None and len(model.signal_groups) > 1:
		raise ValueError("deep_lift_shap attributes one output; set `group` "
			"to one of the model's {} signal groups".format(
			len(model.signal_groups)))

	inner = ControlWrapper(model)
	if output == "counts":
		return LogCountWrapper(inner, group=group)

	return ProfileWrapper(inner, group=group)


def resolve_forward_stats(forward_stats, device):
	"""Resolve "auto": "two_pass" on CUDA and "cpu_mirror" on the CPU.

	"cpu_mirror" reproduces the model's CPU path and is refused on CUDA,
	where the model runs its Triton kernels instead.
	"""

	cuda = torch.device(device).type == "cuda"
	if forward_stats == "auto":
		return "two_pass" if cuda else "cpu_mirror"

	if forward_stats not in FORWARD_STATS:
		raise ValueError("forward_stats must be 'auto' or one of {}, got "
			"{!r}".format(FORWARD_STATS, forward_stats))

	if forward_stats == "cpu_mirror" and cuda:
		raise ValueError("forward_stats='cpu_mirror' mirrors the model's CPU "
			"path and runs on the CPU only")

	return forward_stats


def resolve_compile(compile, device):
	"""Resolve "auto": "default" (torch.compile's default mode) on CUDA and
	"none" (eager) on the CPU. The compiled passes are validated on CUDA
	only, so "default" is refused on the CPU."""

	if compile not in COMPILE_MODES:
		raise ValueError("compile must be one of {}, got {!r}".format(
			COMPILE_MODES, compile))

	cuda = torch.device(device).type == "cuda"
	if compile == "auto":
		return "default" if cuda else "none"

	if compile == "default" and not cuda:
		raise ValueError("compile='default' compiles the CUDA passes; on the "
			"CPU use compile='none'")

	return compile


class BlockSaved(NamedTuple):
	"""What one block's backward rules read. Nothing else is kept: the
	normalized stream and the GELU's output are recomputed."""

	y: torch.Tensor  # (R, L, C) depthwise conv output, the norm's input
	u: torch.Tensor  # (R, L, H) linear1 output, the GELU's input
	mu: torch.Tensor  # (R,) the rule's mean of y over the (L, C) plane
	v: torch.Tensor  # (R,) the rule's inverse std, (var + eps) ** -0.5


class Forward(NamedTuple):
	"""`model_forward`'s results for a batch of R rows."""

	y: torch.Tensor  # (R,) the attributed output, one scalar per row
	z0: torch.Tensor  # (R, C, L) iconv output, the stem GELU's input
	hL: torch.Tensor  # (R, L, C) the final residual stream, channels last
	saved: list


def stem_forward(X1h, W0, b0):
	"""iconv, the operation nn.Conv1d(4, C, 21, padding=10) calls: X1h
	(R, 4, L) -> z0 (R, C, L)."""

	return F.conv1d(X1h, W0, b0, padding=ICONV_PADDING)


def dw3(h, cw, d):
	"""The 3-tap dilated depthwise convolution as a shift-add.

	h has shape (..., L, C), channels last, and cw has shape (3, C). The
	output is ``y[l] = cw[0] * h[l - d] + cw[1] * h[l] + cw[2] * h[l + d]``,
	with zeros outside [0, L).
	"""

	L = h.shape[-2]
	hp = F.pad(h, (0, 0, d, d))
	return (hp[..., :L, :] * cw[0] + hp[..., d:d + L, :] * cw[1]
		+ hp[..., 2 * d:2 * d + L, :] * cw[2])


def norm_two_pass(y, eps):
	"""The forward's normalization over each row's (L, C) plane with the
	rule's own statistics, tangermeme's `_layer_normalization_stats`: (yhat,
	mu, v), with yhat = (y - mu) * v and the reduced axes kept.

	The model's Triton forward computes the variance as E[y^2] - mean^2
	instead; on zero-mean input the two agree to about 1e-6.
	"""

	mu, v = _layer_normalization_stats(y, 2, eps)
	return (y - mu) * v, mu, v


def _one_pass_norm(y, eps):
	"""The Triton forward's formula: var = E[y^2] - mean^2, rstd =
	1 / sqrt(var + eps)."""

	D = y.shape[-2] * y.shape[-1]
	mean = y.sum(dim=(-2, -1), keepdim=True) / D
	var = (y * y).sum(dim=(-2, -1), keepdim=True) / D - mean * mean
	return (y - mean) * (1.0 / torch.sqrt(var + eps))


def forward_save(z0, blocks, rs, eps, stats):
	"""Run the stem GELU and the blocks on z0 (R, C, L), keeping what the
	backward rules read.

	The function is row-wise: a batch can hold sequences and references in
	any order. `stats` picks the normalization of the forward:

	* "cpu_mirror": y from cherimoya's own `_cheri_conv` and the model's CPU
	  normalization, so the forward is the model's CPU path operation for
	  operation. CPU only.
	* "two_pass": y from the shift-add `dw3`; the forward normalizes with the
	  rule's own statistics (`norm_two_pass`).
	* "one_pass": y from `dw3`, normalized with the Triton forward's formula.
	  Diagnostic only.

	In every mode the saved mu and v are the rule's statistics, in the
	model's dtype. On CUDA this function runs compiled; nothing in it may
	break the graph.


	Returns
	-------
	hL: torch.Tensor, shape=(R, L, C)
		The final residual stream, channels last.

	saved: list of BlockSaved
		One per block.
	"""

	if stats not in FORWARD_STATS:
		raise ValueError("stats must be one of {}, got {!r}".format(
			FORWARD_STATS, stats))

	if stats == "cpu_mirror" and z0.is_cuda:
		raise ValueError("forward_stats='cpu_mirror' mirrors the model's CPU "
			"path and runs on the CPU only")

	h = F.gelu(z0, approximate="tanh").transpose(1, 2).contiguous()
	saved = []
	for cw, W1, W2, d in blocks:
		R = h.shape[0]
		y = _cheri_conv(h, cw, d) if stats == "cpu_mirror" else dw3(h, cw, d)
		if stats == "two_pass":
			yhat, mu, v = norm_two_pass(y, eps)
		else:
			mu, v = _layer_normalization_stats(y, 2, eps)
			if stats == "cpu_mirror":
				# The model's CPU path, which computes the convolution again.
				yhat = _cheri_conv_norm_cpu(h, cw, d, eps)
			else:
				yhat = _one_pass_norm(y, eps)

		# The block's residual, in the model's operation order.
		u = F.linear(yhat, W1)
		h = h + F.linear(F.gelu(u, approximate="tanh"), W2) * rs
		saved.append(BlockSaved(y, u, mu.view(R), v.view(R)))

	return h, saved


def head_forward(hL, w, output="counts", group=0):
	"""The attributed scalar per row, from the final residual stream hL
	(R, L, C).

	The count head runs the model's operations on the model's (R, C, L)
	layout, then `LogCountWrapper`'s group slice and `deep_lift_shap`'s
	target 0. Control tracks are zeros, as `ControlWrapper` feeds them, so
	their count feature is log(0 + 1) = 0. The pooled stream is cast to the
	count Linear's dtype, where the model calls ``.float()``: the same
	operation for float32 parameters, and float64 stays float64.
	"""

	if output == "profile":
		raise NotImplementedError("the fast engine attributes the count head "
			"only; attribute the profile head with tangermeme's "
			"deep_lift_shap")

	if output != "counts":
		raise ValueError("output must be either `counts` or `profile`.")

	R, L, _ = hL.shape
	T = w.T
	_check_head(w, group, L)

	H = hL.transpose(1, 2).contiguous()
	pooled = torch.mean(H[:, :, T:L - T].to(w.Wl.dtype), dim=2)
	if w.n_ctl:
		ctl = H.new_zeros(R, w.n_ctl, L)
		ctl = torch.sum(ctl[:, :, T:L - T].to(w.Wl.dtype),
			dim=(1, 2)).unsqueeze(-1)
		pooled = torch.cat([pooled, torch.log(ctl + 1)], dim=-1)

	y = F.linear(pooled, w.Wl, w.bl)
	if group is not None:
		y = y[:, group:group + 1]

	return y[:, 0]


def model_forward(X1h, w, forward_stats="auto", output="counts", group=0):
	"""The whole forward on a batch of one-hot rows X1h (R, 4, L), cast to
	the weights' dtype and device. Run it under torch.no_grad() unless a
	graph is wanted; the engine never needs one for its forward.


	Returns
	-------
	forward: Forward
		The attributed outputs, iconv's output, the final residual stream and
		the saved state of every block.
	"""

	stats = resolve_forward_stats(forward_stats, w.device)
	X1h = X1h.to(device=w.device, dtype=w.dtype)
	z0 = stem_forward(X1h, w.W0, w.b0)
	hL, saved = forward_save(z0, w.blocks, w.rs, w.eps, stats)
	y = head_forward(hL, w, output=output, group=group)
	return Forward(y, z0, hL, saved)


def _check_head(w, group, length):
	"""Raise a ValueError for a group the count head lacks, or an input
	length that trimming leaves empty."""

	if group is not None and (isinstance(group, bool)
			or not isinstance(group, int) or not 0 <= group < w.n_groups):
		raise ValueError("group must be None or an int in [0, {}), got "
			"{!r}".format(w.n_groups, group))

	if length <= 2 * w.T:
		raise ValueError("input length {} leaves no positions after trimming "
			"{} from each end".format(length, w.T))


def counts_cotangent(w, group=0, length=None):
	"""The gradient of the attributed count with respect to the final
	residual stream, (L, C), which every row shares.

	The count head averages the stream over positions [T, L - T) and applies
	the count Linear; the control tracks' feature is a constant. Its
	gradient is therefore the attributed group's row of the count Linear,
	divided by the number of positions averaged, at those positions, and
	zero elsewhere. Compute it once per input length and broadcast it over
	the pairs.
	"""

	if length is None:
		raise ValueError("counts_cotangent needs the input length")

	_check_head(w, group, length)
	T = w.T
	G0 = torch.zeros(length, w.C, device=w.device, dtype=w.dtype)
	G0[T:length - T] = w.Wl[0 if group is None else group, :w.C] / (
		length - 2 * T)
	return G0


def draw_references(X, n_shuffles, random_state):
	"""The references `deep_lift_shap` draws for the sequences X (n, 4, L):
	reference k of sequence e is ``dinucleotide_shuffle(X[e:e+1], n=1,
	random_state=random_state + k)[:, 0]``.

	One call to tangermeme's `dinucleotide_shuffle`, with a seed per row,
	draws them all. Returns (n, n_shuffles, 4, L) in X's dtype.
	"""

	n = len(X)
	rows = X.repeat_interleave(n_shuffles, dim=0)
	seeds = random_state + numpy.tile(numpy.arange(n_shuffles), n)
	references = dinucleotide_shuffle(rows, n=1, random_state=seeds)[:, 0]
	return references.view(n, n_shuffles, *X.shape[1:])


def split_rows(t, S, K):
	"""View rows ``[x_0..x_{S-1}, r_00..r_{S-1,K-1}]`` as sequences
	(S, 1, ...) and references (S, K, ...).

	Both are views, so `t` must be contiguous along its rows. A sequence row
	broadcasts over its K pairs.
	"""

	if t.shape[0] != S + S * K:
		raise ValueError("expected S + S*K = {} rows for S={}, K={}, got "
			"{}".format(S + S * K, S, K, t.shape[0]))

	return t[:S].unsqueeze(1), t[S:].view(S, K, *t.shape[1:])


def gelu_rescale(G, z, S, K):
	"""The rescale rule through GELU(tanh) on the sequence half: tangermeme's
	`_rescale`, as its `_nonlinear` applies it.

	G (S, K, ...) is each pair's gradient at the GELU's output, and z
	(S + S*K, ...) the GELU's input over the joint rows. The result,
	(S, K, ...), is the gradient at the GELU's input: G times the secant,
	or, where the input barely moves, the plain gradient that nn.GELU's
	autograd hands tangermeme, gelu_backward(G, z_x).

	The GELU's outputs are recomputed from z over all rows at once, as the
	forward computed them, so the secants use the forward's own values.
	"""

	zx, zr = split_rows(z, S, K)
	gx, gr = split_rows(F.gelu(z, approximate="tanh"), S, K)
	gradient = torch.ops.aten.gelu_backward(G, zx.expand_as(G),
		approximate="tanh")
	return _rescale(zx - zr, gx - gr, gradient, G)


def ln_rule(Gn, y, mu, v, S, K):
	"""The normalization rule of `conv_norm_op` on the sequence half.

	`cherimoya.deep_lift_shap.conv_norm_op` applies tangermeme's
	`_layer_normalization_helper` to the depthwise convolution's output: the
	whole (L, C) plane of each row, with no affine weight. This applies the
	helper's tensor functions, `_layer_normalization_terms` and
	`_layer_normalization_grad`, to each sequence row broadcast against its
	K reference rows.

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


def dw3_t(G, cw, d):
	"""The transpose of `dw3`, i.e. the depthwise convolution's input
	gradient: `dw3` with the taps reversed, ``g[l] = cw[2] * G[l - d] +
	cw[1] * G[l] + cw[0] * G[l + d]``."""

	return dw3(G, cw.flip(0), d)


def bwd_block(G, saved, block, rs, S, K):
	"""One CheriBlock's DeepLIFT backward on the sequence half: G
	(S, K, L, C) at its output -> at its input.

	The block computes ``h + linear2(GELU(linear1(conv_norm(h)))) * rs``.
	Backwards: the ordinary gradient through the scale and linear2, the
	rescale rule at the GELU, the ordinary gradient through linear1,
	`conv_norm_op`'s normalization rule pushed back through the depthwise
	convolution, and the residual. G may be a broadcast view, such as the
	count gradient expanded over the pairs.
	"""

	Gu = gelu_rescale(torch.matmul(G * rs, block.W2), saved.u, S, K)
	Gy = ln_rule(torch.matmul(Gu, block.W1), saved.y, saved.mu, saved.v, S, K)
	return G + dw3_t(Gy, block.cw, block.d)


def stem_rule(G, z0, S, K):
	"""The stem GELU's rescale rule on the sequence half: G (S, K, L, C) at
	the stem's output -> (S*K, C, L) at iconv's output.

	G is transposed back to the GELU's (C, L) layout. z0 (S + S*K, C, L) is
	iconv's output over the joint rows. The result is contiguous, ready for
	`input_multipliers`.
	"""

	Gz = gelu_rescale(G.transpose(-1, -2), z0, S, K)
	return Gz.reshape(S * K, *Gz.shape[-2:]).contiguous()


def input_multipliers(Gz, W0):
	"""iconv's input gradient: the multipliers m (S*K, 4, L) at the one-hot
	input, from the gradient Gz (S*K, C, L) at iconv's output.

	`torch.nn.grad.conv1d_input` makes the call autograd makes for
	``F.conv1d(X, W0, b0, padding=10)`` when only the input needs a
	gradient.
	"""

	shape = (Gz.shape[0], W0.shape[1], Gz.shape[-1])
	return torch.nn.grad.conv1d_input(shape, W0, Gz, padding=ICONV_PADDING)


def backward_pass(G0, saved, z0, blocks, rs, S, K):
	"""The backward over the S*K pairs, from the final residual stream to
	iconv's output.

	G0 (L, C) is the gradient every pair shares at the final residual stream
	(`counts_cotangent`). It enters broadcast over the pairs, goes through
	every block in reverse (`bwd_block`) and through the stem GELU
	(`stem_rule`). Returns Gz (S*K, C, L), ready for `input_multipliers`.
	On CUDA this function runs compiled.
	"""

	L, C = G0.shape
	G = G0.expand(S, K, L, C)
	for s, block in zip(reversed(saved), reversed(blocks)):
		G = bwd_block(G, s, block, rs, S, K)

	return stem_rule(G, z0, S, K)


def convergence_deltas(m, X1h, y_x, y_r, S, K):
	"""tangermeme's convergence delta per pair, |(y_x - y_r) - sum((x - r) *
	m)| over the full length.

	m (S*K, 4, L) holds the multipliers, X1h (S + S*K, 4, L) the joint
	one-hot rows, y_x (S,) and y_r (S, K) the attributed outputs. Returns
	(S, K), with `deep_lift_shap`'s expression.
	"""

	x, r = split_rows(X1h, S, K)
	m = m.view(S, K, *m.shape[1:])
	return ((y_x[:, None] - y_r) - ((x - r) * m).sum(dim=(-2, -1))).abs()


def epilogue(m, X1h, y_x, y_r, start, end, S, K):
	"""The convergence deltas over the full length, and the hypothetical
	attributions over the window [start, end), averaged over each
	sequence's K references.

	m (S*K, 4, L) holds the multipliers, X1h (S + S*K, 4, L) the joint
	one-hot rows, and y_x (S,) and y_r (S, K) the attributed outputs. The
	attributions of each pair are tangermeme's `hypothetical_attributions`
	of its multipliers in the window.


	Returns
	-------
	attr: torch.Tensor, shape=(S, 4, end - start)
		Each sequence's mean over its references.

	delta: torch.Tensor, shape=(S, K)
		`convergence_deltas` of the multipliers.
	"""

	L = X1h.shape[-1]
	if not 0 <= start < end <= L:
		raise ValueError("the window [{}, {}) must be non-empty and inside "
			"[0, {})".format(start, end, L))

	delta = convergence_deltas(m, X1h, y_x, y_r, S, K)
	x, r = split_rows(X1h[..., start:end], S, K)
	hyp, = hypothetical_attributions((m[..., start:end],),
		(x.expand_as(r).flatten(0, 1),), (r.flatten(0, 1),))
	return hyp.view(S, K, *hyp.shape[1:]).mean(dim=1), delta


def precision_flags():
	"""The process-global flags that decide how float32 matrix products and
	convolutions round, as plain values.

	Both the legacy getters (matmul precision, allow_tf32) and the newer
	fp32_precision ones are read, plus cuDNN's benchmark and determinism
	switches and the environment variables in `PRECISION_ENV` (keyed
	"$NAME"). A getter this torch build lacks is recorded as its error.
	"""

	b = torch.backends
	getters = {
		"torch.get_float32_matmul_precision()":
			torch.get_float32_matmul_precision,
		"torch.backends.fp32_precision": lambda: b.fp32_precision,
		"torch.backends.cuda.matmul.allow_tf32":
			lambda: b.cuda.matmul.allow_tf32,
		"torch.backends.cuda.matmul.fp32_precision":
			lambda: b.cuda.matmul.fp32_precision,
		"torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction":
			lambda: b.cuda.matmul.allow_fp16_reduced_precision_reduction,
		"torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction":
			lambda: b.cuda.matmul.allow_bf16_reduced_precision_reduction,
		"torch.backends.cudnn.enabled": lambda: b.cudnn.enabled,
		"torch.backends.cudnn.allow_tf32": lambda: b.cudnn.allow_tf32,
		"torch.backends.cudnn.fp32_precision":
			lambda: b.cudnn.fp32_precision,
		"torch.backends.cudnn.conv.fp32_precision":
			lambda: b.cudnn.conv.fp32_precision,
		"torch.backends.cudnn.benchmark": lambda: b.cudnn.benchmark,
		"torch.backends.cudnn.deterministic": lambda: b.cudnn.deterministic,
		"torch.are_deterministic_algorithms_enabled()":
			torch.are_deterministic_algorithms_enabled,
		"torch.backends.mkldnn.matmul.fp32_precision":
			lambda: b.mkldnn.matmul.fp32_precision,
	}

	flags = {}
	for name, get in getters.items():
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			try:
				value = get()
			except Exception as exc:
				# Recorded, not fatal: the record is for the log and the meta.
				value = "<{}: {}>".format(type(exc).__name__, exc)

		if value is None or isinstance(value, (bool, int, float, str)):
			flags[name] = value
		else:
			flags[name] = str(value)

	for var in PRECISION_ENV:
		flags["$" + var] = os.environ.get(var)

	return flags


@contextlib.contextmanager
def precision_mode(mode="tf32", deterministic=False):
	"""Set the precision flags of `mode` for the body, log them, and restore
	the previous flags on exit.

	* "tf32": float32 matrix products in TF32 (matmul precision "high"), as
	  after ``import cherimoya``, and TF32 allowed in cuDNN's convolutions,
	  torch's own default. This is the precision the default engine computes
	  in on a GPU that supports TF32.
	* "fp32": strict IEEE float32: matmul precision "highest" and no TF32 in
	  cuDNN.

	In both modes cudnn.benchmark is on, as ``import cherimoya`` sets it.
	`deterministic` turns cudnn.deterministic on and cudnn.benchmark off;
	cuBLAS also needs CUBLAS_WORKSPACE_CONFIG set before CUDA initializes,
	which `Engine` does. Inductor's cache keys hold the matrix-product
	precision, so code compiled under one mode is not reused under the
	other.

	The flags are process-global. They are set when the body starts, so
	after every import, and everything in the body runs under them.


	Yields
	------
	flags: dict
		`precision_flags` as set.
	"""

	if mode not in PRECISIONS:
		raise ValueError("precision must be one of {}, got {!r}".format(
			PRECISIONS, mode))

	cudnn = torch.backends.cudnn
	previous = (torch.get_float32_matmul_precision(), cudnn.allow_tf32,
		cudnn.benchmark, cudnn.deterministic)
	try:
		torch.set_float32_matmul_precision("high" if mode == "tf32"
			else "highest")
		cudnn.allow_tf32 = mode == "tf32"
		cudnn.benchmark = not deterministic
		cudnn.deterministic = bool(deterministic)
		flags = precision_flags()
		logger.info("precision mode %s%s: %s", mode, ", deterministic"
			if deterministic else "", _pretty(flags))
		yield flags
	finally:
		torch.set_float32_matmul_precision(previous[0])
		cudnn.allow_tf32, cudnn.benchmark, cudnn.deterministic = previous[1:]
		logger.info("precision mode %s left, flags restored: %s", mode,
			_pretty(precision_flags()))


def _pretty(flags):
	return ", ".join("{}={}".format(name, value)
		for name, value in flags.items())


def _deterministic_cublas():
	"""Export CUBLAS_WORKSPACE_CONFIG=:4096:8, which makes cuBLAS
	deterministic, unless it already holds such a value.

	cuBLAS reads it when PyTorch first sets up a handle, so it must be set
	before CUDA initializes. If CUDA already is, the setting may come too
	late, and a RuntimeWarning says so.
	"""

	value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
	if value in DETERMINISTIC_CUBLAS:
		return

	if value:
		logger.warning("CUBLAS_WORKSPACE_CONFIG=%s is not deterministic; "
			"setting %s", value, DETERMINISTIC_CUBLAS[0])

	os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS[0]
	if torch.cuda.is_initialized():
		warnings.warn("deterministic=True set CUBLAS_WORKSPACE_CONFIG={} "
			"after CUDA was initialized, which may be too late for cuBLAS; "
			"create the Engine before anything initializes CUDA".format(
			DETERMINISTIC_CUBLAS[0]), RuntimeWarning, stacklevel=3)


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
	never coexist, so the sum overestimates by the smaller of the two. For
	the default architecture (C = 128, H = 256, 9 blocks, L = 2114) and
	K = 20 it is 0.93 GB per sequence. cuDNN's workspace and the weights are
	not counted.
	"""

	rows, pairs = S * (K + 1), S * K
	per_row = (itemsize * L * (C + n_blocks * (C + H))
		+ itemsize * L * (4 * C + H) + itemsize * 4 * L + 4 * L)
	per_pair = itemsize * L * (3 * C + 2 * H) + itemsize * 4 * L
	return rows * per_row + pairs * per_pair


def auto_seqs_per_step(available_bytes, budget_gb, peak_bytes):
	"""The automatic step size: floor(min(budget, 0.6 x available) / bytes
	per sequence), clamped to [1, MAX_SEQS_PER_STEP].

	`available_bytes` is the device memory the engine can use, `budget_gb`
	its budget in GB of 1e9 bytes, and `peak_bytes` `step_bytes` at S = 1.
	"""

	limit = min(budget_gb * GB, FREE_MEMORY_FRACTION * available_bytes)
	return max(1, min(MAX_SEQS_PER_STEP, int(limit // peak_bytes)))


class RawResult(NamedTuple):
	"""`Engine.raw`'s output: CPU tensors in the engine's dtype."""

	m: torch.Tensor  # (N, K, 4, L) multipliers; pair (e, k) at [e, k]
	y_x: torch.Tensor  # (N,) the attributed output of each sequence
	y_r: torch.Tensor  # (N, K) the attributed output of each reference
	deltas: torch.Tensor  # (N, K) convergence deltas
	debug: dict  # what ran: step size, steps, padding, statistics, ...


class Result(NamedTuple):
	"""`Engine.run`'s output: CPU tensors in the engine's dtype."""

	attr: torch.Tensor  # (N, 4, W) hypothetical attributions, averaged
	deltas: torch.Tensor  # (N, K) convergence deltas
	y_x: torch.Tensor  # (N,)
	y_r: torch.Tensor  # (N, K)
	meta: dict  # debug's keys, plus the window, the wall time and pairs/s


class _StepOutputs(NamedTuple):
	"""One step's results: the device tensors a step computes, or the pinned
	host buffers they are copied into."""

	y_x: torch.Tensor  # (S,) the attributed output of each sequence
	y_r: torch.Tensor  # (S, K) the attributed output of each reference
	delta: torch.Tensor  # (S, K) convergence deltas
	values: torch.Tensor  # run(): (S, 4, W) attr; raw(): (S, K, 4, L) m


class _Rings:
	"""Two slots of transfer buffers for a run on CUDA: pinned host inputs,
	device inputs, and pinned host outputs. The inputs are int8 one-hots.

	They are sized for S sequences per step; after an out-of-memory halving,
	steps use leading slices. The engine refills a slot only after the host
	has waited for the step that last used it, so no buffer is written while
	a copy may still read it.
	"""

	def __init__(self, S, K, L, values_shape, dtype, device):
		rows = S + S * K
		self.K = K
		self.host_inputs = [torch.empty(rows, 4, L, dtype=torch.int8,
			pin_memory=True) for _ in range(2)]
		self.device_inputs = [torch.empty(rows, 4, L, dtype=torch.int8,
			device=device) for _ in range(2)]
		self.outputs = [
			_StepOutputs(
				torch.empty(S, dtype=dtype, pin_memory=True),
				torch.empty(S, K, dtype=dtype, pin_memory=True),
				torch.empty(S, K, dtype=dtype, pin_memory=True),
				torch.empty(S, *values_shape, dtype=dtype, pin_memory=True),
			)
			for _ in range(2)
		]

	def fill(self, slot, X, references, S):
		"""Write a step's rows ``[x_0..x_{S-1}, r_00..r_{S-1,K-1}]`` into a
		slot's host inputs, padded with the last sequence.

		X (n, 4, L) and references (n, K, 4, L), n <= S, are int8 one-hots.
		"""

		n = len(X)
		rows = self.host_inputs[slot][:S + S * self.K]
		rows[:n] = X
		rows[n:S] = X[-1]
		refs = rows[S:].view(S, self.K, *X.shape[1:])
		refs[:n] = references
		refs[n:] = references[-1]


class Engine:
	"""DeepLIFT/DeepSHAP attributions of a Cherimoya model's count head, as
	`cherimoya attribute` computes them with tangermeme's `deep_lift_shap`.

	The engine works in steps of S whole sequences, each with its K
	references:

	* The step's joint rows ``[x_0..x_{S-1}, r_00..r_{S-1,K-1}]`` are
	  forwarded once: each sequence once, not once per reference.
	* The backward runs over the sequence half only, as S*K pairs: the count
	  gradient enters broadcast over the pairs, then each block's rules and
	  the stem's, then iconv's input gradient.
	* A last step with fewer than S sequences is padded by repeating its last
	  sequence and references, and the padded pairs are dropped, so every
	  step has the same shapes.
	* `run` takes the hypothetical attributions over the window [start, end)
	  and averages them over each sequence's K references on the device.

	On CUDA:

	* The forward, the backward and the epilogue run under torch.compile,
	  fullgraph and with static shapes. The weights are arguments, so the
	  blocks unroll into one graph each way and every matrix product stays
	  an extern cuBLAS call. iconv, its input gradient and the count head run
	  eagerly, with the default engine's operations. torch 2.13 and later
	  keep each engine's compiled code apart from other engines'; on older
	  torch every engine in a process shares torch.compile's default
	  recompile limit of 8 per pass.
	* Each `run` or `raw` call runs a warm-up step on random sequences, which
	  compiles, lets cudnn.benchmark time its algorithms and meets an
	  out-of-memory error. Compiling leaves the traced step's tensors in
	  reference cycles, so a gc.collect() follows every step that compiled.
	* Steps are pipelined. The host draws a step's references
	  (`draw_references`) and fills its pinned buffers while the device runs
	  the previous step. The step's inputs are copied to the device on a copy
	  stream, the step runs, and its results are copied back, non-blocking,
	  into pinned buffers. The host then waits for the previous step's event,
	  its one synchronization per step, and copies that step's results out
	  while the device runs this one.
	* An out-of-memory error in a step empties the allocator's cache, halves
	  S and retries the step's sequences at the new size. Each halving is
	  recorded in the run's "oom" list. At S = 1 the error propagates.


	Parameters
	----------
	model: cherimoya.Cherimoya
		The model. Its weights are copied once and never changed.

	output: str, optional
		What is attributed. Only "counts" is implemented. Default is
		"counts".

	group: int or None, optional
		The signal group to attribute, as `LogCountWrapper` takes it. A model
		with several groups needs one. Default is 0.

	device: str or torch.device or None, optional
		Where the weights and the steps live. None means the model's device.
		Default is None.

	precision: str, optional
		"tf32", the precision the default engine computes in on a GPU (TF32
		matrix products and convolutions), or "fp32", strict IEEE float32.
		See `precision_mode`. Default is "tf32".

	n_shuffles: int, optional
		The number of dinucleotide-shuffled references per sequence, K, when
		none are passed in. Default is 20.

	random_state: int or None, optional
		The base seed of the references. Reference k of sequence e is
		``dinucleotide_shuffle(X[e:e+1].float(), n=1, random_state=
		random_state + k)[:, 0]``, as `deep_lift_shap` draws it. None draws a
		base seed once, records it, and makes the run reproducible from it.
		Default is 0.

	seqs_per_step: int or str, optional
		S, or "auto": on CUDA the largest S whose `step_bytes` fits both
		`mem_budget_gb` and 0.6 of the device memory this process can use, at
		most 32; on the CPU 8. Default is "auto".

	mem_budget_gb: float, optional
		The device memory, in GB of 1e9 bytes, that "auto" plans a step for.
		An explicit `seqs_per_step` whose estimate exceeds it is kept, with a
		warning. Default is 12.0.

	compile: str, optional
		"auto" ("default" on CUDA, "none" on the CPU), "default", or "none",
		which runs the same functions eagerly. Default is "auto".

	forward_stats: str, optional
		The normalization statistics of the forward (`forward_save`). "auto"
		is "cpu_mirror" on the CPU and "two_pass" on CUDA, where "cpu_mirror"
		is refused. Default is "auto".

	warning_threshold: float or None, optional
		`run` warns once, with a summary, if a convergence delta exceeds it.
		None means no warning. Default is 1e-3.

	deterministic: bool, optional
		Bitwise reproducible runs, for reruns rather than speed. On CUDA it
		exports CUBLAS_WORKSPACE_CONFIG=:4096:8 if unset, so create the
		Engine before anything initializes CUDA; it runs with
		cudnn.deterministic on and cudnn.benchmark off, and compiles in
		Inductor's deterministic mode where the torch build has it. Default
		is False.
	"""

	def __init__(self, model, output="counts", group=0, device=None,
		precision="tf32", n_shuffles=20, random_state=0,
		seqs_per_step="auto", mem_budget_gb=12.0,
		compile="auto", forward_stats="auto", warning_threshold=1e-3,
		deterministic=False):

		if output not in OUTPUTS:
			raise ValueError("output must be either `counts` or `profile`.")

		if output == "profile":
			raise NotImplementedError("the fast engine attributes the count "
				"head only; attribute the profile head with tangermeme's "
				"deep_lift_shap")

		check_options(precision, seqs_per_step, mem_budget_gb)
		if not isinstance(deterministic, bool):
			raise ValueError("deterministic must be True or False, got "
				"{!r}".format(deterministic))

		check_tangermeme_version()
		if deterministic and _target_device_type(model, device) == "cuda":
			# Before CheriWeights initializes CUDA.
			_deterministic_cublas()

		self.weights = CheriWeights.from_model(model, device)

		# The default engine's group checks; the wrapper is kept for the
		# self-check and the audit.
		self.wrapper = make_wrapper(model, output, group)
		self.output, self.group = output, group
		self.precision, self.deterministic = precision, deterministic
		self.forward_stats = resolve_forward_stats(forward_stats,
			self.weights.device)
		self.compile = resolve_compile(compile, self.weights.device)
		self.mem_budget_gb = float(mem_budget_gb)
		self.n_shuffles = _positive_int(n_shuffles, "n_shuffles")
		self.seqs_per_step = (seqs_per_step if seqs_per_step == "auto"
			else int(seqs_per_step))

		if random_state is None:
			random_state = int(numpy.random.randint(0, SEED_RANGE))
			logger.info("random_state is None: drew the base seed %d",
				random_state)

		if (isinstance(random_state, bool)
				or not isinstance(random_state, numbers.Integral)):
			raise ValueError("random_state must be an integer or None, got "
				"{!r}".format(random_state))

		self.random_state = int(random_state)

		if warning_threshold is not None and (
				isinstance(warning_threshold, bool)
				or not isinstance(warning_threshold, numbers.Real)):
			raise ValueError("warning_threshold must be a number or None, got "
				"{!r}".format(warning_threshold))

		self.warning_threshold = (None if warning_threshold is None
			else float(warning_threshold))

		self._copy_stream = None

		passes = (forward_save, backward_pass, epilogue)
		if self.compile != "none":
			passes = tuple(self._compiled(fn) for fn in passes)

		self.fwd_c, self.bwd_c, self.epi_c = passes

	def _compiled(self, fn):
		"""fn under torch.compile: fullgraph, static shapes, the default mode,
		and this engine's own compiled entries where torch supports them."""

		kwargs = {"fullgraph": True, "dynamic": False}
		options = None
		if self.deterministic:
			import torch._inductor.config as inductor_config
			if hasattr(inductor_config, "deterministic"):
				options = {"deterministic": True}
			else:
				logger.warning("this torch build's Inductor has no "
					"deterministic mode; compiling without it")

		if options:
			kwargs["options"] = options
		else:
			kwargs["mode"] = "default"

		if _compile_isolates_recompiles():
			kwargs["recompile_limit"] = RECOMPILE_LIMIT
			kwargs["isolate_recompiles"] = True

		return torch.compile(fn, **kwargs)

	def raw(self, X, references=None):
		"""Each pair's multipliers, outputs and convergence delta: what
		``deep_lift_shap(raw_outputs=True)`` computes. For tests and
		validation: the multipliers take N*K*4*L values.

		The steps run as `run`'s do, but copy back the multipliers instead of
		the hypothetical attributions, and the deltas are computed eagerly.


		Parameters
		----------
		X: torch.Tensor or numpy.ndarray, shape=(N, 4, L)
			The one-hot sequences, such as `extract_loci`'s int8 ones, with
			no N.

		references: torch.Tensor or numpy.ndarray or None, optional
			(N, K, 4, L) one-hot references whose columns may be all zero (N),
			or None to draw `n_shuffles` per sequence. Injected references
			set K, as in `deep_lift_shap`. Default is None.


		Returns
		-------
		result: RawResult
			The multipliers (N, K, 4, L), outputs, deltas and a record of the
			call.
		"""

		X, references, K = self._inputs(X, references)
		N, _, L = X.shape
		dtype = self.weights.dtype
		m = torch.empty(N, K, 4, L, dtype=dtype)
		y_x = torch.empty(N, dtype=dtype)
		y_r = torch.empty(N, K, dtype=dtype)
		deltas = torch.empty(N, K, dtype=dtype)

		def consume(a, b, out):
			n = b - a
			m[a:b], y_x[a:b] = out.values[:n], out.y_x[:n]
			y_r[a:b], deltas[a:b] = out.y_r[:n], out.delta[:n]

		debug = self._execute(X, references, K, None, consume)
		debug["deltas"] = delta_summary(deltas, self.warning_threshold)
		return RawResult(m, y_x, y_r, deltas, debug)

	def run(self, X, start, end, references=None, verbose=False):
		"""The hypothetical attributions over the window [start, end),
		averaged over each sequence's references.

		This is ``deep_lift_shap(wrapper, X, hypothetical=True, ...)[:, :,
		start:end]`` as `cherimoya attribute` calls it. The convergence
		deltas and the outputs of every pair come back too, and one warning
		is given if a delta exceeds `warning_threshold`.


		Parameters
		----------
		X: torch.Tensor or numpy.ndarray, shape=(N, 4, L)
			The one-hot sequences, with no N.

		start, end: int
			The window, 0 <= start < end <= L.

		references: torch.Tensor or numpy.ndarray or None, optional
			As in `raw`. Default is None.

		verbose: bool, optional
			Whether to show a progress bar. Default is False.


		Returns
		-------
		result: Result
			The attributions (N, 4, end - start), deltas, outputs and a record
			of the run.
		"""

		X, references, K = self._inputs(X, references)
		N, _, L = X.shape
		if not 0 <= operator.index(start) < operator.index(end) <= L:
			raise ValueError("the window [{}, {}) must be non-empty and "
				"inside [0, {})".format(start, end, L))

		start, end = int(start), int(end)
		dtype = self.weights.dtype
		attr = torch.empty(N, 4, end - start, dtype=dtype)
		y_x = torch.empty(N, dtype=dtype)
		y_r = torch.empty(N, K, dtype=dtype)
		deltas = torch.empty(N, K, dtype=dtype)

		from tqdm import tqdm
		progress = tqdm(total=N, unit="seq", disable=not verbose)

		def consume(a, b, out):
			n = b - a
			attr[a:b] = out.values[:n]
			y_x[a:b], y_r[a:b] = out.y_x[:n], out.y_r[:n]
			deltas[a:b] = out.delta[:n]
			progress.update(n)

		try:
			meta = self._execute(X, references, K, (start, end), consume)
		finally:
			progress.close()

		meta["window"] = [start, end]
		meta["deltas"] = delta_summary(deltas, self.warning_threshold)
		logger.info("fast DeepLIFT/SHAP: %d sequences x %d references in %d "
			"steps of %d, %.1f s (%.0f pairs/s overall, %s steady; warm-up "
			"%s s)", N, K, meta["steps"], meta["seqs_per_step"],
			meta["seconds"], meta["pairs_per_s"],
			"-" if meta["steady_pairs_per_s"] is None
			else "{:.0f}".format(meta["steady_pairs_per_s"]),
			"-" if meta["compile_s"] is None
			else "{:.1f}".format(meta["compile_s"]))

		if meta["deltas"]["above"]:
			warnings.warn(delta_warning(meta["deltas"]), RuntimeWarning,
				stacklevel=2)

		return Result(attr, deltas, y_x, y_r, meta)

	def step_size(self, n_peaks, length, n_refs):
		"""S, the sequences per step a run of `n_peaks` sequences of `length`,
		with `n_refs` references each, starts with.

		An explicit `seqs_per_step` is kept; on CUDA a warning is logged when
		its `step_bytes` exceeds `mem_budget_gb`. "auto" is, on CUDA,
		`auto_seqs_per_step` over the memory this process can use on the
		device: what is free plus what its caching allocator holds unused.
		On the CPU it is `AUTO_SEQS_PER_STEP`. Never more than `n_peaks`.
		"""

		cuda = self.weights.device.type == "cuda"
		if self.seqs_per_step != "auto":
			S = self.seqs_per_step
			estimate = self.step_estimate(S, n_refs, length)
			if cuda and estimate > self.mem_budget_gb * GB:
				logger.warning("seqs_per_step=%d is estimated at %.2f GB per "
					"step, above mem_budget_gb=%g", S, estimate / GB,
					self.mem_budget_gb)
		elif cuda:
			device = self.weights.device
			free, _ = torch.cuda.mem_get_info(device)
			available = (free + torch.cuda.memory_reserved(device)
				- torch.cuda.memory_allocated(device))
			per_peak = self.step_estimate(1, n_refs, length)
			S = auto_seqs_per_step(available, self.mem_budget_gb, per_peak)
			logger.info("seqs_per_step auto: %d (%.2f GB per sequence, budget "
				"%g GB, %.1f GB available)", S, per_peak / GB,
				self.mem_budget_gb, available / GB)
		else:
			S = AUTO_SEQS_PER_STEP

		return min(S, n_peaks)

	def step_estimate(self, S, n_refs, length):
		"""`step_bytes` for this engine's model."""

		w = self.weights
		return step_bytes(S, n_refs, length, w.C, w.H, len(w.blocks),
			w.dtype.itemsize)

	def _execute(self, X, references, K, window, consume):
		"""Run every step of a `raw` (window None) or `run` call under the
		engine's precision mode.

		X (N, 4, L) and references (N, K, 4, L) or None are int8 one-hots on
		the CPU. consume(a, b, out) receives each step's results for the
		sequences [a, b), in order; out has S >= b - a rows and its buffers
		are reused, so consume copies what it keeps. Returns the call's
		record.
		"""

		N, _, L = X.shape
		cuda = self.weights.device.type == "cuda"
		began = time.perf_counter()
		with contextlib.ExitStack() as stack:
			if cuda:
				stack.enter_context(torch.cuda.device(self.weights.device))

			flags = stack.enter_context(precision_mode(self.precision,
				deterministic=self.deterministic))
			S = self.step_size(N, L, K)
			record = self._describe(N, L, K, S)
			record["precision_flags"] = flags
			record["references"] = ("produced" if references is None
				else "injected")
			G0 = counts_cotangent(self.weights, self.group, L)
			if cuda:
				S = self._warm_up(S, K, L, G0, window, record)
				self._steps_cuda(X, references, K, L, S, G0, window, consume,
					record)
			else:
				self._steps_cpu(X, references, K, S, G0, window, consume,
					record)

		record["seconds"] = time.perf_counter() - began
		record["pairs_per_s"] = N * K / record["seconds"]
		return record

	def _step_references(self, X, references, a, b, K, record):
		"""The references of the sequences [a, b), (b - a, K, 4, L) int8:
		sliced from those passed in, or drawn (`draw_references`)."""

		if references is not None:
			return references[a:b]

		began = time.perf_counter()
		drawn = draw_references(X[a:b], K, self.random_state)
		record["reference_s"] += time.perf_counter() - began
		return drawn

	def _device_step(self, inputs, S, K, G0, window):
		"""One step on the engine's device: the joint rows (S + S*K, 4, L), an
		int8 one-hot -> the step's results, on the device.

		Eager: the one-hot, iconv, the count head and iconv's input gradient.
		Compiled on CUDA: the forward (fwd_c), the backward (bwd_c) and, for
		`run`, the epilogue (epi_c). Each large intermediate is released as
		soon as the next stage no longer needs it.
		"""

		w = self.weights
		X1h = inputs.to(w.dtype)
		z0 = stem_forward(X1h, w.W0, w.b0)
		hL, saved = self.fwd_c(z0, w.blocks, w.rs, w.eps, self.forward_stats)
		y = head_forward(hL, w, output=self.output, group=self.group)
		del hL
		Gz = self.bwd_c(G0, saved, z0, w.blocks, w.rs, S, K)
		del saved, z0
		m = input_multipliers(Gz, w.W0)
		del Gz
		y_x, y_r = y[:S], y[S:].view(S, K)
		if window is None:
			delta = convergence_deltas(m, X1h, y_x, y_r, S, K)
			return _StepOutputs(y_x, y_r, delta, m.view(S, K, *m.shape[1:]))

		attr, delta = self.epi_c(m, X1h, y_x, y_r, window[0], window[1], S, K)
		return _StepOutputs(y_x, y_r, delta, attr)

	def _steps_cpu(self, X, references, K, S, G0, window, consume, record):
		"""The steps on the CPU, one after the other."""

		for a in range(0, len(X), S):
			b = min(a + S, len(X))
			refs = self._step_references(X, references, a, b, K, record)
			joint = self._joint_inputs(X[a:b], refs, S)
			consume(a, b, self._device_step(joint.to(self.weights.device), S,
				K, G0, window))
			record["steps"] += 1
			record["padded_peaks"] += S - (b - a)

	def _warm_up(self, S, K, L, G0, window, record):
		"""One step on random sequences before a CUDA run; returns the step
		size the run then uses.

		It compiles the passes for this S, lets cudnn.benchmark time its
		algorithms, and halves S on an out-of-memory error, all before the
		first step. The sequences are drawn on the device, so the warm-up
		copies nothing from the host, and its results are dropped. It waits
		for the device once, on an event.
		"""

		device = self.weights.device
		generator = torch.Generator(device=device)
		generator.manual_seed(0)
		began = time.perf_counter()
		while True:
			attempt = time.perf_counter()
			tokens = torch.randint(0, 4, (S + S * K, L), generator=generator,
				device=device)
			# Contiguous, as the real steps' inputs are, so that the passes
			# compiled here are not compiled again for other strides.
			inputs = F.one_hot(tokens, 4).transpose(1, 2).to(
				torch.int8).contiguous()
			failed = None
			try:
				self._device_step(inputs, S, K, G0, window)
				done = torch.cuda.Event(blocking=True)
				done.record()
				done.synchronize()
			except torch.cuda.OutOfMemoryError as exc:
				if S == 1:
					raise

				failed = str(exc)

			if failed is None:
				break

			del tokens, inputs
			S = self._after_oom(S, failed, record, "warm-up")

		gc.collect()  # frees the step that compiling kept alive (see _launch)
		record["compile_s"] = time.perf_counter() - attempt
		record["warmup_s"] = time.perf_counter() - began
		return S

	def _steps_cuda(self, X, references, K, L, S, G0, window, consume,
		record):
		"""The steps on CUDA, pipelined: step i is queued before the host
		waits for step i - 1 (see the class docstring)."""

		device = self.weights.device
		if window is None:
			values_shape = (K, 4, L)
		else:
			values_shape = (4, window[1] - window[0])

		rings = _Rings(S, K, L, values_shape, self.weights.dtype, device)
		if self._copy_stream is None:
			self._copy_stream = torch.cuda.Stream(device)

		torch.cuda.reset_peak_memory_stats(device)
		base, first_S = torch.cuda.memory_allocated(device), S
		N = len(X)
		starts = collections.deque(range(0, N, S))
		pending, finished, slot, recompiled = None, [], 0, False
		while starts:
			a = starts.popleft()
			b = min(a + S, N)
			refs = self._step_references(X, references, a, b, K, record)
			rings.fill(slot, X[a:b], refs, S)
			failed = None
			try:
				launched = self._launch(rings, slot, S, K, G0, window)
			except torch.cuda.OutOfMemoryError as exc:
				if S == 1:
					raise

				failed = str(exc)

			if failed is not None:
				if pending is not None:
					self._finish(pending, consume, record, finished)
					pending = None

				S = self._after_oom(S, failed, record, record["steps"] + 1)
				starts = collections.deque(range(a, N, S))
				recompiled = True
				continue

			if recompiled:  # this step compiled the passes for the new S
				gc.collect()
				recompiled = False

			record["steps"] += 1
			record["padded_peaks"] += S - (b - a)
			if pending is not None:
				self._finish(pending, consume, record, finished)

			pending = (a, b, *launched)
			slot ^= 1

		if pending is not None:
			self._finish(pending, consume, record, finished)

		record["step_peak_bytes"] = torch.cuda.max_memory_allocated(device) - base
		record["step_estimate_bytes"] = self.step_estimate(first_S, K, L)
		ratio = record["step_peak_bytes"] / record["step_estimate_bytes"]
		log = logger.warning if ratio > MEMORY_CHECK_FACTOR else logger.info
		log("steps of %d sequences allocated at most %.3f GB, %.2f x the "
			"estimate of %.3f GB (warning above %.1f x)", first_S,
			record["step_peak_bytes"] / GB, ratio,
			record["step_estimate_bytes"] / GB, MEMORY_CHECK_FACTOR)

		if len(finished) > 1 and finished[-1][0] > finished[0][0]:
			pairs = sum(n for _, n in finished[1:])
			seconds = finished[-1][0] - finished[0][0]
			record["steady_pairs_per_s"] = pairs / seconds

	def _launch(self, rings, slot, S, K, G0, window):
		"""Queue one step on the device without waiting: inputs in on the
		copy stream, the step, results out. Returns the pinned host buffers
		the results land in and the event that marks their arrival.

		A call that compiles (the first at a step size) leaves Dynamo's
		tracing state in reference cycles that also hold the real tensors the
		passes were traced with, a whole step's activations. Only the cyclic
		garbage collector frees them, so the engine runs gc.collect() after
		each such call.
		"""

		rows = S + S * K
		compute = torch.cuda.current_stream()
		inputs = rings.device_inputs[slot][:rows]
		with torch.cuda.stream(self._copy_stream):
			inputs.copy_(rings.host_inputs[slot][:rows], non_blocking=True)
			copied = torch.cuda.Event()
			copied.record()

		compute.wait_event(copied)
		out = self._device_step(inputs, S, K, G0, window)
		host = _StepOutputs(*(buffer[:S] for buffer in rings.outputs[slot]))
		for dst, src in zip(host, out):
			dst.copy_(src, non_blocking=True)

		done = torch.cuda.Event(blocking=True)
		done.record(compute)
		return host, done

	@staticmethod
	def _finish(pending, consume, record, finished):
		"""Wait for a queued step's results, the step's one host
		synchronization, and hand them to consume."""

		a, b, host, done = pending
		done.synchronize()
		record["host_syncs"] += 1
		consume(a, b, host)
		finished.append((time.perf_counter(), (b - a) * record["n_shuffles"]))

	def _after_oom(self, S, message, record, step):
		"""Recover from an out-of-memory error at `step`: wait for the device,
		empty the cache, halve S."""

		torch.cuda.synchronize(self.weights.device)
		gc.collect()  # anything the failed step left in reference cycles
		torch.cuda.empty_cache()
		new = S // 2
		record["oom"].append({"step": step, "from": S, "to": new})
		record["seqs_per_step"] = new
		first_line = message.strip().splitlines()[0] if message.strip() else (
			"no message")
		logger.warning("out of GPU memory at step %s with %d sequences per "
			"step; retrying with %d (%s)", step, S, new, first_line)
		return new

	@staticmethod
	def _joint_inputs(X, references, S):
		"""The step's rows ``[x_0..x_{S-1}, r_00..r_{S-1,K-1}]`` as one int8
		one-hot (S + S*K, 4, L), padded with the last sequence.

		X (n, 4, L) and references (n, K, 4, L) are int8 one-hots. With n < S,
		the last sequence and its references are repeated to fill the step.
		"""

		n = len(X)
		if n < S:
			keep = torch.cat([torch.arange(n), torch.full((S - n,), n - 1)])
			X, references = X[keep], references[keep]

		return torch.cat([X, references.flatten(0, 1)])

	def _inputs(self, X, references):
		"""X and the references, checked as `deep_lift_shap` checks them, as
		int8 one-hots on the CPU. Returns X (N, 4, L), the references
		(N, K, 4, L) or None, and K.

		X may not hold an unknown character (an all-zero column); the
		references may. Injected references set K, as in `deep_lift_shap`.
		"""

		X = torch.as_tensor(X)
		validate_input(X, "X", shape=(-1, 4, -1), ohe=True)
		if X.shape[0] == 0:
			raise ValueError("deep_lift_shap requires at least one example; "
				"got X with shape[0] == 0.")

		X = X.to(device="cpu", dtype=torch.int8)
		if references is None:
			return X, None, self.n_shuffles

		R = torch.as_tensor(references)
		validate_input(R, "references", shape=(len(X), -1, 4, X.shape[-1]),
			ohe=True, allow_N=True, ohe_dim=-2)
		if R.shape[1] == 0:
			raise ValueError("references must hold at least one reference per "
				"sequence, got shape {}".format(tuple(R.shape)))

		return X, R.to(device="cpu", dtype=torch.int8), R.shape[1]

	def _describe(self, N, L, K, S):
		return {
			"output": self.output,
			"group": self.group,
			"device": str(self.weights.device),
			"dtype": str(self.weights.dtype).replace("torch.", ""),
			"precision": self.precision,
			"compile": self.compile,
			"deterministic": self.deterministic,
			"forward_stats": self.forward_stats,
			"n_sequences": N,
			"length": L,
			"n_shuffles": K,
			"pairs": N * K,
			"random_state": self.random_state,
			"seqs_per_step": S,
			"mem_budget_gb": self.mem_budget_gb,
			"steps": 0,
			"padded_peaks": 0,
			"reference_s": 0.0,
			"host_syncs": 0,
			"oom": [],
			"compile_s": None,
			"warmup_s": None,
			"steady_pairs_per_s": None,
			"step_peak_bytes": None,
			"step_estimate_bytes": None,
		}


def check_options(precision="tf32", seqs_per_step="auto", mem_budget_gb=12.0):
	"""Raise a ValueError for a `precision`, `seqs_per_step` or
	`mem_budget_gb` that `Engine` does not take. `cherimoya attribute` calls
	it before it loads the model."""

	if precision not in PRECISIONS:
		raise ValueError("precision must be one of {}, got {!r}".format(
			PRECISIONS, precision))

	if seqs_per_step != "auto":
		_positive_int(seqs_per_step, "seqs_per_step", "'auto' or ")

	if (isinstance(mem_budget_gb, bool)
			or not isinstance(mem_budget_gb, numbers.Real)
			or not mem_budget_gb > 0):
		raise ValueError("mem_budget_gb must be a positive number, got "
			"{!r}".format(mem_budget_gb))


def _compile_isolates_recompiles():
	"""Whether torch.compile takes `recompile_limit` and `isolate_recompiles`
	(torch 2.13 and later)."""

	try:
		parameters = inspect.signature(torch.compile).parameters
	except (TypeError, ValueError):
		return False

	return "recompile_limit" in parameters and "isolate_recompiles" in parameters


def _target_device_type(model, device):
	"""The type of the device an Engine will run on, read before its weights
	move there; None if unknown."""

	if device is not None:
		return torch.device(device).type

	weight = getattr(getattr(model, "iconv", None), "weight", None)
	return weight.device.type if isinstance(weight, torch.Tensor) else None


def delta_summary(deltas, threshold):
	"""The convergence deltas in brief: pairs, threshold, how many exceed it,
	max, p50, p99 and p99.9.

	A delta exceeds the threshold as tangermeme decides it, ``deltas >
	threshold`` in the deltas' dtype. `above` is None when the threshold is.
	"""

	values = deltas.detach().cpu()
	p50, p99, p999 = numpy.quantile(values.double().flatten().numpy(),
		[0.5, 0.99, 0.999])
	return {
		"pairs": values.numel(),
		"threshold": threshold,
		"above": None if threshold is None else int((values > threshold).sum()),
		"max": float(values.max()),
		"p50": float(p50),
		"p99": float(p99),
		"p99.9": float(p999),
	}


def delta_warning(summary):
	"""The one warning a run gives for high deltas, where tangermeme warns
	per batch with the whole batch's deltas."""

	return ("Convergence deltas too high: {} of {} pairs > {:g} (max {:.4g}, "
		"p50/p99/p99.9 {:.3g}/{:.3g}/{:.3g})".format(summary["above"],
		summary["pairs"], summary["threshold"], summary["max"],
		summary["p50"], summary["p99"], summary["p99.9"]))


def _positive_int(value, name, alternatives=""):
	if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
			or value < 1):
		raise ValueError("{} must be {}a positive integer, got {!r}".format(
			name, alternatives, value))

	return int(value)
