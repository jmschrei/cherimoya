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
* The rules are tangermeme's: the rescale rule of `_nonlinear` for every
  GELU, and the closed-form normalization rule of
  `_layer_normalization_helper` that `cherimoya.deep_lift_shap.conv_norm_op`
  applies to `FusedDilatedConvNorm`, with the same expressions in the same
  order. The references are tangermeme's `dinucleotide_shuffle`, called with
  the seeds `deep_lift_shap` uses (see `references`).

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
import itertools
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

from ..cheri import CONV_NORM_EPS
from ..cheri import _cheri_conv
from ..cherimoya import Cherimoya
from ..wrappers import ControlWrapper
from ..wrappers import LogCountWrapper
from ..wrappers import ProfileWrapper
from . import references as _references


logger = logging.getLogger(__name__)

VALIDATED_TANGERMEME_VERSIONS = ("1.5.0",)
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

INDUCTOR_CACHE_PREFIX = "fast_deep_lift_shap_"

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

RESCALE_EPS = 1e-6
"""The rescale rule takes the plain gradient where |z_x - z_r| < RESCALE_EPS,
strictly, as tangermeme's `_nonlinear` does."""

DEFAULT_HEAD_ROWS = 128
"""Rows of the head that `counts_cotangent` differentiates: 2 x 64, the
default engine's full batch at its default `batch_size`."""

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


def rule_stats(y, eps):
	"""The statistics of the normalization rule over each row's (L, C)
	plane: (mu, y - mu, v), with the reduced axes kept.

	The expressions are those of tangermeme's `_layer_normalization_helper`,
	applied as `cherimoya.deep_lift_shap._ConvNormView` sets it up:
	normalized over the trailing two axes, eps 1e-3. They are computed in
	y's dtype.
	"""

	mu = y.mean(dim=(-2, -1), keepdim=True)
	a = y - mu
	var = (a**2).mean(dim=(-2, -1), keepdim=True)
	v = (var + eps) ** (-0.5)
	return mu, a, v


def norm_two_pass(y, eps):
	"""The forward's normalization with the rule's own two-pass statistics:
	(yhat, mu, v), mu and v with the reduced axes kept.

	yhat = (y - mu) * v over each row's (L, C) plane. The model's Triton
	forward computes the variance as E[y^2] - mean^2 instead; on zero-mean
	input the two agree to about 1e-6.
	"""

	mu, a, v = rule_stats(y, eps)
	return a * v, mu, v


def _cpu_norm(y, eps):
	"""The model's CPU normalization (`cherimoya.cheri._cheri_conv_norm_cpu`)
	verbatim, its float32 cast included even for float64."""

	N, L, C = y.shape
	flat = y.reshape(N, -1).float()
	mean = flat.mean(dim=1, keepdim=True)
	var = flat.var(dim=1, keepdim=True, unbiased=False)
	rstd = (var + eps).rsqrt()
	flat = (flat - mean) * rstd
	return flat.reshape(N, L, C).to(y.dtype)


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

	In every mode the saved mu and v are the rule's two-pass statistics, in
	the model's dtype. On CUDA this function runs compiled; nothing in it
	may break the graph.


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
			mu, _, v = rule_stats(y, eps)
			if stats == "cpu_mirror":
				yhat = _cpu_norm(y, eps)
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

	if group is not None and (isinstance(group, bool)
			or not isinstance(group, int) or not 0 <= group < w.n_groups):
		raise ValueError("group must be None or an int in [0, {}), got "
			"{!r}".format(w.n_groups, group))

	R, L, _ = hL.shape
	T = w.T
	if L <= 2 * T:
		raise ValueError("input length {} leaves no positions after trimming "
			"{} from each end".format(L, T))

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


def counts_cotangent(w, group=0, length=None, rows=DEFAULT_HEAD_ROWS):
	"""The gradient of the attributed count with respect to the final
	residual stream, (L, C), which every row shares.

	It is autograd's gradient of ``y.sum()`` through `head_forward`, which
	runs the model's own head operations on the model's layout, at `rows`
	rows. With `rows` set to 2 x the default engine's `batch_size`, the
	count Linear's backward is the same matrix product, over the same number
	of rows, as there. The result is zero outside [T, L - T). Compute it once
	per model and broadcast it over the pairs. It turns grad mode on for
	itself, so the engine can run under no_grad, but not under
	torch.inference_mode().
	"""

	if length is None:
		raise ValueError("counts_cotangent needs the input length")

	if torch.is_inference_mode_enabled():
		raise RuntimeError("counts_cotangent needs autograd, which "
			"torch.inference_mode() switches off")

	if rows < 1:
		raise ValueError("rows must be at least 1, got {}".format(rows))

	with torch.enable_grad():
		hL = torch.zeros(rows, length, w.C, device=w.device, dtype=w.dtype,
			requires_grad=True)
		y = head_forward(hL, w, output="counts", group=group)
		grad, = torch.autograd.grad(y.sum(), hL)

	return grad[0].contiguous()


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
	`_nonlinear`.

	G (S, K, ...) is each pair's gradient at the GELU's output, and z
	(S + S*K, ...) the GELU's input over the joint rows. The result,
	(S, K, ...), is the gradient at the GELU's input:

	* G times the secant (gelu(z_x) - gelu(z_r)) / (z_x - z_r);
	* where |z_x - z_r| < 1e-6, strictly, the plain gradient instead: what
	  nn.GELU's autograd hands tangermeme, gelu_backward(G, z_x).

	The GELU's outputs are recomputed from z over all rows at once, as the
	forward computed them, so the secants use the forward's own values.
	"""

	zx, zr = split_rows(z, S, K)
	gx, gr = split_rows(F.gelu(z, approximate="tanh"), S, K)
	din = zx - zr
	small = din.abs() < RESCALE_EPS
	# The masked entries are never used; this keeps them finite.
	secant = (gx - gr) / din.masked_fill(small, 1.0)
	gradient = torch.ops.aten.gelu_backward(G, zx.expand_as(G),
		approximate="tanh")
	return torch.where(small, gradient, G * secant)


def ln_rule(Gn, y, mu, v, S, K):
	"""The normalization rule of `conv_norm_op` on the sequence half.

	`cherimoya.deep_lift_shap.conv_norm_op` applies tangermeme's
	`_layer_normalization_helper` to the depthwise convolution's output: the
	whole (L, C) plane of each row, D = L * C, eps 1e-3, no affine weight.
	The helper's gamma of 1.0 multiplies exactly and is left out.

	Gn (S, K, L, C) is each pair's gradient at the normalization's output, y
	(S + S*K, L, C) its input over the joint rows, and mu and v (S + S*K,)
	the statistics of `rule_stats`. Returns the gradient at y, (S, K, L, C),
	with the helper's expressions in the helper's order.
	"""

	L, C = y.shape[-2:]
	D = L * C
	yx, yr = split_rows(y, S, K)
	mx, mr = (t[..., None, None] for t in split_rows(mu, S, K))
	vx, vr = (t[..., None, None] for t in split_rows(v, S, K))
	v_avg = (vx + vr) / 2
	a_sum = (yx - mx) + (yr - mr)
	ratio = (-(vx**2) * (vr**2)) / (2 * v_avg)
	variance_term = ratio * a_sum / (2 * D)
	term1 = v_avg * (Gn - Gn.mean(dim=(-2, -1), keepdim=True))
	dot = (Gn * a_sum).sum(dim=(-2, -1), keepdim=True)
	return term1 + variance_term * dot


def dw3_t(G, cw, d):
	"""The transpose of `dw3`, i.e. the depthwise convolution's input
	gradient, as a shift-add.

	G has shape (..., L, C). ``g[l] = cw[1] * G[l] + cw[0] * G[l + d] +
	cw[2] * G[l - d]``, with zeros outside [0, L), in the term order of the
	Triton backward.
	"""

	L = G.shape[-2]
	Gp = F.pad(G, (0, 0, d, d))
	return (Gp[..., d:d + L, :] * cw[1] + Gp[..., 2 * d:2 * d + L, :] * cw[0]
		+ Gp[..., :L, :] * cw[2])


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


def input_multipliers(Gz, X_refs, W0):
	"""iconv's input gradient: the multipliers m (S*K, 4, L) at the one-hot
	input.

	This is the call autograd makes for ``F.conv1d(X, W0, b0, padding=10)``
	when only the input needs a gradient: `aten.convolution_backward` with
	the output mask (True, False, False). X_refs, the reference rows
	(S*K, 4, L), is a real contiguous tensor of the input's shape; the input
	gradient does not read its values.
	"""

	return torch.ops.aten.convolution_backward(Gz, X_refs, W0, None, [1],
		[ICONV_PADDING], [1], False, [0], 1, [True, False, False])[0]


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
	attributions over the window [start, end).

	m (S*K, 4, L) holds the multipliers, X1h (S + S*K, 4, L) the joint
	one-hot rows, and y_x (S,) and y_r (S, K) the attributed outputs.


	Returns
	-------
	hyp: torch.Tensor, shape=(S, K, 4, end - start)
		The hypothetical attributions, contiguous. For one-hot references (an
		all-zero N column included) they are bitwise tangermeme's
		`hypothetical_attributions` sliced to the window: both round once, to
		fl(m_i - m_b) for the reference base b.

	delta: torch.Tensor, shape=(S, K)
		`convergence_deltas` of the multipliers.
	"""

	L = X1h.shape[-1]
	if not 0 <= start < end <= L:
		raise ValueError("the window [{}, {}) must be non-empty and inside "
			"[0, {})".format(start, end, L))

	delta = convergence_deltas(m, X1h, y_x, y_r, S, K)
	_, r = split_rows(X1h, S, K)
	m = m.view(S, K, *m.shape[1:])
	mw, rw = m[..., start:end], r[..., start:end]
	hyp = (mw - (rw * mw).sum(dim=-2, keepdim=True)).contiguous()
	return hyp, delta


def mean_over_references(hyp, start, length):
	"""Each sequence's mean over its K references, on the CPU, bitwise as
	tangermeme computes it.

	tangermeme stacks a sequence's K full-length hypothetical attributions
	into a contiguous (K, 4, length) block and takes ``.mean(dim=0)`` on the
	CPU. The CPU kernel's summation order for a column depends on where the
	column sits in that block, so each window is written at its own
	positions into a zeroed (K, 4, length) block, averaged there and sliced.
	Columns are summed independently, so the zeros outside the window change
	nothing inside it.


	Parameters
	----------
	hyp: torch.Tensor, shape=(S, K, 4, W)
		The windows [start, start + W) of the hypothetical attributions.

	start: int
		Where the window starts.

	length: int
		The full input length.


	Returns
	-------
	attr: torch.Tensor, shape=(S, 4, W)
		The mean over each sequence's references.
	"""

	if hyp.device.type != "cpu":
		raise ValueError("the mean over references runs on the CPU, as "
			"tangermeme's does; got {}".format(hyp.device))

	S, K, A, W = hyp.shape
	if not 0 <= start <= start + W <= length:
		raise ValueError("the window [{}, {}) must lie inside [0, {})".format(
			start, start + W, length))

	out = hyp.new_empty(S, A, W)
	block = hyp.new_zeros(K, A, length)
	for s in range(S):
		block[:, :, start:start + W] = hyp[s]
		out[s] = block.mean(dim=0)[:, start:start + W]

	return out


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


def inductor_cache_dir(mode, root=None):
	"""The Inductor cache directory of the precision mode `mode`:
	``<root>/fast_deep_lift_shap_<mode>``.

	`root` defaults to TORCHINDUCTOR_CACHE_DIR, or to Inductor's own default
	when that is unset. A root that is itself a mode's directory, as inside
	`precision_mode`, is replaced by its parent.
	"""

	if root is None:
		root = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

	if not root:
		from torch._inductor.runtime.cache_dir_utils import default_cache_dir
		root = default_cache_dir()

	root = os.path.abspath(os.fspath(root))
	if os.path.basename(root).startswith(INDUCTOR_CACHE_PREFIX):
		root = os.path.dirname(root)

	return os.path.join(root, INDUCTOR_CACHE_PREFIX + mode)


@contextlib.contextmanager
def precision_mode(mode="tf32", deterministic=False, inductor_cache=True):
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
	which `Engine` does. With `inductor_cache`, TORCHINDUCTOR_CACHE_DIR names
	the mode's own directory (`inductor_cache_dir`) for the body, so code
	compiled under one mode is cached apart from the other's.

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
		path = inductor_cache_dir(mode) if inductor_cache else None
		with _inductor_cache(path):
			flags = precision_flags()
			logger.info("precision mode %s%s: %s", mode, ", deterministic"
				if deterministic else "", _pretty(flags))
			yield flags
	finally:
		torch.set_float32_matmul_precision(previous[0])
		cudnn.allow_tf32, cudnn.benchmark, cudnn.deterministic = previous[1:]
		logger.info("precision mode %s left, flags restored: %s", mode,
			_pretty(precision_flags()))


@contextlib.contextmanager
def _inductor_cache(path):
	"""Point Inductor's caches at `path` for the body, clearing its memoized
	directories on entry and exit."""

	if path is None:
		yield
		return

	from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir

	os.makedirs(path, exist_ok=True)
	with temporary_cache_dir(path):
		yield


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
	(L x H)); the one-hot (4 x L) and its uint8 tokens (L bytes). Per pair:
	the backward's transients, counted the same way (3 (L x C) + 2 (L x H)),
	and the multipliers (4 x L). The forward's and the backward's transients
	never coexist, so the sum overestimates by the smaller of the two. For
	the default architecture (C = 128, H = 256, 9 blocks, L = 2114) and
	K = 20 it is 0.93 GB per sequence. cuDNN's workspace and the weights are
	not counted.
	"""

	rows, pairs = S * (K + 1), S * K
	per_row = (itemsize * L * (C + n_blocks * (C + H))
		+ itemsize * L * (4 * C + H) + itemsize * 4 * L + L)
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


def _substeps(a, b, references, S):
	"""Sequences [a, b) and their references (b - a, K, L), in steps of at
	most S sequences."""

	for j in range(0, b - a, S):
		yield a + j, min(a + j + S, b), references[j:j + S]


def _recut(chunks, S):
	"""Consecutive chunks (a, b, references of sequences [a, b)), re-cut into
	chunks of S sequences: [a0, a0 + S), ...

	The reference pool starts drawing before the warm-up, in chunks of the
	planned step size. When the warm-up halves that size, the engine re-cuts
	the chunks with this, so the run takes exactly the steps, and pads
	exactly the sequences, it would have had the pool drawn chunks of the new
	size. Only the last chunk is shorter.
	"""

	held = []
	a = None
	n = 0
	for start, b, references in chunks:
		if a is None:
			a = start

		if start != a + n or b - start != len(references):
			raise RuntimeError("reference chunks are not consecutive: got "
				"[{}, {}) after [{}, {})".format(start, b, a, a + n))

		held.append(references)
		n += b - start
		while n >= S:
			block = held[0] if len(held) == 1 else numpy.concatenate(held)
			yield a, a + S, block[:S]
			held = [block[S:]] if len(block) > S else []
			a, n = a + S, n - S

	if n:
		yield a, a + n, held[0] if len(held) == 1 else numpy.concatenate(held)


def _keeping(chunks, keep, out):
	"""Pass chunks through, copying the references of the sequences in
	`keep` (sorted) into out[i] as they go by."""

	for a, b, references in chunks:
		lo, hi = numpy.searchsorted(keep, [a, b])
		for i in range(lo, hi):
			out[i] = references[keep[i] - a]

		yield a, b, references


def _keep_indices(keep, n):
	"""`keep_references` as a sorted int64 array of distinct indices in
	[0, n), or ValueError."""

	k = numpy.asarray(keep)
	if k.ndim != 1 or (k.size and not numpy.issubdtype(k.dtype, numpy.integer)):
		raise ValueError("keep_references must be a 1-D sequence of sequence "
			"indices, got {!r}".format(keep))

	k = k.astype(numpy.int64)
	if k.size and (k[0] < 0 or k[-1] >= n or bool((numpy.diff(k) <= 0).any())):
		raise ValueError("keep_references must be increasing sequence "
			"indices in [0, {}), got {}".format(n, k.tolist()))

	return k


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
	references: numpy.ndarray = None  # (n_kept, K, L) uint8 tokens, or None


class _StepOutputs(NamedTuple):
	"""One step's results: the device tensors a step computes, or the pinned
	host buffers they are copied into."""

	y_x: torch.Tensor  # (S,) the attributed output of each sequence
	y_r: torch.Tensor  # (S, K) the attributed output of each reference
	delta: torch.Tensor  # (S, K) convergence deltas
	values: torch.Tensor  # run(): (S, K, 4, W) hyp; raw(): (S, K, 4, L) m


class _Rings:
	"""Two slots of transfer buffers for a run on CUDA: pinned host tokens,
	device tokens, and pinned host outputs.

	They are sized for S sequences per step; after an out-of-memory halving,
	steps use leading slices. The engine refills a slot only after the host
	has waited for the step that last used it, so no buffer is written while
	a copy may still read it.
	"""

	def __init__(self, S, K, L, values_shape, dtype, device):
		rows = S + S * K
		self.K, self.L = K, L
		self.host_tokens = [torch.empty(rows, L, dtype=torch.uint8,
			pin_memory=True) for _ in range(2)]
		self.device_tokens = [torch.empty(rows, L, dtype=torch.uint8,
			device=device) for _ in range(2)]
		self.outputs = [
			_StepOutputs(
				torch.empty(S, dtype=dtype, pin_memory=True),
				torch.empty(S, K, dtype=dtype, pin_memory=True),
				torch.empty(S, K, dtype=dtype, pin_memory=True),
				torch.empty(S, K, *values_shape, dtype=dtype, pin_memory=True),
			)
			for _ in range(2)
		]

	def fill(self, slot, x_tokens, ref_tokens, S):
		"""Write a step's rows ``[x_0..x_{S-1}, r_00..r_{S-1,K-1}]`` into a
		slot's host tokens, padded with the last sequence.

		x_tokens (n, L) and ref_tokens (n, K, L), n <= S, are uint8 tokens.
		"""

		n = len(x_tokens)
		rows = self.host_tokens[slot][:S + S * self.K].numpy()
		rows[:n] = x_tokens
		rows[n:S] = x_tokens[-1]
		refs = rows[S:].reshape(S, self.K, self.L)
		refs[:n] = ref_tokens
		refs[n:] = ref_tokens[-1]


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
	  and averages them over each sequence's K references on the CPU,
	  bitwise as tangermeme does.

	On CUDA:

	* The forward, the backward and the epilogue run under torch.compile,
	  fullgraph and with static shapes. The weights are arguments, so the
	  blocks unroll into one graph each way and every matrix product stays
	  an extern cuBLAS call. iconv, its input gradient and the count head run
	  eagerly, with the default engine's operations. torch 2.13 and later
	  keep each engine's compiled code apart from other engines'; on older
	  torch every engine in a process shares torch.compile's default
	  recompile limit of 8 per pass.
	* Each `run` or `raw` call starts the reference pool, then runs a warm-up
	  step on random tokens, which compiles, lets cudnn.benchmark time its
	  algorithms and meets an out-of-memory error, while the pool draws the
	  first chunks of references. Compiling leaves the traced step's tensors
	  in reference cycles, so a gc.collect() follows every step that
	  compiled.
	* Steps are pipelined. A step's tokens are copied to the device on a copy
	  stream, the step runs, and its results are copied back, non-blocking,
	  into pinned buffers. The host then waits for the previous step's event,
	  its one synchronization per step, and averages that step on the CPU
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

	head_rows: int, optional
		The rows of the head `counts_cotangent` differentiates. `cherimoya
		attribute` passes 2 x its `batch_size`, the default engine's full
		batch; nothing else depends on it. Default is 128.

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

	ref_workers: int or None, optional
		The reference worker processes. None means
		`references.default_workers()`, and 0 draws them in this process.
		Default is None.

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
		head_rows=DEFAULT_HEAD_ROWS, seqs_per_step="auto", mem_budget_gb=12.0,
		compile="auto", forward_stats="auto", warning_threshold=1e-3,
		ref_workers=None, deterministic=False):

		if output not in OUTPUTS:
			raise ValueError("output must be either `counts` or `profile`.")

		if output == "profile":
			raise NotImplementedError("the fast engine attributes the count "
				"head only; attribute the profile head with tangermeme's "
				"deep_lift_shap")

		if precision not in PRECISIONS:
			raise ValueError("precision must be one of {}, got {!r}".format(
				PRECISIONS, precision))

		if not isinstance(deterministic, bool):
			raise ValueError("deterministic must be True or False, got "
				"{!r}".format(deterministic))

		if (isinstance(mem_budget_gb, bool)
				or not isinstance(mem_budget_gb, numbers.Real)
				or not mem_budget_gb > 0):
			raise ValueError("mem_budget_gb must be a positive number, got "
				"{!r}".format(mem_budget_gb))

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
		self.head_rows = _positive_int(head_rows, "head_rows")
		if seqs_per_step != "auto":
			seqs_per_step = _positive_int(seqs_per_step, "seqs_per_step",
				"'auto' or ")

		self.seqs_per_step = seqs_per_step

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

		if ref_workers is not None and (isinstance(ref_workers, bool)
				or not isinstance(ref_workers, numbers.Integral)
				or ref_workers < 0):
			raise ValueError("ref_workers must be None or a non-negative "
				"integer, got {!r}".format(ref_workers))

		self.ref_workers = None if ref_workers is None else int(ref_workers)
		self._cotangents = {}
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

		x_tokens, ref_tokens, K = self._tokens(X, references)
		N, L = x_tokens.shape
		dtype = self.weights.dtype
		m = torch.empty(N, K, 4, L, dtype=dtype)
		y_x = torch.empty(N, dtype=dtype)
		y_r = torch.empty(N, K, dtype=dtype)
		deltas = torch.empty(N, K, dtype=dtype)

		def consume(a, b, out):
			n = b - a
			m[a:b], y_x[a:b] = out.values[:n], out.y_x[:n]
			y_r[a:b], deltas[a:b] = out.y_r[:n], out.delta[:n]

		debug = self._execute(x_tokens, ref_tokens, K, None, consume)
		debug["deltas"] = delta_summary(deltas, self.warning_threshold)
		return RawResult(m, y_x, y_r, deltas, debug)

	def run(self, X, start, end, references=None, keep_references=None,
		verbose=False):
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

		keep_references: sequence of int or None, optional
			Increasing indices of sequences whose references the run keeps,
			as it used them, in `Result.references`; `audit` injects them
			into tangermeme's `deep_lift_shap`. Default is None.

		verbose: bool, optional
			Whether to show a progress bar. Default is False.


		Returns
		-------
		result: Result
			The attributions (N, 4, end - start), deltas, outputs, a record of
			the run, and the kept references.
		"""

		x_tokens, ref_tokens, K = self._tokens(X, references)
		N, L = x_tokens.shape
		if not 0 <= operator.index(start) < operator.index(end) <= L:
			raise ValueError("the window [{}, {}) must be non-empty and "
				"inside [0, {})".format(start, end, L))

		start, end = int(start), int(end)
		keep = None
		if keep_references is not None:
			indices = _keep_indices(keep_references, N)
			keep = (indices, numpy.empty((len(indices), K, L), numpy.uint8))

		dtype = self.weights.dtype
		attr = torch.empty(N, 4, end - start, dtype=dtype)
		y_x = torch.empty(N, dtype=dtype)
		y_r = torch.empty(N, K, dtype=dtype)
		deltas = torch.empty(N, K, dtype=dtype)

		from tqdm import tqdm
		progress = tqdm(total=N, unit="seq", disable=not verbose)

		def consume(a, b, out):
			n = b - a
			attr[a:b] = mean_over_references(out.values[:n], start, L)
			y_x[a:b], y_r[a:b] = out.y_x[:n], out.y_r[:n]
			deltas[a:b] = out.delta[:n]
			progress.update(n)

		try:
			meta = self._execute(x_tokens, ref_tokens, K, (start, end),
				consume, keep)
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

		references = None if keep is None else keep[1]
		return Result(attr, deltas, y_x, y_r, meta, references)

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

	def _execute(self, x_tokens, ref_tokens, K, window, consume, keep=None):
		"""Run every step of a `raw` (window None) or `run` call under the
		engine's precision mode.

		consume(a, b, out) receives each step's results for the sequences
		[a, b), in order; out has S >= b - a rows and its buffers are reused,
		so consume copies what it keeps. keep, (sorted indices, a (len, K, L)
		uint8 array), receives those sequences' references as the run uses
		them. Returns the call's record.

		The reference pool starts before the count gradient and the warm-up,
		so the references are drawn while the warm-up compiles and
		cudnn.benchmark times its algorithms. References depend only on the
		sequence and k, never on the chunking, so the results are those of a
		pool started after the warm-up. If the warm-up halves S, the pool's
		chunks, cut for the planned S, are re-cut (`_recut`).
		"""

		N, L = x_tokens.shape
		cuda = self.weights.device.type == "cuda"
		began = time.perf_counter()
		with contextlib.ExitStack() as stack:
			if cuda:
				stack.enter_context(torch.cuda.device(self.weights.device))

			flags = stack.enter_context(precision_mode(self.precision,
				deterministic=self.deterministic,
				inductor_cache=self.compile != "none"))
			S = self.step_size(N, L, K)
			record = self._describe(N, L, K, S)
			record["precision_flags"] = flags
			chunks = stack.enter_context(self._reference_chunks(x_tokens,
				ref_tokens, K, S, record, keep))
			G0 = self._counts_cotangent(L)
			if cuda:
				planned = S
				S = self._warm_up(S, K, L, G0, window, record)
				if S != planned:
					chunks = _recut(chunks, S)

				self._steps_cuda(chunks, x_tokens, K, L, S, G0, window,
					consume, record)
			else:
				self._steps_cpu(chunks, x_tokens, K, S, G0, window, consume,
					record)

		record["seconds"] = time.perf_counter() - began
		record["pairs_per_s"] = N * K / record["seconds"]
		return record

	def _device_step(self, tokens, S, K, G0, window):
		"""One step on the engine's device: tokens (S + S*K, L) uint8 -> the
		step's results, on the device.

		Eager: the one-hot, iconv, the count head and iconv's input gradient.
		Compiled on CUDA: the forward (fwd_c), the backward (bwd_c) and, for
		`run`, the epilogue (epi_c). Each large intermediate is released as
		soon as the next stage no longer needs it.
		"""

		w = self.weights
		X1h = _references.tokens_to_onehot(tokens, w.dtype)
		z0 = stem_forward(X1h, w.W0, w.b0)
		hL, saved = self.fwd_c(z0, w.blocks, w.rs, w.eps, self.forward_stats)
		y = head_forward(hL, w, output=self.output, group=self.group)
		del hL
		Gz = self.bwd_c(G0, saved, z0, w.blocks, w.rs, S, K)
		del saved, z0
		m = input_multipliers(Gz, X1h[S:], w.W0)
		del Gz
		y_x, y_r = y[:S], y[S:].view(S, K)
		if window is None:
			delta = convergence_deltas(m, X1h, y_x, y_r, S, K)
			return _StepOutputs(y_x, y_r, delta, m.view(S, K, *m.shape[1:]))

		hyp, delta = self.epi_c(m, X1h, y_x, y_r, window[0], window[1], S, K)
		return _StepOutputs(y_x, y_r, delta, hyp)

	def _steps_cpu(self, chunks, x_tokens, K, S, G0, window, consume, record):
		"""The steps on the CPU, one after the other."""

		for a, b, refs in chunks:
			joint = self._joint_tokens(x_tokens[a:b], refs, S)
			tokens = torch.from_numpy(joint).to(self.weights.device)
			consume(a, b, self._device_step(tokens, S, K, G0, window))
			record["steps"] += 1
			record["padded_peaks"] += S - (b - a)

	def _warm_up(self, S, K, L, G0, window, record):
		"""One step on random tokens before a CUDA run; returns the step size
		the run then uses.

		It compiles the passes for this S, lets cudnn.benchmark time its
		algorithms, and halves S on an out-of-memory error, all before the
		first step, while the reference pool draws the first chunks. The
		tokens are drawn on the device, so the warm-up copies nothing from
		the host, and its results are dropped. It waits for the device once,
		on an event.
		"""

		device = self.weights.device
		generator = torch.Generator(device=device)
		generator.manual_seed(0)
		began = time.perf_counter()
		while True:
			attempt = time.perf_counter()
			tokens = torch.randint(0, 4, (S + S * K, L), generator=generator,
				dtype=torch.uint8, device=device)
			failed = None
			try:
				self._device_step(tokens, S, K, G0, window)
				done = torch.cuda.Event(blocking=True)
				done.record()
				done.synchronize()
			except torch.cuda.OutOfMemoryError as exc:
				if S == 1:
					raise

				failed = str(exc)

			if failed is None:
				break

			del tokens
			S = self._after_oom(S, failed, record, "warm-up")

		gc.collect()  # frees the step that compiling kept alive (see _launch)
		record["compile_s"] = time.perf_counter() - attempt
		record["warmup_s"] = time.perf_counter() - began
		return S

	def _steps_cuda(self, chunks, x_tokens, K, L, S, G0, window, consume,
		record):
		"""The steps on CUDA, pipelined: step i is queued before the host
		waits for step i - 1 (see the class docstring)."""

		device = self.weights.device
		if window is None:
			values_shape = (4, L)
		else:
			values_shape = (4, window[1] - window[0])

		rings = _Rings(S, K, L, values_shape, self.weights.dtype, device)
		if self._copy_stream is None:
			self._copy_stream = torch.cuda.Stream(device)

		torch.cuda.reset_peak_memory_stats(device)
		base, first_S = torch.cuda.memory_allocated(device), S
		work = collections.deque()
		chunk_iter = iter(chunks)
		pending, finished, slot, recompiled = None, [], 0, False
		while True:
			if not work:
				chunk = next(chunk_iter, None)
				if chunk is None:
					break

				work.extend(_substeps(*chunk, S))

			a, b, refs = work.popleft()
			rings.fill(slot, x_tokens[a:b], refs, S)
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
				work = collections.deque(itertools.chain(_substeps(a, b,
					refs, S), *(_substeps(*w, S) for w in work)))
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
		"""Queue one step on the device without waiting: tokens in on the
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
		tokens = rings.device_tokens[slot][:rows]
		with torch.cuda.stream(self._copy_stream):
			tokens.copy_(rings.host_tokens[slot][:rows], non_blocking=True)
			copied = torch.cuda.Event()
			copied.record()

		compute.wait_event(copied)
		out = self._device_step(tokens, S, K, G0, window)
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
	def _joint_tokens(x_tokens, ref_tokens, S):
		"""The step's rows ``[x_0..x_{S-1}, r_00..r_{S-1,K-1}]`` as uint8
		tokens (S + S*K, L), padded with the last sequence.

		x_tokens (n, L) and ref_tokens (n, K, L) are tokens. With n < S, the
		last sequence and its references are repeated to fill the step.
		"""

		n = len(x_tokens)
		if n < S:
			keep = numpy.concatenate([numpy.arange(n), numpy.full(S - n,
				n - 1)])
			x_tokens, ref_tokens = x_tokens[keep], ref_tokens[keep]

		return numpy.concatenate([x_tokens, ref_tokens.reshape(-1,
			x_tokens.shape[-1])])

	@contextlib.contextmanager
	def _reference_chunks(self, x_tokens, ref_tokens, K, S, record, keep=None):
		"""An iterator of (a, b, references of the sequences [a, b) as
		(b - a, K, L) tokens), over chunks of S sequences in order.

		Injected references are sliced. Otherwise a `RefProducer` draws them,
		ahead of their use, and closes its pool on exit; its first chunks are
		submitted on entry, so the pool works while the caller does other
		things. With keep (sorted indices, an output array), those sequences'
		references are copied into the array as they pass.
		"""

		N = len(x_tokens)
		if keep is not None:
			record["kept_references"] = len(keep[0])

		if ref_tokens is not None:
			record["references"] = "injected"
			if keep is not None:
				keep[1][:] = ref_tokens[keep[0]]

			yield ((a, min(a + S, N), ref_tokens[a:a + S])
				for a in range(0, N, S))
			return

		record["references"] = "produced"
		with _references.RefProducer(self.ref_workers) as producer:
			chunks = producer.chunks(x_tokens, K, self.random_state, S)
			yield chunks if keep is None else _keeping(chunks, *keep)
			record["reference_stats"] = dict(producer.stats)

	def _tokens(self, X, references):
		"""Check X and the references as `deep_lift_shap` does, and encode
		them as uint8 tokens. Returns X's tokens (N, L), the references'
		tokens (N, K, L) or None, and K.

		X must be a one-hot with exactly one base per position; the
		references may have all-zero (N) columns. The tokens are lossless, so
		the one-hot rows of a step equal the inputs.
		"""

		x = torch.as_tensor(X)
		if x.dim() != 3 or x.shape[1] != 4:
			raise ValueError("X must have shape (N, 4, L), got {}".format(
				tuple(x.shape)))

		if x.shape[0] == 0:
			raise ValueError("deep_lift_shap requires at least one example; "
				"got X with shape[0] == 0.")

		x_tokens = _references.onehot_to_tokens(x)
		if bool((x_tokens == _references.N_TOKEN).any()):
			raise ValueError("X must be one-hot encoded and cannot have "
				"unknown characters (all-zero columns)")

		if references is None:
			return x_tokens, None, self.n_shuffles

		r = torch.as_tensor(references)
		N, _, L = x.shape
		if (r.dim() != 4 or r.shape[0] != N or r.shape[1] < 1
				or tuple(r.shape[2:]) != (4, L)):
			raise ValueError("references must have shape ({}, K >= 1, 4, {}), "
				"got {}".format(N, L, tuple(r.shape)))

		return x_tokens, _references.onehot_to_tokens(r), r.shape[1]

	def _counts_cotangent(self, length):
		"""`counts_cotangent` at the input length, computed once per
		length."""

		if length not in self._cotangents:
			self._cotangents[length] = counts_cotangent(self.weights,
				group=self.group, length=length, rows=self.head_rows)

		return self._cotangents[length]

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
			"head_rows": self.head_rows,
			"n_sequences": N,
			"length": L,
			"n_shuffles": K,
			"pairs": N * K,
			"random_state": self.random_state,
			"seqs_per_step": S,
			"mem_budget_gb": self.mem_budget_gb,
			"steps": 0,
			"padded_peaks": 0,
			"host_syncs": 0,
			"oom": [],
			"compile_s": None,
			"warmup_s": None,
			"steady_pairs_per_s": None,
			"step_peak_bytes": None,
			"step_estimate_bytes": None,
		}


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
