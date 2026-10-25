# deep_lift_shap.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
DeepLIFT/SHAP attributions of a Cherimoya model, and the DeepLIFT rule for
the layer of a Cherimoya model that needs one.

`deep_lift_shap` attributes a Cherimoya model's log-counts or profile::

    from cherimoya import Cherimoya
    from cherimoya.deep_lift_shap import deep_lift_shap

    model = Cherimoya.load("model.torch", compile=False)
    X_attr = deep_lift_shap(model, X, output="logcounts", random_state=0)

`tangermeme.deep_lift_shap.deep_lift_shap` attaches a rule to each module
type it knows how to correct and treats everything else as linear.
`FusedDilatedConvNorm` is neither in that table nor linear, so attributing
a Cherimoya model without registering it is not a performance question but
a correctness one: the attributions come back with no guarantee that they
sum to the change in the prediction. The non-linear part of the profile
head, in `ProfileWrapper`, is built from modules tangermeme already has
rules for. `deep_lift_shap` registers the rule itself; a tangermeme function
called directly, such as `tangermeme.pisa.pisa`, takes `attribution_ops()`
as `additional_nonlinear_ops`.

Build the model with `compile=False` when you intend to attribute it. The
backward hooks DeepLIFT installs replace gradients, which Inductor cannot
trace through, so the compiled model graph-breaks and pays for the guards
without getting the fusion.
"""

from __future__ import annotations

import time
import warnings

import numpy
import torch
import tangermeme.deep_lift_shap

from tangermeme._deep_lift_utils import _layer_normalization_grad
from tangermeme._deep_lift_utils import _layer_normalization_stats
from tangermeme._deep_lift_utils import _layer_normalization_terms
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.utils import validate_input

from ._deep_lift_shap_engine import Engine
from .cheri import CONV_NORM_EPS
from .cheri import FusedDilatedConvNorm
from .cheri import _cheri_conv
from .cherimoya import Cherimoya
from .wrappers import ControlWrapper
from .wrappers import LogCountWrapper
from .wrappers import ProfileWrapper


def conv_norm_op(module, grad_input, grad_output):
	"""The DeepLIFT rule for `FusedDilatedConvNorm`.

	The fused op is a normalization applied to a depthwise convolution.
	The convolution is linear, so only the normalization needs a rule, and
	that rule already exists in closed form for `torch.nn.LayerNorm` --
	which is the same operation the fused kernel performs, over the whole
	(length, channels) plane of each example.

	Reaching it without giving up the kernel takes three steps. Recompute
	the convolution from the input the forward hook cached; evaluate the
	closed-form normalization multiplier at that output rather than at the
	module's input; then push the result back through the convolution with
	autograd, which is exact because the convolution is linear. The result
	is the same attribution as decomposing the model into `F.conv1d` and a
	`torch.nn.LayerNorm`, with the fused kernel still running the forward
	and backward passes.

	Registering nothing is the alternative, and it is not safe in general.
	The op is genuinely non-linear -- the normalization divides out the
	scale of its input, so doubling the input leaves the output nearly
	unchanged rather than doubling it -- so whether it can be treated
	as linear depends on how far the normalization's statistics move
	between an example and its reference, which is a property of the model
	and the inputs rather than of the layer. A 9-layer model over 2114bp
	reduces over 270,000 elements per statistic and barely moves them; a
	single block over a short window does not, and leaves a convergence
	delta a quarter of the size of the prediction.

	The cost over registering nothing is one extra convolution forward and
	one backward per call, against `integrated_gradients_op`'s K of each.

	The rule is tangermeme's `_layer_normalization_helper` over the
	trailing two axes with no affine weight, called through the three
	tensor functions it is built from. Requires tangermeme >= 1.6.0, the
	first release with those functions.


	Parameters
	----------
	module: FusedDilatedConvNorm
		The module being corrected. tangermeme's backward hook has put
		`module.input` on it for the forward call being unwound, holding
		the observed batch concatenated with the reference batch along the
		first axis.

	grad_input: tuple of torch.Tensor
		The gradient with respect to the module's input that torch would
		otherwise propagate. Unused; the closed form is evaluated
		everywhere and needs no fallback.

	grad_output: tuple of torch.Tensor
		The gradient of the model output with respect to this module's
		output.


	Returns
	-------
	grad_input: tuple of one torch.Tensor
		The replacement gradient with respect to the module's input, i.e.
		the upstream gradient propagated through the DeepLIFT multiplier
		for this module rather than through its ordinary local gradient.
	"""

	X = module.input

	# Only `F.conv1d` is re-entered here, and hooks attach to modules
	# rather than to functions, so nothing needs to be switched off first.
	with torch.enable_grad():
		X_ = X.detach().requires_grad_()
		y = _cheri_conv(X_, module.conv_weight, module.dilation)

		y_x, y_ref = y.detach().chunk(2)
		mu, v = _layer_normalization_stats(y_x, 2, CONV_NORM_EPS)
		mu_ref, v_ref = _layer_normalization_stats(y_ref, 2, CONV_NORM_EPS)
		terms = _layer_normalization_terms(y_x, mu, v, y_ref, mu_ref, v_ref, 2)
		multipliers = torch.cat([_layer_normalization_grad(g, *terms, 2)
			for g in grad_output[0].chunk(2)])

		grad, = torch.autograd.grad(y, X_, grad_outputs=multipliers)

	return (grad,)


# tangermeme clones only the activations a rule declares it reads, so saying
# so here saves one clone of every block's input per forward pass. A rule
# that declares nothing gets both the input and the output, which is correct
# but larger.
conv_norm_op._reads = ("input",)


def attribution_ops() -> dict:
	"""The rules DeepLIFT needs to attribute a Cherimoya model correctly.

	Pass the result as `additional_nonlinear_ops` to
	`tangermeme.deep_lift_shap.deep_lift_shap` or
	`tangermeme.pisa.pisa`. A fresh dictionary is returned each call, so
	adding a rule of your own to it does not affect the next caller.

	One entry: `FusedDilatedConvNorm` gets `conv_norm_op`, described
	above. The profile head needs none. `ProfileWrapper` multiplies its
	logits by their own softmax through a `torch.nn.Softmax` and a
	tangermeme `BilinearOp`, and tangermeme has a rule for each. Do not add
	a rule for `_ProfileLogitScaling`, the module holding those two: it
	would take the place of both, and an elementwise rule cannot represent
	a softmax, which couples every position to every other.


	Returns
	-------
	ops: dict
		A mapping from module type onto the function that corrects it.
	"""

	return {
		FusedDilatedConvNorm: conv_norm_op,
	}


_CHECK_SEQUENCES = 4
"""`deep_lift_shap` attributes the first this many sequences with the default
path before it decides on the fast path: half to pay the default path's
first-call costs, half to time it, and all to check the fast path against."""

_FAST_MIN_SECONDS = 30.0
"""`deep_lift_shap` takes the fast path only for a job that the default path
would take at least this many seconds to finish. On a GPU the fast path
warms up the first time a process runs it, compiling its passes, which took
23 seconds on an H200 with empty compile caches and from 7 to 23 with warm
ones in our measurements."""

_CHECK_PEARSON_MIN = 0.9999
_CHECK_REL_L2_MAX = 1e-2
"""The fast path's attributions of a sequence agree with the default path's
when their Pearson correlation is at least `_CHECK_PEARSON_MIN` and their
relative L2 distance at most `_CHECK_REL_L2_MAX`."""

_SEED_RANGE = 9_999_999
"""`deep_lift_shap` draws a seed from [0, _SEED_RANGE) when `random_state`
is None and it may take the fast path."""


def _make_wrapper(model, output, group):
	"""`LogCountWrapper` or `ProfileWrapper` over `ControlWrapper(model)`,
	the module `cherimoya attribute` attributes."""

	if output not in ("logcounts", "profile"):
		raise ValueError("output must be either `logcounts` or `profile`.")

	if output == "logcounts" and group is None and len(model.signal_groups) > 1:
		raise ValueError("deep_lift_shap attributes one output; set `group` "
			"to one of the model's {} signal groups".format(
			len(model.signal_groups)))

	inner = ControlWrapper(model)
	if output == "logcounts":
		return LogCountWrapper(inner, group=group)

	return ProfileWrapper(inner, group=group)


def _agreement(ours, stock):
	"""The lowest Pearson correlation and the highest relative L2 distance,
	||ours - stock|| / ||stock||, between the attributions of each sequence
	(n, 4, length), in float64. A NaN in either gives NaN."""

	a, b = ours.double().flatten(1), stock.double().flatten(1)
	rel = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)
	ac = a - a.mean(dim=1, keepdim=True)
	bc = b - b.mean(dim=1, keepdim=True)
	r = (ac * bc).sum(dim=1) / (ac.norm(dim=1) * bc.norm(dim=1)).clamp_min(
		1e-300)
	return float(r.min()), float(rel.max())


def deep_lift_shap(model, X, output="logcounts", group=0, batch_size=32,
	references=dinucleotide_shuffle, n_shuffles=20, hypothetical=False,
	warning_threshold=0.001, dtype=None, device=None, random_state=None,
	verbose=False, fast=True):
	"""DeepLIFT/SHAP attributions of a Cherimoya model's log-counts or
	profile.

	The model is attributed as `cherimoya attribute` attributes it: wrapped
	in `ControlWrapper`, which gives it all-zero control tracks, and then in
	`LogCountWrapper` or `ProfileWrapper` for `group`, with
	`attribution_ops()` registered for its fused convolution and
	normalization.

	There are two paths. The default path is tangermeme's `deep_lift_shap`,
	which attributes through backward hooks on the model's modules. The fast
	path computes the same attributions of the log-counts from a forward
	and a backward pass written out for Cherimoya's architecture, with
	tangermeme's DeepLIFT rules and references: each sequence is forwarded
	once rather than once per reference, and the backward pass runs over the
	sequence half of each sequence-reference pair only. It runs several
	times faster on a GPU. The two paths agree up to floating-point
	rounding, which on a GPU is the rounding of TF32, which ``import
	cherimoya`` turns on for matrix products.

	The fast path is used when `fast` is True, `output` is "logcounts",
	`model` is a `Cherimoya` that runs in float32, `X` is a tensor of more
	than four sequences, and `references` is `dinucleotide_shuffle` or a
	tensor. All of `X` and the references are then checked as tangermeme
	checks them, so an unknown character raises before either path runs,
	and the first four sequences are attributed with the default path, the
	last two of them timed. When the whole job would take the default path less
	than thirty seconds at that rate, the default path attributes the rest,
	since the fast path's first compile on a GPU can take longer than it
	saves. Otherwise the fast path attributes every sequence, and its
	attributions of the first four are compared with the default path's.
	When they disagree beyond rounding, with a Pearson correlation below
	0.9999 or a relative L2 distance above 1e-2 on any of them, or when the
	fast path raises, a `RuntimeWarning` gives the reason and the default
	path attributes the rest.

	Both paths draw the same references. With `random_state`, reference k of
	every sequence is the dinucleotide shuffle drawn with seed
	``random_state + k``, as tangermeme's `deep_lift_shap` draws it. With
	`random_state` None, a seed is drawn once per call when the fast path
	may be used.


	Parameters
	----------
	model: cherimoya.Cherimoya
		The model. It is not moved or changed.

	X: torch.Tensor, shape=(n, 4, length)
		The one-hot sequences, such as those `extract_loci` returns, without
		unknown characters.

	output: str, optional
		"logcounts", to attribute the log-count prediction of `group`, or
		"profile", to attribute the profile through `ProfileWrapper`.
		Default is "logcounts".

	group: int or None, optional
		The index into ``model.signal_groups`` of the group to attribute.
		None attributes every group: for "profile", their profiles together,
		and for "logcounts", only on a model with one group, since
		DeepLIFT/SHAP attributes one output. Default is 0.

	batch_size: int, optional
		The number of sequence-reference pairs per batch on the default
		path. The fast path sizes its steps from the memory free on the
		device. Default is 32.

	references: callable or torch.Tensor, optional
		`dinucleotide_shuffle`; another function that takes ``(X, n=...,
		random_state=...)`` and returns ``(n_examples, n, 4, length)``
		references, which uses the default path; or a tensor of shape
		``(n, n_references, 4, length)``. Default is `dinucleotide_shuffle`.

	n_shuffles: int, optional
		The number of references drawn per sequence when `references` is a
		function. Default is 20.

	hypothetical: bool, optional
		Whether to return the hypothetical attributions of every character
		at every position, rather than those multiplied by `X`. Default is
		False.

	warning_threshold: float or None, optional
		A `RuntimeWarning` is given when a sequence-reference pair's
		convergence delta exceeds this: per batch on the default path, once
		per call on the fast path. None gives no such warning. Default is
		0.001.

	dtype: str or torch.dtype or None, optional
		The dtype to compute in, or None for the model's. Only float32 uses
		the fast path. Default is None.

	device: str or torch.device or None, optional
		The device to compute on. None means "cuda" when CUDA is available
		and "cpu" otherwise. Default is None.

	random_state: int or None, optional
		The seed of the references. Default is None.

	verbose: bool, optional
		Whether to show progress bars. Default is False.

	fast: bool, optional
		Whether to use the fast path where it applies. False always uses
		tangermeme's `deep_lift_shap`. Default is True.


	Returns
	-------
	attributions: torch.Tensor, shape=(n, 4, length)
		The attributions, averaged over each sequence's references, on the
		CPU.
	"""

	wrapper = _make_wrapper(model, output, group)
	threshold = float("inf") if warning_threshold is None else warning_threshold
	given = isinstance(references, torch.Tensor)

	def default(X_, references_):
		return tangermeme.deep_lift_shap.deep_lift_shap(wrapper, X_,
			batch_size=batch_size, references=references_,
			n_shuffles=n_shuffles, hypothetical=True,
			warning_threshold=threshold,
			additional_nonlinear_ops=attribution_ops(), dtype=dtype,
			device=device, random_state=random_state, verbose=verbose)

	def default_part(a, b=None):
		return default(X[a:b], references[a:b] if given else references)

	def finish(attributions):
		return attributions if hypothetical else attributions * X.cpu()

	float32 = dtype in (None, "float32", torch.float32) and all(
		p.dtype == torch.float32 for p in model.parameters())
	n, half = _CHECK_SEQUENCES, _CHECK_SEQUENCES // 2
	if not (fast and output == "logcounts" and isinstance(model, Cherimoya)
			and float32 and isinstance(X, torch.Tensor) and len(X) > n
			and (references is dinucleotide_shuffle or given)):
		return finish(default(X, references))

	# The default path checks only the sequences it is given, so the whole
	# input is checked here, before either path runs, as tangermeme checks
	# it: an unknown character in a sequence the check does not attribute
	# raises the same error with fast=True as with fast=False.
	validate_input(X, "X", shape=(-1, -1, -1), ohe=True)
	if given:
		validate_input(references, "references", shape=(X.shape[0], -1,
			X.shape[1], X.shape[2]), ohe=True, allow_N=True, ohe_dim=-2)

	# Both paths draw reference k of every sequence with the seed
	# random_state + k, so they draw the same references only with a seed.
	if random_state is None:
		random_state = int(numpy.random.randint(0, _SEED_RANGE))

	checked = [default_part(0, half)]
	began = time.perf_counter()
	checked.append(default_part(half, n))
	seconds = time.perf_counter() - began
	checked = torch.cat(checked)

	if seconds * (len(X) - half) / (n - half) >= _FAST_MIN_SECONDS:
		try:
			if device is None:
				device = "cuda" if torch.cuda.is_available() else "cpu"

			engine = Engine(model, group=group, device=device,
				n_shuffles=n_shuffles, random_state=random_state,
				warning_threshold=warning_threshold)
			attributions = engine.run(X, references if given else None,
				verbose=verbose).attr
			pearson, rel_l2 = _agreement(attributions[:n], checked)
			if pearson >= _CHECK_PEARSON_MIN and rel_l2 <= _CHECK_REL_L2_MAX:
				return finish(attributions)

			reason = ("its attributions of the first {} sequences had a "
				"Pearson correlation as low as {:.6f} and a relative L2 "
				"distance as high as {:.3g} with the default path's".format(n,
				pearson, rel_l2))
		except Exception as exc:
			reason = "it raised {}: {}".format(type(exc).__name__,
				str(exc).strip().split("\n")[0])

		warnings.warn("deep_lift_shap is using the default path because the "
			"fast path {}. Pass fast=False to skip the fast path."
			.format(reason), RuntimeWarning, stacklevel=2)

	return finish(torch.cat([checked, default_part(n)]))
