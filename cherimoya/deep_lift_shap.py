# deep_lift_shap.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
DeepLIFT rules for the layer of a Cherimoya model that needs one.

`tangermeme.deep_lift_shap.deep_lift_shap` attaches a rule to each module
type it knows how to correct and treats everything else as linear.
`FusedDilatedConvNorm` is neither in that table nor linear, so attributing
a Cherimoya model without registering it is not a performance question but
a correctness one: the attributions come back with no guarantee that they
sum to the change in the prediction. The non-linear part of the profile
head, in `ProfileWrapper`, is built from modules tangermeme already has
rules for.

Pass `attribution_ops()` as `additional_nonlinear_ops` and the whole model
is covered::

    from tangermeme.deep_lift_shap import deep_lift_shap

    from cherimoya import Cherimoya
    from cherimoya import ProfileWrapper
    from cherimoya.deep_lift_shap import attribution_ops

    model = Cherimoya.load("model.torch", compile=False)
    X_attr = deep_lift_shap(ProfileWrapper(model), X, references=references,
        additional_nonlinear_ops=attribution_ops())

Build the model with `compile=False` when you intend to attribute it. The
backward hooks DeepLIFT installs replace gradients, which Inductor cannot
trace through, so the compiled model graph-breaks and pays for the guards
without getting the fusion.
"""

from __future__ import annotations

import torch

from tangermeme._deep_lift_utils import _layer_normalization_grad
from tangermeme._deep_lift_utils import _layer_normalization_stats
from tangermeme._deep_lift_utils import _layer_normalization_terms

from .cheri import CONV_NORM_EPS
from .cheri import FusedDilatedConvNorm
from .cheri import _cheri_conv


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
