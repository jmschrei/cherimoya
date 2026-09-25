# deep_lift_shap.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
DeepLIFT rules for the layers of a Cherimoya model that need one.

`tangermeme.deep_lift_shap.deep_lift_shap` attaches a rule to each module
type it knows how to correct and treats everything else as linear. Two of
Cherimoya's layers are neither in that table nor linear, so attributing a
Cherimoya model without registering them is not a performance question but
a correctness one: the attributions come back with no guarantee that they
sum to the change in the prediction.

Pass `attribution_ops()` as `additional_nonlinear_ops` and both are
covered::

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

from tangermeme._deep_lift_utils import _layer_normalization_helper
from tangermeme._deep_lift_utils import _nonlinear

from .cheri import CONV_NORM_EPS
from .cheri import FusedDilatedConvNorm
from .cheri import _cheri_conv
from .wrappers import _ProfileLogitScaling


class _ConvNormView:
	"""A stand-in normalization for `_layer_normalization_helper` to read.

	That helper takes four things off the module it is correcting: the
	input tangermeme's forward hook cached, `eps`, `normalized_shape`, and
	the affine weight. The last three are `torch.nn.LayerNorm`'s own
	attributes, so supplying them for a normalization that is not a module
	is a matter of naming, not of reimplementing a protocol.

	The fused op normalizes over the whole (length, channels) plane of each
	example and carries no affine weight, which is `normalized_shape` of
	the trailing two axes and a weight of None.


	Parameters
	----------
	y: torch.Tensor, shape=(2 * N, L, C)
		The convolution's output, with the observed batch concatenated
		with the reference batch along the first axis, which is the layout
		every DeepLIFT rule reads.
	"""

	def __init__(self, y: torch.Tensor):
		self.input = y
		self.eps = CONV_NORM_EPS
		self.normalized_shape = list(y.shape[1:])
		self.weight = None


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

	Requires tangermeme >= 1.5.0, which is where the closed-form
	normalization rule this reuses was added.


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

		multipliers, = _layer_normalization_helper(_ConvNormView(y.detach()),
			grad_input, grad_output, norm_type="layernorm")

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

	Two entries. `FusedDilatedConvNorm` gets `conv_norm_op`, described
	above. `_ProfileLogitScaling`, which the profile head multiplies its
	logits by their own softmax in, gets tangermeme's elementwise rescale
	rule -- it is elementwise and shape-preserving, which is what that rule
	requires. The count head does not need the second entry, but including
	it costs nothing, since a rule for a module the model does not contain
	is never dispatched.


	Returns
	-------
	ops: dict
		A mapping from module type onto the function that corrects it.
	"""

	return {
		FusedDilatedConvNorm: conv_norm_op,
		_ProfileLogitScaling: _nonlinear,
	}
