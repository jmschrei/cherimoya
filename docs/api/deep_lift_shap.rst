cherimoya.deep_lift_shap
========================

.. module:: cherimoya.deep_lift_shap

DeepLIFT rules for the layer of a Cherimoya model that needs one.
``tangermeme.deep_lift_shap.deep_lift_shap`` attaches a rule to each module
type it knows and treats everything else as linear, so attributing a
Cherimoya model without registering this one is a correctness question
rather than a performance one — the attributions come back with no
guarantee that they sum to the change in the prediction. The non-linear
part of the profile head, in :class:`~cherimoya.ProfileWrapper`, is a
``torch.nn.Softmax`` and a tangermeme ``BilinearOp``, which tangermeme
already has rules for.

This is only needed for DeepLIFT/SHAP. ``cherimoya attribute`` registers
the rules itself when ``algorithm`` is ``"deep_lift_shap"``, its default.
Saturation mutagenesis makes forward passes only and registers nothing.

Requires ``tangermeme >= 1.5.0``, which is where the closed-form
normalization rule that :func:`conv_norm_op` reuses was added.


attribution_ops
---------------

.. autofunction:: attribution_ops

Start here. It returns the rules in the dictionary
``additional_nonlinear_ops`` expects::

    from tangermeme.deep_lift_shap import deep_lift_shap

    from cherimoya import Cherimoya
    from cherimoya import ProfileWrapper
    from cherimoya.deep_lift_shap import attribution_ops

    model = Cherimoya.load("model.torch", compile=False)
    X_attr = deep_lift_shap(ProfileWrapper(model), X, references=references,
        additional_nonlinear_ops=attribution_ops())

Build the model with ``compile=False`` when you intend to attribute it. The
backward hooks DeepLIFT installs replace gradients, which Inductor cannot
trace through, so a compiled model graph-breaks and pays for the guards
without getting the fusion.


conv_norm_op
------------

.. autofunction:: conv_norm_op
