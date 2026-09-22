cherimoya.deep_lift_shap
========================

.. module:: cherimoya.deep_lift_shap

DeepLIFT rules for the two layers of a Cherimoya model that need one.
``tangermeme.deep_lift_shap.deep_lift_shap`` attaches a rule to each module
type it knows and treats everything else as linear, so attributing a
Cherimoya model without registering these is a correctness question rather
than a performance one — the attributions come back with no guarantee that
they sum to the change in the prediction.

This is only needed for DeepLIFT/SHAP. ``cherimoya attribute`` and the
saturation-mutagenesis path in :doc:`../tutorials/attribution` make forward
passes only and register nothing.

Requires ``tangermeme >= 1.5.0``, which is where the closed-form
normalization rule that :func:`conv_norm_op` reuses was added.


attribution_ops
---------------

.. autofunction:: attribution_ops

Start here. It returns both rules in the dictionary
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
