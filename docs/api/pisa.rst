cherimoya.pisa
==============

.. module:: cherimoya.pisa

PISA attributions of a Cherimoya model's profile logits: for every output
position of one profile channel, the DeepLIFT/SHAP attributions of the
logit there to every input nucleotide. The function registers
:func:`cherimoya.deep_lift_shap.attribution_ops` itself, and takes a fast
path where it applies. See :doc:`../tutorials/attribution` for when it
does and how close the two paths are.


pisa
----

.. autofunction:: pisa

It takes the model itself and the profile channel to attribute::

    from cherimoya import Cherimoya
    from cherimoya.pisa import pisa

    model = Cherimoya.load("model.torch", compile=False)
    X_attr = pisa(model, X, channel=0, random_state=0)

``X_attr[i, j]`` holds the attributions of sequence ``i`` for the logit
at output position ``j``, which is input position ``j + model.trimming``.
The result holds ``out_length * 4 * length`` values per sequence: 34 MB in
float32 for a 2,114 bp input and the default trimming of a 9-layer model.

Build the model with ``compile=False`` when you intend to attribute it.
The backward hooks DeepLIFT installs replace gradients, which Inductor
cannot trace through, so a compiled model graph-breaks and pays for the
guards without getting the fusion.
