cherimoya.wrappers
==================

.. module:: cherimoya.wrappers

Thin :class:`torch.nn.Module` wrappers around a :class:`~cherimoya.Cherimoya`
that expose a single tensor from its ``(profile, log-count)`` output, so the
model can be dropped straight into attribution and design tools that expect a
one-tensor forward pass, plus :class:`ControlWrapper`, which supplies a
zero control track to models that expect one. All four are re-exported from
the top-level ``cherimoya`` namespace.


ControlWrapper
--------------

.. autoclass:: ControlWrapper
   :members: forward, signal_groups
   :undoc-members:
   :show-inheritance:

   .. rubric:: Constructor

   .. automethod:: __init__

Returns the model's full ``(profile, log-count)`` output, synthesizing an
all-zero control track when the model expects one but none is passed. It is
the inner wrapper that the output wrappers below are layered on top of — for
example ``LogCountWrapper(ControlWrapper(model))`` — and is what
``cherimoya attribute`` and ``cherimoya marginalize`` use so a model can be
called with the sequence alone. A drop-in port of
``bpnetlite.bpnet.ControlWrapper``.

It forwards ``signal_groups`` from the model it wraps, which is what lets
:class:`ExpectedCountsWrapper` be layered on top of it —
``torch.nn.Module`` does not forward attribute lookups to submodules, so
without it that composition raised ``AttributeError``.


ProfileWrapper
--------------

.. autoclass:: ProfileWrapper
   :members: forward
   :undoc-members:
   :show-inheritance:

   .. rubric:: Constructor

   .. automethod:: __init__

Returns the mean-centered profile logits weighted by their own softmax and
summed across positions, collapsing the predicted profile into a single
shape-sensitive number per example. This is the wrapper used by
``cherimoya attribute`` when ``output`` is ``"profile"`` (see
:doc:`../tutorials/attribution`). It is a drop-in port of
``bpnetlite.bpnet.ProfileWrapper`` so that attribution does not require
bpnet-lite. As with :class:`LogCountWrapper`, pair it with
:class:`ControlWrapper` for models trained with control tracks.

For a model with several signal groups the default flattens every group
together, so one softmax spans all of them and the output summarizes the
whole profile head. ``ProfileWrapper(model, group=i)`` restricts the
calculation to group ``i``'s channels, which is how to attribute one
modality's profile shape (see :doc:`../multi_task`).


LogCountWrapper
---------------

.. autoclass:: LogCountWrapper
   :members: forward
   :undoc-members:
   :show-inheritance:

   .. rubric:: Constructor

   .. automethod:: __init__

Returns the model's per-group log-count predictions. This is the wrapper
used by ``cherimoya attribute`` when ``output`` is ``"counts"`` (see
:doc:`../tutorials/attribution`). Pair it with :class:`ControlWrapper` when
attributing a model that was trained with control tracks, so that zero
controls are supplied automatically.

``LogCountWrapper(model, group=i)`` returns only group ``i``'s log-count, with
shape ``(batch_size, 1)``, for attributing one modality of a multi-group
model. Without ``group``, tangermeme's ``deep_lift_shap`` attributes its
default ``target=0`` — group 0 alone — while ``saturation_mutagenesis``
averages over every group.


ExpectedCountsWrapper
---------------------

.. autoclass:: ExpectedCountsWrapper
   :members: forward
   :undoc-members:
   :show-inheritance:

   .. rubric:: Constructor

   .. automethod:: __init__

Combines the profile and log-count heads into the expected number of reads
at each position. Within each signal group the profile channels and
positions are softmaxed jointly and scaled by ``expm1`` of that group's
log-count, so the expected counts summed over the whole group (e.g. both
strands of a stranded pair) equal the predicted count for the group.
