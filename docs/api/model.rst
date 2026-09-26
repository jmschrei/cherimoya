cherimoya.cherimoya
===================

.. module:: cherimoya.cherimoya

The top-level module exposes the model and the EMA helper used during
training. The Cheri Block, the fused conv+norm dispatcher, and the
inference megakernel internals are in :doc:`cheri`.


Cherimoya
---------

.. autoclass:: Cherimoya
   :members: forward, save, load
   :undoc-members:
   :show-inheritance:

   .. rubric:: Constructor

   .. automethod:: __init__


EMA
---

.. autoclass:: EMA
   :members: update, apply_shadow, restore
   :undoc-members:

   .. rubric:: Constructor

   .. automethod:: __init__

Used internally by :func:`cherimoya.training.fit`: a shadow copy of every
floating-point parameter (decay 0.999 by default) is updated after
every optimizer step, swapped in for validation
(``apply_shadow``/``restore``), and applied to the saved checkpoints
at the end of each improvement and at the very end of training.

The shadow weights are kept on the same device as the model. They
are not part of the ``state_dict`` and are not saved with
``Cherimoya.save``; they exist only for the duration of the training
run.
