cherimoya.training
==================

.. module:: cherimoya.training

Training a Cherimoya model with PyTorch Lightning. :func:`fit` builds a
``lightning.Trainer`` around a :class:`CherimoyaModule`, trains on one
device or, with DDP, on several, and writes the best and final EMA
checkpoints and the training logs ``{name}.log`` and
``{name}.detailed.log`` next to ``model.name``.

:doc:`../tutorials/python_api` walks through a training run, what the
returned trainer holds, and what changes on several GPUs;
:ref:`training on several devices <cli-several-devices>` covers GPU
selection, memory and SLURM.


fit
---

.. autofunction:: fit


CherimoyaModule
---------------

.. autoclass:: CherimoyaModule
   :show-inheritance:
