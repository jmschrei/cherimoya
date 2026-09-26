cherimoya.training
==================

.. module:: cherimoya.training

Training a Cherimoya model with PyTorch Lightning. :func:`fit` builds a
``lightning.Trainer`` around a :class:`CherimoyaModule`, trains on one
device or, with DDP, on several, and writes the best and final EMA
checkpoints and ``{name}.metrics.csv`` next to ``model.name``. The
metrics file's columns are listed in :doc:`../cli`.


fit
---

.. autofunction:: fit


CherimoyaModule
---------------

.. autoclass:: CherimoyaModule
   :show-inheritance:
