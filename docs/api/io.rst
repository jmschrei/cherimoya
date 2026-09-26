cherimoya.io
============

.. module:: cherimoya.io

Data loading utilities for training Cherimoya. The dataset is the
peak/negative mixture sampler; the function-style entry point
:func:`PeakGenerator` is the typical way to build it. It returns a
``DataLoader``, and :func:`cherimoya.training.fit` takes the dataset
inside it (``PeakGenerator(...).dataset``).


PeakGenerator
-------------

.. autofunction:: PeakGenerator


Signal group helpers
--------------------

.. autofunction:: normalize_signal_groups

.. autofunction:: channel_permutation_from_groups


PeakNegativeSampler
-------------------

.. autoclass:: PeakNegativeSampler
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Constructor

   .. automethod:: __init__

The sampler is fully deterministic given ``random_state`` and the
epoch number. ``__getitem__(idx)`` is a pure function of ``idx`` and
the current epoch, so ``num_workers > 1`` yields the same batch
sequence as ``num_workers = 1`` and two runs with the same seed
produce bit-identical training data. An index may also be an
``(epoch, index)`` pair, as :class:`ShardedEpochSampler` yields, which
names the epoch directly.


ShardedEpochSampler
-------------------

.. autoclass:: ShardedEpochSampler
   :members: set_epoch
   :show-inheritance:

This is the sampler :class:`cherimoya.training.CherimoyaModule` builds
its training loader with, on one device or several.
