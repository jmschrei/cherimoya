Python API Tutorial
===================

This tutorial shows how to use the Cherimoya Python API to build a
model, train it, and generate predictions. For attribution, motif
analysis, and variant effect prediction see the dedicated tutorials.


Creating a model
----------------

``Cherimoya`` is a ``torch.nn.Module``:

.. code-block:: python

   from cherimoya import Cherimoya

   model = Cherimoya(
       n_filters=128,             # backbone width
       n_layers=9,                # number of Cheri Blocks (dilations 1, 2, ..., 256)
       signal_groups=[2],         # one stranded (+, -) group; see below
       n_control_tracks=0,        # number of control input tracks
       expansion=2,               # MLP expansion factor inside each Cheri Block
       residual_scale=0.15,       # fixed residual scale
       name="my_model",           # used for save filenames
       random_state=0,            # seeds the weight initialization
   ).cuda()

``signal_groups`` is the list of channel counts per signal group, one
group per biological modality. Examples:

* ``[1]`` — single unstranded track (e.g. ATAC). Default.
* ``[2]`` — one stranded ``(+, -)`` pair (e.g. BPNet-style ChIP).
  Two profile channels but one shared count prediction; the two
  strands swap places under reverse-complement augmentation.
* ``[1, 2]`` — co-train an unstranded ATAC head with a stranded TF
  head. Three profile channels, two count predictions. Groups are
  independent under RC: only the inner ``(+, -)`` channels swap, the
  ATAC channel stays put.

The full constructor signature, including ``trimming`` and
``verbose``, is in :doc:`../api/model`.


Input/output shapes
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Tensor
     - Shape
     - Description
   * - ``X`` (input)
     - ``(N, 4, in_window)``
     - One-hot encoded DNA over the input window. ``in_window`` is
       2114 by default.
   * - ``X_ctl`` (optional)
     - ``(N, n_control_tracks, in_window)``
     - Per-position control signal. Pass ``None`` when ``n_control_tracks
       == 0``.
   * - ``y_profile`` (output)
     - ``(N, sum(signal_groups), out_window)``
     - Predicted profile logits — one channel per signal channel.
       ``out_window`` is 1000 by default.
   * - ``y_counts`` (output)
     - ``(N, len(signal_groups))``
     - Predicted log counts — one per signal *group*. A stranded
       ``(+, -)`` group shares a single per-group count.

By default ``trimming = 46 + sum(2**i for i in range(n_layers))``,
which is 557 for the default 9-layer model and gives the 2114 → 1000
window pair.


Loading training data
---------------------

:func:`cherimoya.io.PeakGenerator` reads peaks, negatives, sequences,
and signal/control bigWigs, applies filtering and jitter, and returns
a ``torch.utils.data.DataLoader``. Training takes the dataset inside
it, a :class:`cherimoya.io.PeakNegativeSampler`, and builds its own
loader around it:

.. code-block:: python

   from cherimoya.io import PeakGenerator

   training_data = PeakGenerator(
       peaks="peaks.narrowPeak",
       negatives="negatives.bed",
       sequences="hg38.fa",
       signals=[["signal.+.bw", "signal.-.bw"]],   # one stranded group
       controls=None,                              # or list of bigWigs
       chroms=["chr2", "chr4", "chr5"],     # training chromosomes
       in_window=2114,
       out_window=1000,
       max_jitter=500,                      # peak-center jitter at training time
       negative_ratio=0.25,                 # n_negatives per n_peaks per epoch
       reverse_complement=True,             # augment with reverse complements
       random_state=0,                      # base seed; reproducible
       verbose=True,                        # print progress and filter counts
   ).dataset

Setting ``verbose=True`` prints per-step counts of filtered peaks and
filtered negatives, which is the easiest way to verify the loader is
seeing the data you expect.


Reproducible sampling
~~~~~~~~~~~~~~~~~~~~~

The underlying :class:`cherimoya.io.PeakNegativeSampler` is fully
deterministic given ``random_state``. ``__getitem__(idx)`` is a pure
function of ``idx`` and the current epoch, with no dependence on call
history.

* Each epoch yields exactly ``n_peaks + int(n_peaks * negative_ratio)``
  examples; every peak appears exactly once and the peak/negative
  interleaving is reproducible.
* Setting ``num_workers > 1`` produces the *same* sequence of batches
  as ``num_workers = 1``, just faster.
* Per-position jitter and reverse-complement flips are drawn from the
  per-epoch RNG, so two runs with the same seed produce bit-identical
  training data.

The sampler is only half of a reproducible run. Pass ``random_state``
to ``Cherimoya`` as well to fix the weight initialization, which is
what ``cherimoya fit`` does with the single seed in its JSON.


Preparing validation data
-------------------------

Validation data is loaded as a single block of tensors using
``tangermeme.io.extract_loci``:

.. code-block:: python

   from tangermeme.io import extract_loci

   valid_data = extract_loci(
       sequences="hg38.fa",
       signals=["signal.+.bw", "signal.-.bw"],
       loci="peaks.narrowPeak",
       chroms=["chr8", "chr20"],
       in_window=2114,
       out_window=1000,
       max_jitter=0,
       ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
   )

   X_valid, y_valid = valid_data
   # X_valid, y_valid, X_ctl_valid = valid_data   # with controls


Training
--------

:func:`cherimoya.training.fit` builds a PyTorch Lightning ``Trainer``
around a :class:`cherimoya.training.CherimoyaModule` and returns the
trainer after fitting. The module builds the optimizers and learning
rate schedules itself, so only their hyperparameters are passed. The
module's schedule defaults (``n_warmup_steps=0``, ``n_decay_steps=1``)
are not the CLI's schedule; to match ``cherimoya fit``, count the
steps per epoch the way it does and pass both:

.. code-block:: python

   from cherimoya.training import fit

   batch_size = 64
   max_epochs = 20
   n_warmup_epochs = 2
   steps_per_epoch = -(-len(training_data) // batch_size)   # ceiling division

   trainer = fit(
       model,
       training_data,
       X_valid,
       y_valid,
       X_ctl_valid=None,            # pass control tensors here if using controls
       max_epochs=max_epochs,
       early_stopping=None,         # default: train all max_epochs; an int stops
                                    # after that many epochs without count-Pearson gain
       dtype='float32',             # or 'bfloat16' / 'float16' for mixed precision
       accelerator='gpu',
       devices=1,                   # more than one trains with DDP
       batch_size=batch_size,       # global batch, split evenly across devices
       num_workers=1,               # data-loading workers per device
       n_warmup_steps=steps_per_epoch * n_warmup_epochs,
       n_decay_steps=steps_per_epoch * max(1, max_epochs - n_warmup_epochs),
   )

The remaining keyword arguments are passed to
:class:`~cherimoya.training.CherimoyaModule`: the learning rates and
weight decays (``muon_lr``, ``muon_wd``, ``adam_lr``, ``adam_wd``,
``lw_lr``, ``lw_wd``, ``lw_momentum``), ``loss_weights`` and
``ema_decay``. The learning rate, weight decay, momentum and
``loss_weights`` defaults match the CLI defaults.

Cherimoya uses a three-optimizer strategy: Muon for the 2D projection
weights in the Cheri Blocks, AdamW for the head/tail layers, biases,
and the per-block ``conv_weight``, and SGD for the Kendall uncertainty
weights ``lw0`` / ``lw1``. The Muon and AdamW rates warm up linearly
from 1% over ``n_warmup_steps`` and then follow a cosine decay to
``1e-5`` over ``n_decay_steps``; the ``lw`` rate warms up and is then
held constant.

What ``fit`` does internally:

* Maintains an :class:`~cherimoya.cherimoya.EMA` shadow of every
  floating-point parameter (decay 0.999 by default). The shadow is
  updated after every optimizer step.
* Runs the training step under Lightning's precision setting for
  ``dtype``: ``'32-true'`` for ``'float32'``, ``'bf16-mixed'`` for
  ``'bfloat16'`` and ``'16-mixed'`` for ``'float16'``, which also
  scales the loss.
* Validates at the end of each epoch using the EMA-applied weights;
  the validation Pearson correlation on counts is the metric used for
  best-checkpoint selection.
* Saves ``{model.name}.torch`` whenever validation count Pearson
  improves, and ``{model.name}.final.torch`` at the very end (also
  with EMA weights applied).
* Saves ``{model.name}.log`` with the training and validation metrics
  per epoch.

After ``fit`` returns, ``model`` holds the EMA weights, and
``trainer.checkpoint_callback.best_model_score`` is the best
validation count Pearson.

``fit`` raises a ``ValueError`` when the training set has fewer
examples than one ``batch_size``, since no training step could be
taken.

Once the gradients on ``lw0`` (the profile loss-weight scalar) become
small at the end of an epoch, both loss-weight scalars are frozen and
the loss reduces to a fixed weighted sum for the rest of training.


Training on several GPUs
~~~~~~~~~~~~~~~~~~~~~~~~

With ``devices`` greater than one (or ``-1`` for every visible
device), ``fit`` trains with DDP. ``batch_size`` is the global batch
and must be divisible by the number of devices; each device takes an
equal contiguous slice of every global batch through
:class:`cherimoya.io.ShardedEpochSampler`, so each step sees exactly
the examples a single device would. Validation is split across the
devices without padding, and the metrics are computed over the whole
validation set.

Lightning starts every rank after the first by re-running the current
command, so the code before the ``fit`` call runs once per rank.


Saving and loading
------------------

See :doc:`save_load` for the full discussion. Briefly:

.. code-block:: python

   model.save("my_model.torch")
   model = Cherimoya.load("my_model.torch", device="cuda")


Making predictions
------------------

For evaluation use the standard ``tangermeme.predict`` helper, which
batches the input and concatenates the outputs:

.. code-block:: python

   from tangermeme.predict import predict

   model.eval()
   y_profile, y_counts = predict(
       model, X_test,
       batch_size=64,
       device='cuda',
       dtype='float32',
   )

Reverse-complement averaging often improves performance and is what
the ``evaluate`` CLI uses when ``reverse_complement_average`` is set:

.. code-block:: python

   import torch

   y_profile_rc, y_counts_rc = predict(
       model, torch.flip(X_test, dims=(-1, -2)),
       batch_size=64, device='cuda',
   )
   y_profile_avg = (y_profile + torch.flip(y_profile_rc, dims=(-1, -2))) / 2
   y_counts_avg = (y_counts + y_counts_rc) / 2


Evaluating performance
----------------------

:func:`cherimoya.performance.calculate_performance_measures` computes
profile and counts metrics. It takes predicted logits, observed
counts, and predicted log counts, and returns a dict of tensors:

.. code-block:: python

   from cherimoya.performance import calculate_performance_measures

   measures = calculate_performance_measures(
       y_profile, y_valid, y_counts,
       measures=['profile_pearson', 'count_pearson', 'profile_jsd'],
   )

   for name, values in measures.items():
       print(f"{name}: {values.mean().item():.4f}")

If ``measures`` is ``None`` (the default), all built-in measures are
computed. The full list and signature is in :doc:`../api/performance`.
For multi-group models (see :doc:`../multi_task`), pass
``signal_groups=model.signal_groups`` so the count metrics are
computed per group rather than against a single total target.


Interpreting the metrics
~~~~~~~~~~~~~~~~~~~~~~~~

Rough ballparks from typical ChIP-seq and ATAC-seq experiments,
useful for sanity-checking a trained model:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Metric
     - Usable
     - Strong
   * - ``count_pearson``
     - ≥ 0.5
     - ≥ 0.7
   * - ``profile_pearson``
     - ≥ 0.3
     - ≥ 0.5
   * - ``profile_jsd``
     - ≤ 0.5
     - ≤ 0.3
   * - ``profile_mnll``
     - context-dependent — compare to baseline
     - context-dependent

Notes:

* Count Pearson is computed across the held-out set as a single
  scalar (one correlation across all examples), so it is sensitive
  to dynamic range. Datasets with a wider distribution of peak
  heights produce higher count Pearson at fixed model quality;
  comparing count Pearson across datasets is not apples-to-apples.
* Profile Pearson and JSD are per-example and then averaged, so
  they're more comparable across datasets but noisier per example.
* ``count_pearson`` near zero is almost always a sign of a setup
  problem (see :doc:`../troubleshooting`); a well-trained model on
  real data essentially never lands there.
* When training with controls, omitting ``controls`` at evaluation
  collapses ``count_pearson`` — the count head sees the wrong
  feature distribution.
