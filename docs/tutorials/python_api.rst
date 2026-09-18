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
a ``torch.utils.data.DataLoader``:

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
       batch_size=64,
       num_workers=1,                       # async prefetch workers
       random_state=0,                      # base seed; reproducible
       verbose=True,                        # print progress and filter counts
   )

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


Optimizers and schedulers
-------------------------

Cherimoya uses a three-optimizer strategy: Muon for the 2D projection
weights in the Cheri Blocks, AdamW for the head/tail layers, biases,
and the per-block ``conv_weight``, and SGD for the Kendall uncertainty
weights ``lw0`` / ``lw1``. To match the CLI defaults exactly:

.. code-block:: python

   from torch.optim import AdamW, Muon, SGD
   from torch.optim.lr_scheduler import (LinearLR, CosineAnnealingLR,
       ConstantLR, SequentialLR)

   # Route parameters into three buckets. Muon takes the 2D projection
   # weights inside Cheri Blocks (linear1.weight, linear2.weight); SGD
   # takes lw0/lw1; AdamW takes everything else, including the count
   # head (name == "linear.weight") and the per-block ``conv_weight``.
   muon_params, adam_params, lw_params = [], [], []
   for name, p in model.named_parameters():
       if name in ("lw0", "lw1"):
           lw_params.append(p)
       elif (p.ndim == 2 and "weight" in name and name != "linear.weight"
               and "conv_weight" not in name):
           muon_params.append(p)
       else:
           adam_params.append(p)

   muon_optimizer = Muon(muon_params, lr=0.025, weight_decay=0.03)
   adam_optimizer = AdamW(adam_params, lr=0.001, weight_decay=0.0)
   lw_optimizer = SGD(lw_params, lr=0.001, weight_decay=0.0, momentum=0.9)

   # Warmup for 2 epochs, then cosine decay for the rest of training
   # down to eta_min=1e-5. Note T_max uses (max_epochs - n_warmup_epochs),
   # not max_epochs.
   max_epochs = 20
   n_warmup_epochs = 2
   num_warmup_iters = len(training_data) * n_warmup_epochs
   num_decay_iters = len(training_data) * max(1, max_epochs - n_warmup_epochs)

   def make_scheduler(opt):
       warm = LinearLR(opt, start_factor=0.01, total_iters=num_warmup_iters)
       cos = CosineAnnealingLR(opt, T_max=num_decay_iters, eta_min=1e-5)
       return SequentialLR(opt, schedulers=[warm, cos], milestones=[num_warmup_iters])

   muon_scheduler = make_scheduler(muon_optimizer)
   adam_scheduler = make_scheduler(adam_optimizer)

   # lw schedule is warmup then flat — the Kendall weights are not
   # cosine-decayed.
   lw_warm = LinearLR(lw_optimizer, start_factor=0.01, total_iters=num_warmup_iters)
   lw_const = ConstantLR(lw_optimizer, factor=1.0, total_iters=1)
   lw_scheduler = SequentialLR(lw_optimizer,
       schedulers=[lw_warm, lw_const], milestones=[num_warmup_iters])


Training
--------

.. code-block:: python

   model.fit(
       training_data,
       muon_optimizer, adam_optimizer, lw_optimizer,
       muon_scheduler, adam_scheduler, lw_scheduler,
       X_valid=X_valid,
       X_ctl_valid=None,            # pass control tensors here if using controls
       y_valid=y_valid,
       max_epochs=20,
       batch_size=64,
       early_stopping=5,            # stop after 5 epochs without count-Pearson gain
       dtype='float32',             # or 'bfloat16' for mixed precision via autocast
       device='cuda',
   )

What ``fit`` does internally:

* Maintains an :class:`~cherimoya.cherimoya.EMA` shadow of every
  floating-point parameter (decay 0.999). The shadow is updated after
  every optimizer step.
* Runs the training step with ``torch.autocast`` using ``dtype``.
* Validates at the end of each epoch using the EMA-applied weights;
  the validation Pearson correlation on counts is the metric used for
  best-checkpoint selection.
* Saves ``{model.name}.torch`` whenever validation count Pearson
  improves, and ``{model.name}.final.torch`` at the very end (also
  with EMA weights applied).
* Saves ``{model.name}.log`` with the training and validation metrics
  per epoch.

Once the gradients on ``lw0`` (the profile loss-weight scalar) become
small at the end of an epoch, both loss-weight scalars are frozen and
the loss reduces to a fixed weighted sum for the rest of training.


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
