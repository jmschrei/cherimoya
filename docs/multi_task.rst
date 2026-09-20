Multi-task training
===================

Cherimoya can predict more than one biological readout from the same
DNA sequence. A model that emits an unstranded ATAC profile, a
stranded CTCF binding profile, and a stranded YY1 binding profile —
all from one shared backbone — is a **multi-task** model. The
parameter that controls this is called ``signal_groups``.

This page explains what signal groups are, how they shape every part
of training (sampling, loss, validation metrics, the saved logs), and
walks through the three configurations users hit in practice:

1. **Unstranded single-task** — one assay, one output channel.
2. **Stranded single-experiment** — one assay, two output channels
   that are a strand pair.
3. **Variably-multitask** — multiple assays, mixing unstranded and
   stranded.

For the math of the Kendall-Gal loss combiner that ties the two
loss components together, see :doc:`architecture`. For the
end-to-end CLI command list, see :doc:`cli`. For assay-specific
recipes, see :doc:`recipes/chipseq_tf`, :doc:`recipes/atacseq`, and
:doc:`recipes/dnaseq`.


What a signal group is
----------------------

A **signal group** is one biological modality whose output channels
share an orientation. An unstranded coverage track (ATAC, DNase, a
single-strand ChIP) is a single-channel group; a stranded
``(+, -)`` BPNet-style pair is a two-channel group.

``signal_groups`` is a list of integers, one entry per group, giving
the number of channels in that group. Some examples:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - ``signal_groups``
     - Profile channels
     - Count predictions
     - Setting
   * - ``[1]``
     - 1
     - 1
     - Default. One unstranded track (e.g. ATAC).
   * - ``[2]``
     - 2
     - 1
     - One stranded experiment (e.g. CTCF ChIP). Two profile
       channels share a single per-locus count.
   * - ``[1, 2]``
     - 3
     - 2
     - Co-train one unstranded track and one stranded pair (e.g.
       ATAC + CTCF). Two separate count predictions.
   * - ``[1, 2, 2, 1, 2]``
     - 8
     - 5
     - Co-train five experiments (two unstranded, three stranded
       pairs).

The model's profile head emits ``sum(signal_groups)`` output channels
— one per signal channel. The count head emits ``len(signal_groups)``
predictions — one per group, so the two strands of a stranded pair
share a single per-locus count target. That tying matches the
biology: the total read count for a ChIP experiment is a single
number per locus, not a per-strand number.

Cherimoya enforces this contract everywhere:

* **Data loading**. The grouped form of ``signals`` is what tells the
  data loader which output channels belong to which biological
  modality.
* **Reverse-complement augmentation**. Channels swap *within* each
  group (unstranded groups stay put, stranded pairs swap ``+`` ↔
  ``-``); groups never bleed into one another. This is what makes
  multi-task training actually train rather than scramble itself.
* **Loss balancing**. Every group contributes exactly one
  profile-loss term and one count-loss term, regardless of how many
  channels it has. A stranded TF group with two channels does not
  outweigh an unstranded ATAC group with one channel.
* **Validation metrics and best-model selection**. Both the headline
  profile Pearson and the headline count Pearson are means across
  groups, not across channels. A detail log (``{name}.detailed.log``)
  also records the per-group breakdown.
* **Outlier filtering**. The PeakGenerator computes a 99th-percentile
  threshold per group and drops a locus if it's an outlier in *any*
  group, so a high-count TF can't mask outliers in a co-trained
  low-count ATAC.

The rest of this page shows how to wire each configuration up.


Mode 1 — Unstranded single-task
-------------------------------

**When to use.** You have a single assay whose coverage is not
strand-resolved: ATAC-seq, DNase-seq, ChIP-seq with reads pooled
across strands, MNase-seq, etc. This is the canonical and most
common case, and it's the default — calling ``Cherimoya()`` with no
arguments gives you exactly this model.

**Shapes.** ``signal_groups=[1]``. One profile channel,
``(N, 1, out_window)``. One count prediction, ``(N, 1)``.

**Loss.** The profile MNLL and the count MSE each contribute one term;
Kendall-Gal weights are scalars. Nothing about training differs from
a classical BPNet-style single-task fit.

Python:

.. code-block:: python

   from cherimoya import Cherimoya

   model = Cherimoya(n_filters=128, n_layers=9).cuda()

   X = ...                                       # (N, 4, 2114) one-hot DNA
   y_profile, y_counts = model(X)
   # y_profile.shape == (N, 1, 1000)
   # y_counts.shape  == (N, 1)

Fit JSON (processed bigWigs already on disk). ``signals`` is a flat
list of one string — that's exactly what the default
``signal_groups=[1]`` expects:

.. code-block:: json

   {
     "name": "atac_k562",
     "sequences": "hg38.fa",
     "loci": "k562_atac_peaks.narrowPeak",
     "negatives": "k562_atac_negatives.bed",
     "signals": ["k562_atac.bw"],
     "controls": null
   }

Run as:

.. code-block:: bash

   cherimoya fit -p atac_k562.fit.json

Pipeline JSON (you have BAMs and want Cherimoya to call peaks,
convert to bigWig, sample negatives, and train in one command).
``preprocessing_parameters.unstranded`` is ``true`` so ``bam2bw``
emits a single ``.bw`` file:

.. code-block:: json

   {
     "name": "atac_k562",
     "sequences": "hg38.fa",
     "signals": ["k562_atac.bam"],
     "controls": null,
     "preprocessing_parameters": {
       "unstranded": true,
       "fragments": true,
       "paired_end": true,
       "pos_shift": 4,
       "neg_shift": -4
     }
   }

After preprocessing, the pipeline rewrites ``signals`` to
``["atac_k562.bw"]`` and hands the JSON off to the fit step.


Mode 2 — Stranded single-experiment
-----------------------------------

**When to use.** You have a single assay that's strand-resolved and
you want to predict the ``+`` and ``-`` strand coverage as two
separate profile channels: TF ChIP-seq (the BPNet convention),
GRO-seq, PRO-seq, CAGE, CoPRO, NET-seq. The two strands are tied:
they share a single per-locus count target (because the total read
count for that experiment is one number per locus, not two), and
they swap under reverse-complement augmentation as a unit.

**Shapes.** ``signal_groups=[2]``. Two profile channels (channel 0 =
``+``, channel 1 = ``-``), ``(N, 2, out_window)``. One count
prediction (the shared per-locus total), ``(N, 1)``.

**Loss.** The group's two strands are scored as a **single joint
multinomial** — the ``+`` and ``-`` channels are concatenated and one
``log_softmax`` is taken over the flattened ``channels * length`` — giving
one per-group profile loss before Kendall-Gal weighting. Normalizing the
strands together (rather than independently) makes their relative
magnitude part of the trained objective. This configuration therefore
contributes one profile-loss term and one count-loss term — the same
number of loss terms as the unstranded case. The Kendall-Gal weight
tensors ``lw0`` and ``lw1`` are each shape ``(1,)``.

Python:

.. code-block:: python

   from cherimoya import Cherimoya

   model = Cherimoya(
       n_filters=128, n_layers=9,
       signal_groups=[2],            # one stranded (+, -) pair
       n_control_tracks=2,           # stranded input control
   ).cuda()

   X = ...                                       # (N, 4, 2114)
   X_ctl = ...                                   # (N, 2, 2114)
   y_profile, y_counts = model(X, X_ctl)
   # y_profile.shape == (N, 2, 1000)
   # y_counts.shape  == (N, 1)

Fit JSON (processed bigWigs). The stranded pair is **one group of
two files** — wrap it in an inner list so the data loader knows
those two files are strand-paired (and need to swap channels
together under RC):

.. code-block:: json

   {
     "name": "ctcf_k562",
     "sequences": "hg38.fa",
     "loci": "ctcf_k562_peaks.narrowPeak",
     "negatives": "ctcf_k562_negatives.bed",
     "signals": [["ctcf_k562.+.bw", "ctcf_k562.-.bw"]],
     "controls": [["input_k562.+.bw", "input_k562.-.bw"]]
   }

Note the **double brackets** around each list of bigWigs. A bare flat
list ``["ctcf.+.bw", "ctcf.-.bw"]`` would be interpreted as two
*independent* unstranded tracks (``signal_groups=[1, 1]``), which is
not what you want for a stranded experiment — the ``+`` / ``-``
swap on RC depends on knowing the two files are paired.

Pipeline JSON (BAMs). ``preprocessing_parameters.unstranded`` is
``false``, so ``bam2bw`` emits the ``+`` / ``-`` pair and the
pipeline rewrites ``signals`` to the nested form above before
calling fit:

.. code-block:: json

   {
     "name": "ctcf_k562",
     "sequences": "hg38.fa",
     "signals": ["ctcf_k562.bam"],
     "controls": ["input_k562.bam"],
     "preprocessing_parameters": {
       "unstranded": false
     }
   }


Mode 3 — Variably-multitask
---------------------------

**When to use.** You want one model that predicts several
experiments at once, possibly mixing unstranded and stranded
modalities. Examples:

* ATAC + one TF (one unstranded group + one stranded pair).
* DNase + several TFs (one unstranded group + several stranded
  pairs).
* A panel of TF ChIP-seq experiments, plus a chromatin
  accessibility track to share the backbone's representation.

The motivation is parameter sharing: a single backbone can learn
features useful for every modality simultaneously, and the
attribution scores it produces can be inspected per modality.

**Shapes.** ``signal_groups`` has one entry per experiment. For
example, ``[1, 2, 2]`` means one unstranded ATAC track plus two
stranded TF pairs — 5 profile channels and 3 count predictions:
``(N, 5, out_window)`` and ``(N, 3)``.

**Loss.** Each group contributes exactly one profile-loss term and
one count-loss term, regardless of how many channels it has. The
ATAC group's profile MNLL is the per-length MNLL of its one channel;
each TF group's profile MNLL is a single multinomial over its ``+``
and ``-`` strands concatenated (normalized jointly, so the strand
balance is learned). All three group-level profile losses then get
weighted by Kendall-Gal uncertainty weights (``lw0`` is shape
``(3,)``), summed with the per-group count MSEs (also weighted by
``lw1`` shape ``(3,)``), and that's the total loss. The result is
that **each modality contributes equally** to the optimization —
no modality is double-weighted because it happens to be stranded.

This is the configuration where the per-group RC handling is most
load-bearing: with a mix of unstranded and stranded outputs, the
naive "reverse channels and length" RC corrupts the cross-modality
mapping. Cherimoya's per-group permutation keeps each group's
internal channels swapping independently, and never moves channels
across group boundaries.

Python:

.. code-block:: python

   from cherimoya import Cherimoya

   # ATAC (1 channel) + CTCF stranded (2 channels) + YY1 stranded
   # (2 channels): three groups, five profile channels, three count
   # predictions.
   model = Cherimoya(
       n_filters=128, n_layers=9,
       signal_groups=[1, 2, 2],
       n_control_tracks=0,
   ).cuda()

   X = ...                                       # (N, 4, 2114)
   y_profile, y_counts = model(X)
   # y_profile.shape == (N, 5, 1000)
   # y_counts.shape  == (N, 3)

Fit JSON (processed bigWigs). Each top-level entry of ``signals`` is
one group; a string is shorthand for a one-channel unstranded group,
and a nested list is a multi-channel group. Mix freely:

.. code-block:: json

   {
     "name": "atac_ctcf_yy1_k562",
     "sequences": "hg38.fa",
     "loci": "joint_peaks.narrowPeak",
     "negatives": "joint_negatives.bed",
     "signals": [
       "k562_atac.bw",
       ["ctcf_k562.+.bw", "ctcf_k562.-.bw"],
       ["yy1_k562.+.bw", "yy1_k562.-.bw"]
     ],
     "controls": null
   }

The order of entries in ``signals`` defines the order of the model's
output channels: in this example channel 0 is ATAC, channels 1–2 are
CTCF (+/-), channels 3–4 are YY1 (+/-). Count predictions are in the
same group order: count 0 is ATAC, count 1 is CTCF, count 2 is YY1.

.. admonition:: BAM input for variably-multitask
   :class: note

   The ``cherimoya pipeline`` BAM-to-bigWig preprocessing step
   applies a single set of ``bam2bw`` flags (``unstranded``,
   ``fragments``, ``paired_end``, ``pos_shift``, ``neg_shift``) to
   every input file in one batch. Mixing modalities — for example,
   paired-end unstranded ATAC alongside single-end stranded ChIP —
   requires different flag combinations for different inputs, which
   the pipeline does not support in a single run.

   For a variably-multitask model with BAM inputs, run
   ``cherimoya pipeline`` separately for each modality (each one
   uses Mode 1 or Mode 2 above) to produce per-modality bigWigs and
   peak BEDs, then write the multi-group fit JSON shown above and
   call ``cherimoya fit`` directly. The pipeline command does
   not need to be involved in the joint step.


How signal groups affect training
---------------------------------

The grouping decision propagates through the entire training stack.
The behaviors below are all automatic once ``signal_groups`` is set
(via the model constructor for Python use, or via the structured
``signals`` form for CLI use).

Reverse-complement augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``reverse_complement=True`` (the default), about half the
training examples are augmented to their reverse complement. The
input DNA is complemented (ACGT → TGCA) and reversed along the
length axis. The signal tensor needs a matching transformation:

* The length axis always flips.
* Each group's channels are reversed *within the group*. For a
  size-1 unstranded group this is the identity (the channel stays
  in place); for a stranded ``(+, -)`` pair the two channels swap;
  for a hypothetical size-3 group the three channels reverse order.
* Groups never exchange channels with each other.

This is what makes a mixed ``[1, 2]`` model trainable: the ATAC
channel stays at index 0, and the TF ``+``/``-`` channels swap with
each other at indices 1 and 2. Without per-group handling, a naive
"flip the whole channel dimension" RC would scramble ATAC's
predictions into the TF slots and vice versa on every RC-augmented
example.

Per-group loss balancing
~~~~~~~~~~~~~~~~~~~~~~~~

Cherimoya combines profile and count losses with Kendall-Gal
uncertainty weighting (see :doc:`architecture` for the math), with
two learnable weight tensors ``lw0`` and ``lw1`` both shape
``(len(signal_groups),)`` — one uncertainty weight per group on each
side of the profile / counts split. Concretely:

* Each group's profile MNLL is a single multinomial over that group's
  channels and length jointly, then weighted by ``lw0``. A stranded
  pair contributes one profile-loss term (both strands normalized
  together, so their relative magnitude is trained), exactly the same
  number of terms an unstranded track contributes.
* The per-group count MSE is weighted by ``lw1`` directly.

A useful mental model: ``lw0`` and ``lw1`` are *task-level*
balance knobs. Cherimoya doesn't have separate balance knobs for
the ``+`` and ``-`` strand of a stranded experiment — they're
tied, just as they share a count head.

Per-group outlier filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data loader filters loci whose total counts are above 1.2× the
99th-percentile total. This threshold is computed **per group**: each
modality's per-locus sum is compared against that modality's own
threshold, and a locus is dropped if it's an outlier in any group.
This matters in multi-task training where one modality (often a TF
ChIP) can have peaks orders of magnitude larger than a co-trained
modality (often ATAC). A single global threshold would be dominated
by the larger modality and would silently let the smaller modality's
outliers through.

Validation metrics and the two training logs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two log files are written every epoch.

``{name}.log`` — the summary log, printed to stdout when
``verbose=True`` and saved to disk as a tab-separated table. Same
columns regardless of how many groups the model has, so it stays
readable for any configuration. The two Pearson columns are means
across groups:

* **Validation Profile Pearson** is the mean over groups of (mean
  over channels in that group of profile Pearson).
* **Validation Count Pearson** is the mean over groups of the
  per-group count Pearson.

Each group contributes one number to each mean, so no modality is
double-counted because it happens to be stranded.

The two training columns are epoch averages:

* **Training MNLL** is the mean, over every full batch in the epoch,
  of that batch's profile loss averaged across groups.
* **Training Count MSE** is the same average of the batch's count
  loss.

Both are the unweighted mean across groups of the raw per-group loss,
not the Kendall-weighted sum the optimizer descends, which is what
makes them comparable to the validation columns beside them. They are
still measured with the live training weights as those weights change
through the epoch, while the validation columns are computed once at
the end of the epoch with the EMA weights, so a small gap between the
training and validation MNLL is expected even on data the model fits
well.

``{name}.detailed.log`` — saved to disk only (never printed). Same
columns as the summary log, plus one ``ProfilePearson_g{i}`` and one
``CountPearson_g{i}`` column per signal group, for offline
per-modality analysis. For a model with three groups you get three
extra Profile columns and three extra Count columns; for a model
with three hundred groups you get six hundred. Use this file when
you need to see whether one particular modality is failing while the
others train fine.

The ``cherimoya evaluate`` performance TSV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cherimoya evaluate`` writes a ``{name}.performance.tsv`` file with
the same seven columns as before (``profile_mnll``, ``profile_jsd``,
``profile_pearson``, ``profile_spearman``, ``count_pearson``,
``count_spearman``, ``count_mse``). When the loaded model has more
than one signal group the file has **one row per group**, in
``signal_groups`` order — row 0 holds the metrics for the first
group, row 1 for the second, and so on. Profile metrics are pooled
the same way the training-time per-group profile Pearson is (mean
over the group's channels and the validation loci); count metrics
are read out of the per-group ``(n_groups,)`` tensors that
``calculate_performance_measures`` already produces.

Single-group models still write exactly one data row, byte-identical
to the prior ``.mean()``-of-everything format — a single-group model
trained today and evaluated tomorrow produces the same TSV as the
single-group model trained before this grouping change.

There is no group-identifier column. The row order is the contract;
in a Python consumer, pairing the rows with ``signal_groups`` from
the model checkpoint identifies each row's modality.

Best-model selection (which epoch's weights get saved as the
``{name}.torch`` checkpoint) uses the mean-across-groups count
Pearson — the same scalar shown in the summary log's "Validation
Count Pearson" column.
