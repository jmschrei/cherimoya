Troubleshooting and FAQ
=======================

The most common things that go wrong on a first Cherimoya run, with
the symptom each one produces and what to change. See :doc:`glossary`
for any unfamiliar terms.


Is my dataset big enough?
-------------------------

Two related questions: how many peaks, and how deep a library?

**Peaks.** Cherimoya trains comfortably on tens of thousands of
peaks. As a rough guide:

* Fewer than ~2,000 peaks → expect underfitting; consider transfer
  learning or pooling related experiments instead.
* 2,000–10,000 peaks → workable but on the edge; reduce
  ``n_filters`` to 48–64 and watch the validation count Pearson
  carefully.
* 10,000+ peaks → default configuration is appropriate.
* 50,000+ peaks → no special handling needed; larger models
  (``n_filters=192`` or ``256``) become worth trying.

**Library depth.** Cherimoya needs enough reads per peak that the
profile MNLL is informative. As a rough guide:

* Fewer than ~50 reads per peak → MNLL is dominated by noise;
  ``count_pearson`` may still be meaningful but profile-shape
  metrics will be flat.
* Hundreds of reads per peak → normal regime.
* Thousands of reads per peak → ATAC/DNase-style; in this regime
  the model is signal-limited, not data-limited.

If your library is below the lower threshold, the cheapest fix is
to pool replicates before peak calling and use the pooled BAM as
the signal.


CUDA out of memory at the start of training
-------------------------------------------

Symptom: ``torch.cuda.OutOfMemoryError`` in the first or second
training step.

The default training batch size (64) and input window (2114 bp) fit
comfortably on a 16 GB GPU at the default 9-layer, 128-filter model.
If you hit OOM, the fastest fixes:

* Reduce ``fit_parameters.batch_size`` from 64 to 32 (or 16).
* Use bf16 autocast: set ``fit_parameters.dtype`` to ``"bfloat16"``.
* Shrink the model: ``fit_parameters.n_filters`` from 128 to 64 or 48.

GPU memory at training time is dominated by activations
(``batch_size × in_window × n_filters × n_layers`` plus the
``expansion``-wide MLP activations) and the optimizer state. Reducing
``batch_size`` is the cheapest way to fit; reducing ``n_filters`` is
the cheapest way to keep batch size high.


CUDA out of memory in the attribute step
----------------------------------------

Symptom: ``torch.cuda.OutOfMemoryError`` from ``cherimoya attribute``
or from step 2 of ``cherimoya pipeline``.

Under DeepLIFT/SHAP, the default algorithm, ``batch_size`` counts
sequence-reference pairs, and each pair is run forward and backward
with its activations kept. On the default 9-layer, 128-filter model at
2114 bp, 16 pairs peaked at 4.0 GB, 32 at 7.8 GB, 64 (the default) at
15.5 GB and 512 at 124 GB, with about the same wall time at each, so
memory grows with ``batch_size``. Lower
``attribute_parameters.batch_size`` (64 → 32 → 16). With a fixed
``random_state`` the attributions are the same at any batch size, up
to float rounding.


"Convergence deltas too high" during attribution
------------------------------------------------

Symptom: ``RuntimeWarning: Convergence deltas too high: tensor([...])``
from ``cherimoya attribute`` or from ``deep_lift_shap`` in Python.

DeepLIFT/SHAP attributions for an example and a reference should sum
to the difference between the two predictions. The warning lists, for
each pair in a batch, how far they missed, and fires when any miss
exceeds ``warning_threshold`` (default 0.001). The threshold is
absolute, in the units of the attributed output — log counts, or the
profile wrapper's scalar — so compare the reported deltas with the
size of the predictions before deciding what they mean.

The known cause is calling ``deep_lift_shap`` in Python without
``additional_nonlinear_ops=attribution_ops()``: two of Cherimoya's
layers need rules, and for the profile head the error then exceeds the
prediction itself (see :doc:`api/deep_lift_shap`). ``cherimoya
attribute`` always registers them.


The first iteration is very slow, then it speeds up
---------------------------------------------------

Symptom: the first training step takes tens of seconds; subsequent
steps are fast.

That is Triton autotune. The first call to each Triton kernel sweeps a
list of (block size, num_warps, num_stages) configurations to find the
best one, which takes wall time. Once autotune is done, the chosen
configuration is cached for the process lifetime. There is nothing
wrong; nothing to fix. The same is true on the first inference call
(the inference megakernel autotunes separately).

If you need to amortize this across runs you can pre-warm by running
one batch through the model before timing anything you care about; see
``bench_kernels.py`` in the repo for the pattern.


Training loss is NaN
--------------------

Symptom: ``profile_loss`` or ``count_loss`` becomes ``nan`` partway
through an epoch, or the validation metrics are ``nan``.

The most common causes, in order:

1. **A peak with zero counts**. The multinomial log-likelihood is
   ``-inf`` when the true counts sum to zero across the entire output
   window. :func:`cherimoya.io.PeakGenerator` filters peaks above
   the 99th-percentile-by-1.2x ceiling but does not filter empty
   peaks. Set ``fit_parameters.min_counts`` to a small positive
   value (e.g. 5 or 10).
2. **Mismatched strand counts**. If your signals are stranded but you
   passed only one bigWig (or vice versa) the loaded ``y`` will have
   the wrong shape. Verify by setting ``verbose=True`` and inspecting
   the training-set and validation-set shapes printed at startup.

   Also confirm that a stranded ``(+, -)`` pair is wrapped as a single
   inner list — e.g. ``signals=[["plus.bw", "minus.bw"]]`` — rather
   than passed flat. A flat two-element list is now interpreted as
   two *independent* unstranded tracks, which silently disables the
   ``(+, -)`` swap during reverse-complement augmentation.
3. **bf16 overflow in the count head**. If you use ``dtype="bfloat16"``
   and the per-locus counts are very large, the log-counts loss can
   overflow. Fall back to ``"float32"`` to confirm; if that fixes it,
   either scale your signal down (``preprocessing_parameters.scale_factor``
   when generating bigWigs) or cap with ``max_counts``.


Stranded predictions come from only one strand
----------------------------------------------

Symptom: a stranded ``(+, -)`` model (TF ChIP, PRO-cap, ...) produces
a reconstructed profile (via
:class:`cherimoya.wrappers.ExpectedCountsWrapper`) with nearly all of
its signal on one strand, even though the observed data has comparable
coverage on both strands offset by ~100-300 bp.

This was a profile-loss bug fixed in the **Unreleased** release. The
loss previously normalized each strand of a group independently, which
left the relative magnitude between strands an unconstrained gauge; the
inference wrapper distributes a group's counts with a *joint* softmax
across both strands, exponentiating that arbitrary offset and collapsing
the signal onto one strand. The loss now normalizes each signal group's
channels jointly, so the strand balance is a trained quantity. Models
trained on an older release have the miscalibrated offset baked into
their weights and must be **retrained**. Unstranded ATAC/DNase models
are unaffected (the change is bit-identical for single-channel groups).
See the :doc:`CHANGELOG`.


Training Pearson is stuck near zero
-----------------------------------

Symptom: validation count Pearson hovers at 0 or below for many
epochs and never climbs.

Things to check, in order:

1. **You're passing the same signal as both training signal and
   validation signal.** If they differ (e.g. by replicate), the model
   is being asked to generalize across replicates, which is much
   harder than generalizing across chromosomes.
2. **Validation chromosomes contain no peaks.** Check the
   ``Validation Set Size`` value printed at startup. If it's zero or
   tiny, your ``validation_chroms`` don't intersect your peak file —
   common when your peaks are subset to a single chromosome.
3. **The signal really is uninformative.** Train on a known-good
   ChIP-seq target (e.g. CTCF in K562 from ENCODE) as a control. If
   that converges, the issue is upstream of Cherimoya.
4. **You silently dropped controls.** If your trained-with-controls
   model is being evaluated without ``X_ctl``, the count head sees
   garbage and Pearson collapses. The evaluation step JSON must list
   the same ``controls`` as the fit step.


Pipeline cannot find a file
---------------------------

Symptom: ``FileNotFoundError: Pipeline cannot start; the following
inputs are missing: ...`` from ``cherimoya pipeline``.

The pre-flight check inside ``pipeline`` validates that every local
input path exists before any expensive work starts. Remote paths
(``http://``, ``https://``, ``s3://``, ``gs://``) are skipped because
resolving them requires network access. If a path looks remote but
isn't (e.g. a relative URL fragment), the check won't catch it.

If the missing path is a file the pipeline is *supposed* to generate
later in the run (e.g. you specified ``loci`` pointing at the not-yet
called peak file), set the offending key to ``null`` instead and let
the pipeline produce it.


MACS3 hangs or returns no peaks
-------------------------------

Symptom: ``cherimoya pipeline`` sits on step 0.1 for a long time, or
the resulting ``*_peaks.narrowPeak`` is empty.

* If the BAM is large and you don't already have peaks, MACS3 itself
  can take ~10 min on a typical ChIP-seq library. This is normal.
* If the BAM is small and the resulting peak file is empty, your
  ``callpeaks_q`` is too strict. Try 0.1 or 0.5 to see if any peaks
  emerge; if so, your library is just shallow.
* If the file format auto-detection picked wrong (e.g. ``BAM`` when
  it should have been ``BAMPE``), set
  ``preprocessing_parameters.callpeaks_format`` explicitly.


bam2bw says "couldn't open" a remote URL
----------------------------------------

Symptom: ``bam2bw`` fails on an ``s3://`` or ``https://`` path.

Cherimoya streams BAM/SAM and FASTA inputs through ``bam2bw`` /
``tangermeme.io``. Streaming requires the remote storage to support
range requests for the file type involved. Public HTTPS BAMs hosted
on ENCODE, S3-presigned URLs, and standard GCS objects all work. If
the remote path requires credentials, set them in your environment
(``AWS_*`` for S3, ``GOOGLE_APPLICATION_CREDENTIALS`` for GCS) before
running ``cherimoya pipeline``.


.. _torch_compile_cudagraph_errors:

``torch.compile`` / CUDA-graph errors at inference time
-------------------------------------------------------

Symptom: an inference script raises a ``torch._dynamo`` /
``torch._inductor`` traceback originating from inside
``Cherimoya.forward``, or a CUDA-graph runtime error such as
``accessing tensor output of CUDAGraphs that has been overwritten
by a subsequent run``.

:class:`cherimoya.Cherimoya` wraps its forward in
``torch.compile(mode='max-autotune')`` by default, which captures a
CUDA graph and reuses preallocated buffer slots across calls. This
is fast but can interact unhappily with caller code that holds
references across calls, mixes graph and non-graph allocators, or
hits an inductor edge case for a specific PyTorch version.

If you hit any such error, two opt-outs are available, both
numerically equivalent to the default:

.. code-block:: python

   from cherimoya import Cherimoya

   # Full bypass: eager forward, no torch.compile at all.
   model = Cherimoya.load("checkpoint.torch", device="cuda",
                          compile=False)

   # Targeted: keep autotuned kernels, skip the CUDA-graph capture.
   model = Cherimoya.load("checkpoint.torch", device="cuda",
                          compile_mode='max-autotune-no-cudagraphs')

``compile_mode`` is forwarded directly to ``torch.compile(mode=...)``,
so any mode PyTorch accepts works (``'reduce-overhead'``,
``'default'``, etc.). When ``compile=False`` the mode is ignored.
Both opt-outs still go through Cherimoya's Triton inference kernels;
only the ``torch.compile`` wrapping changes. The test suite verifies
parity at ``atol=rtol=1e-4`` between ``compile=True`` and
``compile=False``.

The performance cost is the compile speedup itself: typically
~10-20% on the inference megakernel on recent GPUs, smaller on
training. **If you don't immediately recognize a** ``torch.compile``
**or CUDA-graph error, the safest fix is** ``compile=False`` — it
sidesteps the entire class of compile/cudagraph foot-guns at the
cost of that speedup.

From the CLI the same two settings are the ``compile`` and
``compile_mode`` JSON keys, accepted by ``evaluate``, ``attribute``,
``marginalize`` and at the top level of a ``pipeline`` JSON, where one
value reaches every step that loads a model. ``attribute`` ignores them
under DeepLIFT/SHAP, its default, and always loads eagerly::

    {"compile": false}
    {"compile_mode": "max-autotune-no-cudagraphs"}

.. note::

   For the megakernel to hit its fast path (precomputed bf16 weight
   cast reused across calls), call ``model.eval()`` before inference.
   Without ``.eval()`` the megakernel still runs correctly, but it
   recomputes the cast on every call — adding ~10-27% latency at small
   batch sizes and under ~2% at production batch sizes. See
   :doc:`benchmarks` for the breakdown.


``Cherimoya.load`` rejects a checkpoint
---------------------------------------

Symptom: ``Cherimoya.load("...torch")`` raises ``KeyError: 'config'``
or ``RuntimeError`` about ``weights_only``.

The checkpoint was saved with the legacy ``torch.save(model, ...)``
pickle path that pre-dates v0.1.0 and is not loadable through the
current config-plus-state-dict loader. Retrain from scratch with the
current release.


Loaded model gives different predictions than training did
----------------------------------------------------------

Symptom: at the last training-epoch validation, count Pearson was X;
after ``Cherimoya.load`` it is materially different.

The saved checkpoint contains the **EMA-applied** weights, not the
running training weights. That is intentional and is what produces
the best validation numbers during training. There is no mismatch to
fix — the model you load is the correct one. Confirming: ``model.fit``
applies the EMA shadow before saving, so the comparison should be
against the *EMA* number printed in the training log, not the
mid-epoch training-loss number.
