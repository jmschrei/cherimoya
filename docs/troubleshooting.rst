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
  (``n_filters=128``) become worth trying.

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
comfortably on a 16 GB GPU at the default 9-layer, 96-filter model.
If you hit OOM, the fastest fixes:

* Reduce ``fit_parameters.batch_size`` from 64 to 32 or 16.
* Use bf16 autocast: set ``fit_parameters.dtype`` to ``"bfloat16"``.
* Shrink the model: ``fit_parameters.n_filters`` from 96 to 64 or 48.

GPU memory at training time is dominated by activations
(``batch_size × in_window × n_filters × n_layers`` plus the
``expansion``-wide MLP activations) and the optimizer state. Reducing
``batch_size`` is the cheapest way to fit; reducing ``n_filters`` is
the cheapest way to keep batch size high.


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
3. **bf16 overflow in the count head**. If you use ``dtype="bfloat16"``
   and the per-locus counts are very large, the log-counts loss can
   overflow. Fall back to ``"float32"`` to confirm; if that fixes it,
   either scale your signal down (``preprocessing_parameters.scale_factor``
   when generating bigWigs) or cap with ``max_counts``.


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

``torch.compile`` / CUDA graphs errors at inference time
--------------------------------------------------------

Symptom: an inference script raises one of

* ``RuntimeError: Error: accessing tensor output of CUDAGraphs that
  has been overwritten by a subsequent run.``
* a less specific ``torch._dynamo`` / ``torch._inductor`` traceback
  originating from inside ``Cherimoya.forward``.

By default :class:`cherimoya.Cherimoya` wraps its forward pass in
``torch.compile(mode='max-autotune')``, which captures a CUDA graph
and reuses preallocated buffer slots across calls. A small tensor
cache inside the inference kernels (used to hold the cast MLP
weights) can end up referencing those graph buffers, and certain
usage patterns then trip the CUDA-graph runtime's overwrite guard.
The two patterns that surface this in practice are:

* **Loading multiple Cherimoya instances in one process and calling
  them in sequence** (e.g. running all five folds of a comprehensive
  model on the same loci). The shared compiled forward sees a cache
  miss on the second instance and tries to reuse a buffer the first
  instance is still holding.
* **Loading one instance, then re-loading or re-initializing weights
  in-place**, then calling forward again. The same buffer-reuse path
  triggers.

**Fix: load without compile.** Both the class constructor and
:meth:`Cherimoya.load` accept ``compile=False``, which skips
``torch.compile`` and uses the eager forward path:

.. code-block:: python

   from cherimoya import Cherimoya

   # When loading a trained checkpoint
   model = Cherimoya.load("checkpoint.torch", device="cuda",
                          compile=False)

   # When constructing a fresh model
   model = Cherimoya(n_filters=96, n_layers=9, compile=False)

The eager path still goes through Cherimoya's Triton inference
kernels — only the ``torch.compile`` wrapping is dropped — so forward
outputs are numerically equivalent. The test suite verifies parity at
``atol=rtol=1e-4`` between ``compile=True`` and ``compile=False``.

The performance cost is the compile speedup: typically ~10-20% on
the inference megakernel on recent GPUs, smaller on training. For
multi-instance inference loops the speedup is moot anyway because
the cache thrashes across instances.

**More general guidance.** Other ``torch.compile`` or CUDA-graph
issues can surface depending on your PyTorch version, GPU
architecture, and how aggressively a calling script reuses model
instances. **If you hit any** ``torch.compile`` **or CUDA-graph
error you don't immediately recognize, the safest fix is to load
the model with** ``compile=False``. You give up the compile speedup
but keep the Triton inference kernel and full numerical parity, and
you sidestep the entire class of compile/cudagraph foot-guns.


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
