Changelog
=========

v0.2.0
------

Tooling
~~~~~~~

* Bundled a Claude Code agent skill under
  ``cherimoya_cli/skills/cherimoya`` (a ``SKILL.md`` plus a
  ``references/`` set) and shipped it as package data, so it installs
  with the ``cherimoya`` package. Added a ``cherimoya install-skill``
  subcommand that copies (or, with ``--symlink``, links) the bundled
  skill into a Claude Code skills directory (``~/.claude/skills`` by
  default), with ``--directory`` to pick another location and
  ``--force`` to overwrite an existing install.

Loss (**breaking** for stranded/multi-channel models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed the profile loss so a multi-channel signal group is normalized
  as a **single multinomial over its channels and length jointly**,
  rather than one independent per-channel multinomial per strand then
  averaged. Previously the relative additive offset between a stranded
  ``(+, -)`` pair's logits was an unconstrained gauge (a per-channel
  ``log_softmax`` over length is invariant to a per-channel shift), so a
  trained model could place that offset arbitrarily. At inference,
  :class:`cherimoya.wrappers.ExpectedCountsWrapper` distributes a
  group's predicted counts with a *joint* softmax across the group's
  channels and positions, which exponentiates that arbitrary offset and
  collapses nearly all predicted signal onto a single strand — the
  symptom being stranded TF models whose predictions came almost
  entirely from one strand. The loss now matches the wrapper's joint
  normalization, so the strand balance is a trained quantity.
* **Single-channel (unstranded) models are unaffected — bit-for-bit.**
  A joint softmax over a one-channel group is identical to a per-channel
  softmax over length, so ATAC-seq / DNase-seq losses, gradients, and
  training trajectories are unchanged and existing accessibility
  checkpoints need no retraining. Only groups with two or more channels
  (stranded TF / co-trained stranded modalities) change, and those
  models should be **retrained** to benefit from the fix.
* ``cherimoya.performance.calculate_performance_measures`` is
  unchanged: ``profile_pearson`` / ``profile_spearman`` are invariant to
  per-channel vs. joint normalization (both operate over the length axis
  and are scale-invariant), and ``profile_jsd`` re-normalizes each
  channel internally, so reported metrics are identical.

Training defaults
~~~~~~~~~~~~~~~~~

* The default training ``batch_size`` is now 64 (was 192), in the
  ``Cherimoya.fit`` method, the ``cherimoya.io.PeakGenerator`` generator,
  and the CLI ``fit_parameters`` defaults used by ``cherimoya fit`` /
  ``cherimoya pipeline``. The smaller batch lowers the training-time
  memory footprint; reduce ``batch_size`` further to 32 or 16 if you
  still run out of GPU memory.

v0.1.1
------

Data pipeline (**breaking**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed a reverse-complement bug in
  :class:`cherimoya.io.PeakNegativeSampler` that scrambled tracks when
  training on a mix of unstranded and stranded signals (e.g.
  co-training ATAC with a stranded TF). Previously,
  ``torch.flip(yi, [0, 1])`` flipped both the channel dimension and the
  length dimension, which was only correct when *every* track was
  unstranded (no-op channel flip) or *every* track was part of one
  stranded pair (clean +/- swap). With a mix of three or more tracks
  the channel flip cross-wired the modalities. The sampler now applies
  a per-group channel permutation (precomputed once from the group
  structure) plus a length-only flip, so each group's internal
  channels are swapped independently and groups never bleed into one
  another.
* The ``signals`` and ``controls`` API now accepts a **grouped** form
  in addition to a flat list. Each entry of the outer list is one
  group — either a ``str`` (one-channel unstranded group) or a
  ``list[str]`` (multi-channel group, e.g. a stranded ``(+, -)``
  pair). Example::

      signals = ["atac.bw", ["ctcf.+.bw", "ctcf.-.bw"]]

  *Breaking semantic change:* a flat list of N files is now
  interpreted as N independent **unstranded** groups, not as a single
  N-channel block. BPNet-style callers that previously passed
  ``["plus.bw", "minus.bw"]`` as a stranded pair must update to the
  nested form ``[["plus.bw", "minus.bw"]]``.
* Added :func:`cherimoya.io.normalize_signal_groups` and
  :func:`cherimoya.io.channel_permutation_from_groups` as the public
  helpers callers can use to convert between the grouped form and the
  flat (file-list, group-sizes) form, and to derive the per-group RC
  permutation.
* :func:`cherimoya.io.PeakGenerator`'s outlier filter is now
  per-group: it computes one 99th-percentile-times-1.2 threshold per
  signal group and drops a locus if it's an outlier in *any* group.
  Previously the threshold was computed over the sum of counts across
  all channels and the full length, which collapsed distinct
  modalities into one number — a TF with peaks two orders of
  magnitude higher than a co-trained ATAC track would dominate the
  threshold. The single-group case reduces exactly to the legacy
  behavior.
* The ``cherimoya batch`` command's ``signals`` JSON field is now a
  list of *per-model* signal specs, with each entry itself in the new
  grouped form. Stranded batch jobs that previously wrote
  ``signals=[[plus, minus], [plus, minus]]`` (two stranded models)
  must now write ``signals=[[[plus, minus]], [[plus, minus]]]`` — see
  the batch section of :doc:`cli` for details.
* Training now writes two log files instead of one. ``{name}.log``
  is the existing summary log (same columns as before, printed to
  stdout when ``verbose=True``). ``{name}.detailed.log`` is a new
  disk-only TSV that extends the summary columns with one
  ``ProfilePearson_g{i}`` and one ``CountPearson_g{i}`` column per
  signal group — useful for offline per-modality analysis. The
  detail log never prints to stdout, so models with hundreds of
  groups still get a readable terminal. Best-model selection
  continues to use the mean-across-groups count Pearson and is
  unchanged.
* ``cherimoya evaluate`` writes one row per signal group to its
  performance TSV. The seven columns are unchanged
  (``profile_mnll``, ``profile_jsd``, ``profile_pearson``,
  ``profile_spearman``, ``count_pearson``, ``count_spearman``,
  ``count_mse``); rows are in ``signal_groups`` order. Single-group
  models write exactly one row, byte-identical to the legacy
  ``.mean()``-of-everything line. Multi-group models write N rows
  for N groups, with no extra identifier column — pair the rows
  with the model's ``signal_groups`` to recover which row belongs
  to which modality.
* Every signal group now contributes one term to the loss
  regardless of how many channels it has. ``_mixture_loss``'s
  profile component combined a stranded ``(+, -)`` pair's two
  per-strand MNLLs into one per-group profile loss before
  Kendall-Gal weighting (this per-channel averaging was later
  replaced by a joint per-group multinomial — see the Unreleased
  entry above); ``lw0`` drops from shape ``(sum(signal_groups),)``
  to ``(len(signal_groups),)``, matching ``lw1``. The summary log's
  ``Validation Profile Pearson`` now reports the mean over groups
  of (mean over the group's channels) so the headline metric
  agrees with the loss weighting — no double-counting of stranded
  pairs. Single-track models (``signal_groups=[1]``) are
  unaffected: every shape and value collapses to ``(1,)`` as
  before.

Model (**breaking**)
~~~~~~~~~~~~~~~~~~~~

* The ``Cherimoya`` constructor now takes ``signal_groups`` (list of
  per-group channel counts) instead of ``n_outputs``.
  ``signal_groups`` controls both the profile head width
  (``sum(signal_groups)``) and the count head width (always
  ``len(signal_groups)``). So a stranded ``(+, -)`` pair emits two
  profile channels but a single count prediction — the per-strand
  counts are always tied. ``n_outputs`` is removed as a constructor
  kwarg; ``model.n_outputs`` is retained as a derived attribute equal
  to ``sum(signal_groups)``.
* Removed the ``single_count_output`` constructor flag. The count head
  is now always one prediction per signal group; the legacy
  "collapse every channel into one shared scalar" mode is gone
  because in the grouped formulation it conflates distinct biological
  modalities.
* Pre-grouping checkpoints (whose ``config`` dict stored ``n_outputs``
  / ``single_count_output``) no longer load. The project is too early
  to carry a back-compat shim; retrain with the new API.
* :func:`cherimoya.losses._mixture_loss` and
  :func:`cherimoya.performance.calculate_performance_measures` both
  accept an optional ``signal_groups`` argument. When supplied, the
  true counts are pooled per group before the count loss / count
  Pearson are computed, so a stranded pair contributes a single
  per-group target instead of one per strand.
* The profile head (``fconv``) is now a 75-bp convolution
  (``kernel_size=75``, padding 37) instead of a 1×1 pointwise
  convolution. The padding keeps it length-preserving, so the output
  window is still ``in_window - 2 * trimming`` and stays positionally
  aligned with the target; the wider kernel gives the head a local
  receptive field (37 bp each side) that matches the ``46`` constant in
  the default ``trimming``. Checkpoints saved with the 1×1 head do not
  load — the ``fconv`` weight shape changed from ``(n_outputs,
  n_filters, 1)`` to ``(n_outputs, n_filters, 75)``; retrain with the
  new head. For the default single-output model this adds ~9.5K
  parameters (``128 * 75`` vs ``128``), bringing the default 9-layer,
  128-filter model to ~610K parameters total.

Training defaults
~~~~~~~~~~~~~~~~~

* The default backbone width ``n_filters`` is now 128 (was 96), so the
  default 9-layer model has roughly 600K parameters (was ~340K). This
  applies to the ``Cherimoya`` constructor and the ``fit_parameters``
  defaults used by ``cherimoya fit`` / ``cherimoya pipeline``.
* The default training ``batch_size`` is now 192 (was 128), in both the
  ``Cherimoya.fit`` method, the ``cherimoya.io.PeakGenerator`` generator,
  and the CLI ``fit_parameters`` defaults. The 128-filter, 192-batch
  defaults still fit comfortably on a 16 GB GPU; reduce ``batch_size`` to
  128 or 64 if you run out of GPU memory.
* The default ``negative_ratio`` is now 0.25 (was 0.02), in both the
  ``cherimoya.io.PeakGenerator`` generator and the CLI ``fit_parameters``
  defaults, sampling more GC-matched background loci per peak each epoch.
* The default ``max_jitter`` for fitting is now 500 bp (was 50), in both
  the ``cherimoya.io.PeakGenerator`` generator and the CLI
  ``fit_parameters`` defaults. The jitter is absorbed by the flank
  between the default ``in_window`` (2114) and ``out_window`` (1000).

v0.1.0
------

Model
~~~~~

* Added a fully fused **forward-only inference megakernel** for the
  Cheri Block: conv + norm + MLP + residual in two GPU passes, with
  bf16 dot products. Used automatically when
  ``torch.is_grad_enabled()`` is ``False`` and the MLP hidden width is
  a multiple of 16, with automatic fallback to the training Triton
  path otherwise. Numerically equivalent to the training path within
  ~1e-5 max-abs at unit-scale outputs, and roughly 1.9× faster than
  the training-fwd path on H200 at the default model size.
* The inference megakernel's bf16 weight cast is now materialized at
  ``.eval()`` time as non-persistent buffers and refreshed by a
  ``load_state_dict`` post-hook, instead of cached inside the
  compiled forward. This fixes a
  ``RuntimeError: accessing tensor output of CUDAGraphs that has been
  overwritten`` that previously surfaced when running multiple model
  instances or reloading weights mid-process, and removes the need
  for ``compile=False`` / ``compile_mode='max-autotune-no-cudagraphs'``
  as a workaround for that specific error. **User-visible
  consequence:** call ``model.eval()`` before inference to hit the
  fast path; the megakernel still runs without ``.eval()`` but
  recomputes the cast inline per call (adds ~10-27% at small batch,
  under ~2% at production batch). See :doc:`benchmarks` for the
  breakdown.
* Generalized the Kendall-Gal loss-weight parameters ``lw0`` and
  ``lw1`` from scalars to per-track vectors. ``lw0`` is now shape
  ``(n_outputs,)`` (one weight per profile track) and ``lw1`` is shape
  ``(n_count_outputs,)`` (one weight per count-head output). For
  single-task models both shapes are ``(1,)``, matching the format of
  every pre-vector checkpoint — existing single-task checkpoints load
  without changes. The freeze threshold now uses
  ``|grad(lw0)|.mean() < 1`` so it doesn't scale with track count.
  ``_mixture_loss`` correspondingly returns per-track loss vectors
  instead of scalars.
* The training Triton kernel and the CPU fallback are unchanged.
  Existing trained checkpoints are bit-compatible.
* Replaced the learnable channel-wise scaling with a fixed
  ``residual_scale`` constant (default 0.15).
* Added an exponential moving average (EMA) of model weights during
  training; validation and saved checkpoints use the EMA-applied
  weights.
* Changed the final profile convolution to ``kernel_width=1``.
* Set the default model size to 96 filters.
* Tuned the Muon and AdamW learning rates and weight decay values
  for improved convergence (Muon ``lr=0.025, wd=0.01``; AdamW
  ``lr=0.004, wd=0.2``).
* Best-model selection now monitors the validation count Pearson
  correlation rather than the total validation loss.

API
~~~

* ``Cherimoya.save`` / ``Cherimoya.load`` checkpoints now use a
  config + state_dict payload that is robust to source-layout
  changes and loads with PyTorch's ``weights_only=True``. Older
  pickle-based checkpoints (``torch.save(model, ...)``) are not
  compatible and must be migrated or retrained.
* :class:`cherimoya.cherimoya.EMA` is now a public top-level symbol
  alongside :class:`cherimoya.Cherimoya` and
  :class:`cherimoya.CheriBlock`.
* Added a :mod:`cherimoya.wrappers` module exposing four public
  wrappers: :class:`cherimoya.ControlWrapper`,
  :class:`cherimoya.ProfileWrapper`, :class:`cherimoya.LogCountWrapper`,
  and :class:`cherimoya.ExpectedCountsWrapper`. ``ControlWrapper`` and
  ``ProfileWrapper`` are drop-in ports of the bpnet-lite wrappers;
  ``LogCountWrapper`` returns the per-group log-counts; and
  ``ExpectedCountsWrapper`` distributes each group's counts (``expm1``
  of the log-count) across its channels and positions via a joint
  softmax, so the expected counts summed over a group equal its
  predicted count. ``cherimoya attribute`` and ``cherimoya marginalize``
  now use these in place of ``bpnetlite``'s ``ControlWrapper``,
  ``CountWrapper``, and ``ProfileWrapper``, so the subcommands no longer
  import any wrappers from bpnet-lite.

Training
~~~~~~~~

* Default ``max_jitter`` for fitting lowered from 500 to 50.

Packaging and tooling
~~~~~~~~~~~~~~~~~~~~~

* Migrated from ``setup.py`` to ``pyproject.toml`` with ``uv``
  support.
* Refactored the CLI from a monolithic script into the
  ``cherimoya_cli`` modular package.
* Raised the minimum Python version to 3.10 and minimum PyTorch
  to 2.9.
* Added ``macs3``, ``bam2bw``, ``bpnet-lite``, ``triton``, and
  ``joblib`` as dependencies.
* Added a Sphinx documentation site hosted on Read the Docs.

v0.0.1
------

* Initial release of the Cherimoya model and pipeline.
* Includes the ``CheriBlock`` architecture and custom kernels.
* Features a dual-optimizer training strategy (AdamW + Muon).
* Implements a full end-to-end processing and modeling pipeline.
