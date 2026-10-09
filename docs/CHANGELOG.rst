Changelog
=========

v0.3.1 (unreleased)
-------------------

Attribution
~~~~~~~~~~~

* ``cherimoya attribute`` has a fast DeepLIFT/SHAP engine for the count
  head, ``"engine": "fast"``, in ``cherimoya.fast_deep_lift_shap``. It
  computes what tangermeme's ``deep_lift_shap`` computes with Cherimoya's
  rules, up to floating-point rounding: the same rules, the same
  dinucleotide-shuffled references with the same seeds, and the same
  hypothetical projection. It forwards each sequence once rather than
  once per reference, runs the backward over the sequence half of each
  pair only, and compiles its passes on a GPU. In an interleaved
  benchmark on one H100 shared with four, then five, other processes
  (three of them training runs), it attributed 948 sequence-reference
  pairs per second at steady state, with ``mem_budget_gb`` 7.5 and
  ``ref_workers`` 3, against the default engine's 106 at a
  ``batch_size`` of 20.
  Every run compares its forward with the model's before it starts and,
  after writing its outputs, attributes 8 evenly spaced sequences again
  with ``deep_lift_shap`` to compare (``audit``). It also writes the
  convergence deltas and a record of the run next to ``attr_filename``.
  The default engine is unchanged, and the profile head and saturation
  mutagenesis always use it. New keys: ``engine``, ``precision``,
  ``seqs_per_step``, ``mem_budget_gb``, ``ref_workers`` and ``audit``;
  ``engine`` is also in the pipeline's ``attribute_parameters``.

CLI and pipeline
~~~~~~~~~~~~~~~~

* The pipeline converts SAM/BAM and fragment files to bigWigs with
  ``figwig bam2bw`` instead of ``bam2bw``, and the dependency is
  ``figwig[bam2bw]>=0.1.0`` in place of ``bam2bw>=0.4.1``. ``figwig
  bam2bw`` takes bam2bw's arguments and writes the same bigWig entries,
  and reads local BAM and BED/tsv files faster: figwig's benchmarks give
  4.65 s against bam2bw 0.5.1's 58.48 s on a 2.4 GB ATAC-seq BAM at
  ``-p 2``.

* ``preprocessing_parameters.n_jobs``, 8 by default, is the number of
  cores the conversion runs on. The pipeline passed ``-p -1`` to bam2bw,
  where it meant one process per input file. figwig's ``-p`` is a
  number of cores, so ``-1`` would take every CPU of the machine, and
  figwig's benchmarks show no gain past 8 threads on that BAM (1.74 s at
  8, 1.82 s at 32).

Documentation
~~~~~~~~~~~~~

* ``paired_end`` (``-pe``) was documented as changing how bam2bw
  reconstructs fragments. It only sets the MACS3 format to ``BAMPE``,
  and the converter never receives it.

v0.3.0
------

This release moves training onto PyTorch Lightning, with training on
several GPUs; evaluates every model on a held-out test set and reports
how well it separates peaks from negatives; makes DeepLIFT/SHAP the
default attribution, with correct rules for Cherimoya's layers; and
fixes bugs that silently gave wrong results in the CLI, the kernels and
the seqlet coordinates. It contains everything since v0.2.0: v0.2.1 was
versioned but never tagged or published, and its entries are included
here.

**Checkpoints are unchanged.** Every checkpoint from an earlier version
loads and computes the same outputs, and a checkpoint written by this
version loads in earlier ones. Training is not bitwise identical to
v0.2.0 with the same seed, because the peak jitter now covers its full
range (below).

Breaking changes
~~~~~~~~~~~~~~~~

* ``Cherimoya.fit`` is removed. Training moved to PyTorch Lightning, and
  the replacement is :func:`cherimoya.training.fit`, which builds a
  ``lightning.Trainer`` around the new
  :class:`cherimoya.training.CherimoyaModule` and returns the trainer
  rather than the best validation correlation (that is
  ``trainer.checkpoint_callback.best_model_score``). The signature
  differs: ``training_data`` is the dataset, a
  :class:`~cherimoya.io.PeakNegativeSampler`, rather than a
  ``DataLoader``, since the module builds the loader itself; the three
  optimizers and three schedulers are no longer passed in but built
  from ``muon_lr``/``muon_wd``, ``adam_lr``/``adam_wd``,
  ``lw_lr``/``lw_wd``/``lw_momentum`` and the schedule lengths
  ``n_warmup_steps`` / ``n_decay_steps``; ``device`` is replaced by
  Lightning's ``accelerator`` and ``devices``; and ``dtype`` must be one
  of the strings ``'float32'``, ``'bfloat16'`` or ``'float16'``.
  **Code that calls** ``model.fit(...)`` **fails with**
  ``AttributeError``. The :doc:`tutorials/python_api` shows the
  replacement call with the CLI's schedule.

* ``cherimoya batch`` is removed, along with its subparser, its CLI
  reference section and the "Batch mode" section of the pipeline
  tutorial. **A script that invokes it now fails with an argparse
  error.** The equivalent is to write the per-experiment pipeline JSONs
  and run ``cherimoya pipeline`` on each, which is what ``batch`` did
  internally. ``joblib``, which only ``batch`` imported, is dropped from
  the dependencies.

* **``cherimoya fit`` evaluates on a held-out test set as well as the
  validation set, and its output files are renamed.** The validation
  chromosomes choose the checkpoint and drive early stopping, so the
  single ``{name}.performance.tsv`` they produced was not an
  independent estimate. A new ``test_chroms`` key (default ``chr1``,
  ``chr3``, ``chr6``, which the default split already left out of
  training and validation) is evaluated after training too.
  ``{name}.performance.tsv`` becomes
  ``{name}.validation.performance.tsv``, ``{name}.evaluate.json``
  becomes ``{name}.validation.evaluate.json``, and the test run writes
  ``{name}.test.performance.tsv`` and ``{name}.test.evaluate.json``;
  ``test_chroms: null`` skips it. **Scripts that read**
  ``{name}.performance.tsv`` **must switch to one of the new names.**

* **``fit`` refuses to start when two of** ``training_chroms``,
  ``validation_chroms`` **and** ``test_chroms`` **share a chromosome.**
  A fit JSON written before ``test_chroms`` existed gets the default
  ``chr1``/``chr3``/``chr6``, so one whose custom split trains on any of
  them now raises ``ValueError`` naming the shared chromosomes; set
  ``test_chroms`` to held-out chromosomes, or to ``null``.

* **``cherimoya attribute`` computes DeepLIFT/SHAP by default**, the way
  bpnet-lite's attribute step does. A new ``algorithm`` key takes
  ``"deep_lift_shap"`` (the default) or ``"saturation_mutagenesis"``,
  the previous behaviour; both write ``(n, 4, attr_window)`` arrays to
  the same files, so ``cherimoya seqlets`` and TF-MoDISco read either.
  DeepLIFT/SHAP attributes the whole ``in_window`` against
  dinucleotide-shuffled references with
  :func:`cherimoya.deep_lift_shap.attribution_ops` registered, and saves
  the centred slice. Three keys configure it, with bpnet-lite's
  defaults: ``n_shuffles`` (20), ``warning_threshold`` (0.001) and
  ``random_state`` (0; in a pipeline JSON, ``null`` inherits the
  top-level seed). Its ``batch_size`` default is 64 rather than 512:
  each item is a sequence-reference pair run forward and backward, and
  on a 9-layer, 128-filter model at 2114 bp, 512 pairs peaked at 124 GB
  of GPU memory and 64 at 15.5 GB, with the same wall time. The step no
  longer compiles the model by default (``compile: false``): neither
  algorithm ran faster compiled, and compiling added 6-70 s to the
  first call; DeepLIFT/SHAP never compiles, since its backward hooks
  cause graph breaks and recompiles.

* A new attribute ``group`` key (default ``0``) selects which signal
  group of a multi-group model is attributed (#52). **For a multi-group
  model this changes the default target** from every group -- ISM
  averaged the count head's outputs and the profile target softmaxed all
  groups together -- to group 0. ``"group": null`` restores the old
  target under saturation mutagenesis. Single-group models are
  unaffected.

* **Training is seeded by default.** ``random_state`` defaults to ``0``
  in the fit and pipeline JSONs, and :class:`cherimoya.Cherimoya` takes
  a ``random_state`` that seeds its initialization from a local
  ``torch.Generator``. The seed previously reached only the sampler, so
  two runs with the same seed started from different weights.
  **Rerunning an unchanged fit JSON now rebuilds the same model; vary**
  ``random_state`` **to get replicates.** ``random_state: null`` no
  longer means "unseeded": ``fit`` draws a seed, prints it, and records
  it in the evaluate JSONs, so the run can be repeated. A JSON that
  omits the key is accepted.

* **Peak jitter now reaches** ``+max_jitter``. The window offset was
  drawn from ``randint(0, 2 * max_jitter)``, which excludes its upper
  bound, so shifts ran from ``-max_jitter`` to ``max_jitter - 1``.
  Including both ends changes the sampler's random stream: **the same
  seed now draws different batches than before.**

* **Early stopping is off by default** (``early_stopping: null``; it was
  ``5``), and **a minimum step count overrides** ``max_epochs``: the new
  ``min_total_steps`` key (default 20000) raises ``max_epochs`` until the
  run reaches that many optimizer steps, and lays the warmup and cosine
  decay out over the raised value. An epoch is one pass over the peaks,
  so 20 epochs was 280 steps for an experiment with 14 batches of peaks
  and 54,000 for one with 2,700, and a patience counter on the EMA
  validation metric was ending runs partway through the decay.
  ``min_total_steps: null`` disables the floor.

* **``{name}.log``'s training columns are averaged over the epoch's
  batches.** They held whatever the last full batch produced, a
  single-batch estimate noisy enough to look flat or non-monotonic while
  the model improved; expect new logs to sit at a different level and
  move more smoothly. Thanks to Ethan Armand for the report in issue
  #19. The log also gains four columns before ``Saved?``, which stays
  last (see Evaluation).

* Keys removed because nothing read them, or because ``fit`` always
  overwrote them: the fit ``performance_filename``, the seqlets
  ``exclusion_lists`` and ``in_window``, ``attribute_parameters.out_window``
  and the pipeline's ``fit_parameters.count_loss_weight``. A JSON that
  still sets one is accepted and the key ignored.
  ``marginalize_parameters.output_folder`` is renamed to
  ``output_filename``, the key ``cherimoya marginalize`` reads.

* ``cherimoya pipeline-json`` requires ``-s``, ``-i``, ``-n`` and
  ``-o``, and ``cherimoya negatives`` requires ``-f``; omitting one used
  to produce a JSON of nulls or fail later with ``TypeError``.

Training
~~~~~~~~

* **Training can run on several GPUs.** ``devices`` is a new parameter
  of :func:`cherimoya.training.fit` and of the fit and pipeline JSONs,
  ``1`` by default; ``-1`` uses every visible device. More than one
  device trains with DDP. ``batch_size`` is the global batch, split
  evenly across the devices, and the new
  :class:`cherimoya.io.ShardedEpochSampler` gives each device a
  contiguous slice of every global batch, so each step sees exactly the
  examples one device would. Validation is split across the devices
  without padding and its metrics are computed over the whole set. With
  fixed ``loss_weights``, the read depths the profile loss is divided by
  are those of the global batch. Lightning starts every rank after the
  first by re-running the command, so ``cherimoya fit`` prints and
  evaluates on rank 0 only, and every rank uses rank 0's seed; under
  ``srun``, which starts every rank at once, each derives the same seed
  from ``SLURM_JOB_ID`` and ``SLURM_STEP_ID``. In ``cherimoya pipeline``,
  a ``devices`` other than 1 runs the fit step as its own ``python -m
  cherimoya_cli fit`` process. Training on several devices sets
  ``torch._dynamo.config.optimize_ddp = False``, because the DDP graph
  splitting it controls hung under the model's CUDA-graph compile mode.

* **The Kendall loss weights can be replaced by constants.** A
  ``loss_weights`` tuple, in :func:`~cherimoya.training.fit` and the fit
  and pipeline JSONs, replaces the learned ``lw0`` / ``lw1``, and the
  profile loss is then divided by each signal group's own batch-mean
  read depth, since the MNLL scales with depth and ``lw0``/``lw1`` hold
  one weight per group. ``(1.333, 0.274)`` reproduces the operating
  point the learned weights reach: +0.0001 median count Pearson over 44
  accessibility experiments (95% CI [-0.0012, +0.0011]), within 0.005 on
  two TF panels of 22 and 26, and on 24 four-experiment multi-task
  models +0.0021 per group over a pooled divisor (68 of 92 groups) and
  +0.0014 over the learned weights (59 of 92). ``verbose`` reports which
  scheme is in force.

* ``dtype='float16'`` scales the loss, through Lightning's
  ``16-mixed``; the previous loop ran fp16 under autocast with none.
  ``verbose`` prints the per-epoch table above Lightning's progress bar,
  and a ``progress_bar`` key (default ``null``) draws the bar only when
  stdout is a terminal or a Jupyter kernel. ``compile`` and
  ``compile_mode`` in a fit JSON apply to the training model and the
  evaluations.

* :func:`~cherimoya.training.fit` raises ``ValueError`` for a training
  set smaller than one global batch, an empty validation set, or one
  with fewer examples than devices, rather than training on nothing or
  failing with an unrelated error. An unseeded sampler on several
  devices gets rank 0's seed, rather than each rank taking its slice of
  a different epoch.

* :class:`~cherimoya.io.PeakNegativeSampler` rejects ``negative_ratio >
  0`` with an empty negative set at construction, rather than raising
  ``IndexError`` partway into the first epoch, and
  :func:`~cherimoya.io.PeakGenerator` accepts ``negatives=None`` for a
  peaks-only training set with ``negative_ratio=0``; it used to fail
  inside tangermeme.

* The CUDA limit on reproducibility is documented. The fused convolution
  + normalization kernel accumulates its statistics with atomic adds,
  whose order varies between launches, so two seeded GPU runs share an
  initialization and an example order but are not bitwise equal. CPU
  runs with the same seed are bitwise identical.

Evaluation
~~~~~~~~~~

* **Measures that use the negatives.** ``cherimoya fit`` adds every
  ``negatives`` locus on the ``validation_chroms`` to the validation set,
  and ``cherimoya evaluate`` reads an optional ``negatives`` key, which
  the evaluate JSONs ``fit`` writes carry. Both report the count Pearson
  and MSE over peaks and negatives together and the AUROC and AUPRC of
  the predicted log counts at separating peaks from negatives, per
  signal group; the existing measures and the checkpoint criterion stay
  on the peaks. ``{name}.log`` gains ``Validation Count Pearson
  (Peaks+Negatives)``, ``Validation Count MSE (Peaks+Negatives)``,
  ``Validation AUROC`` and ``Validation AUPRC`` before ``Saved?``;
  ``{name}.detailed.log`` gains ``AUROC_g{i}`` and ``AUPRC_g{i}``; and the
  performance TSVs gain ``all_count_pearson``, ``all_count_spearman``,
  ``all_count_mse``, ``auroc`` and ``auprc``. They are empty or ``nan``
  where there are no negatives. In Python, ``labels_valid`` marks the
  negative rows, and ``trainer.callback_metrics`` gains
  ``valid_count_pearson_all``, ``valid_count_mse_all``, ``valid_auroc``
  and ``valid_auprc``.

* **``reverse_complement_average`` scrambled the channels of multi-group
  models.** Flipping the whole channel axis averaged, for groups
  ``[1, 2]``, the ATAC prediction into the TF's minus strand. The
  reverse complement now swaps strands within each group, as training
  does. Single-group models are unchanged.

* With ``summits: true``, validation and evaluation windows were
  centered on the peak midpoint while training windows were centered on
  the summit; both now use it (``evaluate`` gains a ``summits`` key).
  ``evaluate`` declares ``signals``, returns with a message when no
  locus falls on its ``chroms``, and handles a negatives file with
  nothing on them.

* :func:`~cherimoya.performance.calculate_performance_measures` dropped
  ``signal_groups`` for its ``within_peak_`` measures, so for a
  multi-group model they used a different count target from the outer
  measures, and it scored every count against the all-channel total when
  ``signal_groups`` did not match the count head; it now passes the
  groups through and raises on a mismatch. ``labels`` is documented.

Attribution
~~~~~~~~~~~

* The fused dilated convolution + per-example norm inside
  :class:`cherimoya.CheriBlock` lives on a
  :class:`~cherimoya.cheri.FusedDilatedConvNorm` submodule
  (``block.conv``), so DeepLIFT and SHAP have a concrete node to hook.
  The kernel, the dispatch and the numerics are unchanged, and nothing is
  registered against the node by default, so existing results do not
  change. Thanks @watiss! (#34)

* :mod:`cherimoya.deep_lift_shap` provides the rule a Cherimoya model
  needs before ``tangermeme.deep_lift_shap.deep_lift_shap`` can attribute
  it correctly; :func:`~cherimoya.deep_lift_shap.attribution_ops`, now
  public, returns it for ``additional_nonlinear_ops``. ``conv_norm_op``
  recomputes the convolution, applies the closed-form normalization
  rule tangermeme uses for ``torch.nn.LayerNorm``, and pushes the result
  back through the linear convolution. It reproduces the attributions
  of the model rewritten as ``F.conv1d`` plus ``LayerNorm`` to 1.5e-08
  on CPU and 7.5e-09 on CUDA, faster than that rewrite (4.85 against
  5.78 ms per sequence for a 9-layer, 128-filter model on an H200).
  Treating the op as linear is not safe in general: one block over a
  short window leaves a convergence delta a quarter of the prediction.
  Requires ``tangermeme >= 1.5.0``. On CUDA, check agreement against a
  decomposed model rather than the convergence delta, which comes from
  PyTorch's CUDA path and is not stable between processes.

* The profile head's product of its logits and their softmax is a
  tangermeme ``BilinearOp``, so DeepLIFT/SHAP through
  :class:`cherimoya.ProfileWrapper` converges with tangermeme's own
  rules. As a bare multiplication it was treated as linear, leaving a
  convergence delta larger than the prediction on a 2-layer model; with
  ``BilinearOp`` it is 1.6e-09. An elementwise rescale rule for the
  softmax scaling, as first proposed, drops the change wherever a logit
  does not move, because the softmax couples positions (#83). Compared
  with that rule, against *in silico* saturation mutagenesis on six
  trained models, the median per-locus Pearson rose from 0.399-0.449 to 0.651-0.721 for two ATAC-seq
  and one DNase-seq model and from 0.828-0.880 to 0.875-0.894 for three
  TF ChIP-seq models (Wilcoxon signed-rank p <= 2.6e-09 each), and the
  CPU-CUDA difference on a CTCF model fell from 6.9e-02 to 7.9e-06 of
  the largest attribution (#83). Thanks @bjmt! Forward outputs are
  unchanged.

* :class:`cherimoya.ProfileWrapper` and :class:`cherimoya.LogCountWrapper`
  take an optional ``group`` index, so one modality of a multi-group
  model can be attributed on its own (#52), and
  ``ExpectedCountsWrapper(ControlWrapper(model))`` works:
  :class:`~cherimoya.ControlWrapper` exposes ``signal_groups``.

* **``cherimoya seqlets`` emitted coordinates 857 bases to the left of
  the seqlets it found.** It converted positions in the 400 bp
  attributed slice as though they were in the 2114 bp extraction
  window, so a seqlet at slice positions 100-110 of ``chr1:10000-11000``
  was written as ``chr1:9543-9553`` rather than ``chr1:10400-10410``.
  The window is now read from the saved arrays. This also affected the
  pipeline's tomtom-lite annotation, which reads sequence at those
  coordinates; TF-MoDISco reads the attribution arrays and was not
  affected. Re-run ``cherimoya seqlets`` on existing ``.npz`` files to
  correct a run. ``seqlets`` also writes an empty BED rather than
  raising when no seqlets are found.

* ``cherimoya attribute`` extracted tangermeme's default 2114 bp
  whatever ``in_window`` said, and ignored ``exclusion_lists``; both now
  reach ``extract_loci``. The attributed slice is a new ``attr_window``
  key (default 400, the previous hard-coded slice). Dropping loci near a
  chromosome end no longer raises ``IndexError``: the ambiguous-base
  filter is projected back into peak space. Thanks @ramirc0! (#31)

Kernels and model
~~~~~~~~~~~~~~~~~

* **The training kernels were never autotuned.** Their 12 configs
  passed ``num_warps`` and ``num_stages`` as kernel arguments, where
  Triton overrides them, so every config launched with 4 warps and 3
  stages. On an H200 a training forward+backward at batch 64 became 1.69x
  faster for a 512-filter model in bf16 and 1.04x in fp32; 128-filter
  models are unchanged.

* **The first backward at any new** ``(C, L)`` **returned a wrong
  gradient.** ``_bwd_apply_kernel`` overwrites the scratch buffer it
  reads, and autotune's trials ran it repeatedly, so the real launch
  read garbage; the depthwise weight gradient was off by up to 4.9e-01
  against CPU autograd. It now declares ``restore_value=['Conv_ptr']``,
  and the error is 2.9e-03 or less, TF32 in the surrounding ``Linear``
  layers (2.0e-06 at ``'highest'`` matmul precision). This mattered most
  for attribution, where a fresh process hits each shape once.

* **Applying an EMA snapshot to a model in eval mode left the Cheri
  Block's cached MLP weights stale**, so on CUDA the convolution ran on
  the snapshot and the MLP on the previous weights: profile logits
  1.9e-2 off, about 2.5% of the signal, for ``model.eval();
  ema.apply_shadow(model)``. :class:`~cherimoya.EMA` no longer writes
  through ``.data``, and ``CheriBlock`` checks its weights' version
  counters, falling back to an inline cast when they moved; under
  ``torch.compile`` the check is skipped and ``Cherimoya.forward``
  rebuilds stale caches before the compiled region, which avoids a graph
  break that made the compiled eval forward 6-17% slower on torch 2.12.

* :class:`~cherimoya.cheri.CheriBlock` on CUDA read the wrong elements of
  a non-contiguous input (full models were not affected); ``.eval()``
  raised on a model loaded inside ``torch.inference_mode()``; and the
  inference megakernel failed to compile below 16 filters, which now take
  the fallback path.

CLI and pipeline
~~~~~~~~~~~~~~~~

* **The pipeline's marginalization inserted motifs into the peaks.** It
  copied the top-level ``loci`` before falling back to ``negatives``, so
  the fallback never applied. ``cherimoya marginalize`` also extracted
  tangermeme's default 2114 bp rather than its ``in_window``, ignored
  ``exclusion_lists``, ignored its ``random_state``, and did not sample
  under ``shuffle`` -- it permuted the first ``n_loci`` rows of the file.
  **Marginalization reports now use different background loci.**

* ``"skip": true`` ended the whole run rather than the step, because
  the subcommands honoured it with ``sys.exit()``; a top-level ``skip``
  still ran MACS3, ``bam2bw``, negative sampling and TF-MoDISco; and
  ``annotation_parameters.skip`` was ignored. All three now behave as
  documented.

* ``evaluate``, ``attribute`` and ``marginalize`` accept ``compile`` and
  ``compile_mode``, and the pipeline's top-level values reach every step
  that loads a model, including training and the evaluations; the
  documented ``compile: false`` fix for CUDA-graph errors was not
  reachable from the CLI.

* Pipeline JSONs may omit ``motifs`` and ``model``, which used to raise
  ``KeyError`` after training or ``Must provide value``; the
  "Must provide value" message says to write ``null``; ``"dry_run":
  true`` no longer crashes with a motif database; a ``loci`` given as a
  string no longer makes the negatives step read its first character;
  and a second pipeline run in one process no longer inherits the
  first's output names.

* ``pipeline-json -sf`` is parsed as a float, the ``--unstranded`` help
  no longer says the opposite, and ``install-skill --force`` refuses a
  ``--directory`` that is the bundled skill itself.

Compatibility
~~~~~~~~~~~~~

* **Existing checkpoints are unaffected, bit for bit.** The depthwise
  weight is still written to and read from ``state_dict`` under its
  historical ``conv_weight`` key, even though the parameter now lives at
  ``block.conv.conv_weight``, and loading accepts either spelling.
  Checkpoints round-trip with earlier versions in both directions, and
  block initialization draws from the RNG in the same order. Thanks
  @watiss! (#34)

* The parameter's *name* changes: ``named_parameters()`` reports
  ``blocks.N.conv.conv_weight`` rather than ``blocks.N.conv_weight``, so
  a lookup by the exact old name raises ``KeyError``. Substring matches,
  including the optimizer routing, and ``model.get_parameter`` are
  unaffected. ``CheriBlock.conv_weight`` remains as a read-only alias.

Packaging and installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

* New dependencies: ``lightning>=2.6.1`` and ``scikit-learn>=1.7.2``.
  ``triton`` is a Linux-only dependency, since it publishes no wheels for
  macOS or Windows, which made cherimoya impossible to install there
  although the model runs without Triton. The ``tangermeme`` floor is
  ``>=1.5.0``; the old ``>=0.2.3`` admitted releases that could not run
  the package. The ``setuptools`` build floor is 77, which ``license =
  "MIT"`` needs.

* A container image is published to the GitHub Container Registry on
  every push to ``main`` (``ghcr.io/jmschrei/cherimoya``), built from
  ``uv.lock`` without dev dependencies. Thanks @ramirc0! (#27, #28, #29,
  #30)

* The installation guide covers ``sm_70`` GPUs such as the V100, which
  the default PyPI torch build no longer supports: install torch from
  the CUDA 12.6 index first. Thanks @ramirc0! (#81)

* A one-week ``exclude-newer`` window added to ``[tool.uv]`` with the
  0.2.1 version bump (#32) hid just-released dependencies from the
  resolver and is removed. Thanks @ramirc0! ``uv.lock`` now includes
  lightning, which the container needs to train.

* ``import cherimoya`` works from a source tree that was never installed,
  as on Read the Docs, where the training, DeepLIFT/SHAP and kernel API
  pages rendered empty. The sdist ships the whole test tree, and the
  README's images render on PyPI.

Documentation
~~~~~~~~~~~~~

* Training on several GPUs is documented in the CLI reference (GPU
  selection, memory, the schedule, SLURM), the Python tutorial,
  troubleshooting and the glossary. The README has a multi-GPU section
  with measured speedups.

* The three forward paths were documented as agreeing to "~1e-5
  max-abs". On the default model over 2114 bp, against a profile-logit
  scale of 0.73, they agree to 1.4e-04-2.5e-04 in fp32, 4.9e-04-5.9e-04
  in fp16 and 3.9e-03-5.0e-03 in bf16, and every page now says so. The
  receptive field is 1117 bp, not 1115.

* The docs, the bundled skill and the docstrings were checked against
  the code, and some forty statements corrected: defaults (``max_epochs``
  20, the ATAC shifts), file and key names, the autotune grids, the
  optimizers and loss weights, and advice to set ``min_counts`` and
  ``max_counts``, which ``fit`` never read. Links to the top-level
  re-exports such as :class:`cherimoya.Cherimoya` now resolve. Re-run
  ``cherimoya install-skill --force`` to pick up the skill's changes.

Tooling
~~~~~~~

* CI runs a strict Sphinx build and ``ruff``'s error-class rules on
  every pull request, and
  its pip cache is keyed on ``pyproject.toml``, so source-only
  dependencies are no longer rebuilt on every run. No hosted runner has
  a GPU; ``docs/development.rst`` says which checks, including the
  three-way forward parity, are run by hand.

* The test suite gained coverage for ``cherimoya negatives``, the
  attribute-to-seqlets seam, the compile keys, the pipeline's step
  wiring, the evaluate step after training and multi-device setup, and
  lost tests that could not fail, among them five compile-parity tests
  that ran eager on both sides under ``TORCH_COMPILE_DISABLE``. The CLI
  tests live under ``tests/commands/``, named after the modules they
  cover.

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
