Changelog
=========

Unreleased
----------

Attribution
~~~~~~~~~~~

* Adds :mod:`cherimoya.deep_lift_shap`, the DeepLIFT rules a Cherimoya
  model needs before ``tangermeme.deep_lift_shap.deep_lift_shap`` can
  attribute it correctly. ``deep_lift_shap`` corrects the module types
  it knows and treats the rest as linear, and two of Cherimoya's layers
  are neither known nor linear, so this is a correctness question rather
  than a performance one. ``attribution_ops()`` returns both rules for
  ``additional_nonlinear_ops``. Requires ``tangermeme >= 1.5.0``, where
  the closed-form normalization rule it reuses was added.

* ``conv_norm_op`` gives :class:`~cherimoya.cheri.FusedDilatedConvNorm` a
  closed-form rule without giving up the Triton kernel. The fused op is a
  normalization applied to a depthwise convolution; the convolution is
  linear, so only the normalization needs a rule, and the closed form for
  that already exists for ``torch.nn.LayerNorm`` — the same operation the
  kernel performs. The rule recomputes the convolution from the cached
  input, evaluates the normalization multiplier at its output, and pushes
  the result back through the convolution with autograd, which is exact
  because the convolution is linear. It reproduces the attributions of a
  model rewritten as ``F.conv1d`` plus ``torch.nn.LayerNorm`` to 1.5e-08
  on CPU and 7.5e-09 on CUDA, and costs 6.45 ms per sequence against that
  rewrite's 10.24 ms, since the rewrite gives up the kernel for the whole
  forward and backward. ``integrated_gradients_op`` also converges but is
  a path integral rather than the closed form, and lands 2.78% away from
  it on both devices.

  Registering nothing is not safe in general. The op is genuinely
  non-linear, and whether it can be treated as linear depends on how far
  the normalization's statistics move between a sequence and its
  reference, which is a property of the model and the inputs rather than
  of the layer. A 9-layer model over 2114bp reduces over 270,000 elements
  per statistic and barely moves them; one block over a short window
  leaves a convergence delta a third of the size of the prediction.

* On CUDA the convergence delta is not a usable check for this op. Which
  Triton config autotune settles on varies between processes, and the two
  it picks from move the delta between 1.0e-07 and 3.3e-04 on the test
  fixture -- two discrete values, 2 runs in 8 on a fixed GPU, and warming
  the kernel first does not change it. The attributions are unaffected,
  matching the decomposed model to 7.5e-09 or better under either config,
  because what differs cancels in the channel-wise projection. Check
  agreement against a decomposed model rather than the delta when
  validating on GPU.

* ``attribution_ops`` also registers the profile head's
  ``_ProfileLogitScaling`` with tangermeme's elementwise rescale rule.
  Left unregistered it is treated as linear, which on a 2-layer model
  leaves a convergence delta of 8.0e-04 against a prediction of 3.7e-04
  — the error exceeds the signal, so the attributions carry no
  information about their own scale. Registering it brings the delta to
  1.3e-09. ``conv_norm_op`` alone does not reach this; the count head is
  unaffected.

* ``_cheri_conv`` splits the 3-tap depthwise dilated convolution out of
  ``_cheri_conv_norm_cpu``, so the CPU reference and ``conv_norm_op``
  share one definition of the weight layout and the padding instead of
  two that can drift. No behavior change.

Reproducibility
~~~~~~~~~~~~~~~
Removed (**breaking**)
~~~~~~~~~~~~~~~~~~~~~~

* ``cherimoya batch`` is removed, along with
  ``cherimoya_cli/commands/batch.py``, its subparser, its CLI reference
  section and the "Batch mode" section of the pipeline tutorial. It fanned
  one JSON out into several pipeline JSONs and ran them with
  ``joblib.Parallel``, one per CUDA device.

  **A script that invokes** ``cherimoya batch`` **will now fail with an
  argparse error naming the valid subcommands.** The equivalent is to write
  the per-experiment pipeline JSONs yourself and run ``cherimoya pipeline``
  on each, which is what ``batch`` did internally — it assigned devices
  round-robin by ``i % len(device)`` and shelled out to
  ``cherimoya pipeline -p {name}.pipeline.json``.

* ``joblib`` is dropped from ``dependencies``. ``batch.py`` was the only
  thing in the package that imported it. It usually remains installed
  anyway, as a transitive dependency of scikit-learn.

Bug fixes
~~~~~~~~~

* ``cherimoya seqlets`` raised ``IndexError: arrays used as indices must
  be of integer or boolean type`` when no seqlets were found. An empty
  result is a legitimate outcome — a weak model, or a strict
  ``threshold`` — but the empty frame ``recursive_seqlets`` returns has
  ``object`` dtype columns, and indexing the locus table with an object
  array raises inside pandas rather than producing an empty result.
  Inside ``cherimoya pipeline`` that ended the run after training had
  already finished. An empty BED is now written; the annotation step
  opens that path either way.

* ``cherimoya seqlets`` emitted genomic coordinates shifted 857 bases to
  the left of the seqlets it found. ``cherimoya attribute`` scores a
  400bp slice centred inside the 2114bp extraction window and saves the
  attributions over that slice only, so a seqlet position is an offset
  into the slice; the conversion back to the genome centred a window of
  ``in_window`` (2114) on each locus instead of one of the slice's own
  width (400), which puts the reported start
  ``(2114 - 400) / 2 = 857`` bases early. A seqlet at slice positions
  100-110 of a peak at ``chr1:10000-11000`` was written as
  ``chr1:9543-9553`` rather than ``chr1:10400-10410``, outside the peak
  it came from. The window is now read from the width of the
  attribution array, so it stays correct if a run attributes a
  different slice. ``seqlet_parameters.in_window`` is removed; it was
  only ever used for this conversion. A JSON that still sets it is
  passed through and ignored.

  This also affected the pipeline's motif annotation: ``ttl`` reads the
  seqlet BED to pull sequence out of the FASTA, so
  ``{name}.seqlets_annotated.bed`` and ``{name}.motif_seqlet_count.tsv``
  were built from the wrong sequence. TF-MoDISco is unaffected — it
  reads the attribution ``.npz`` files directly and never sees these
  coordinates. Re-run ``cherimoya seqlets`` over the existing
  ``.ohe.npz`` / ``.attr.npz`` files to correct an affected run without
  recomputing attributions.

* The first backward at any new ``(C, L)`` shape returned a wrong
  gradient. ``_bwd_apply_kernel`` reads the convolution out of a
  scratch buffer and writes the normalization gradient back over it, so
  it is not idempotent — and ``triton.autotune`` benchmarks a
  configuration by running the kernel repeatedly, so every trial after
  the first read the previous trial's output as though it were the
  convolution, leaving the buffer garbage before the real launch. The
  kernel now declares ``restore_value=['Conv_ptr']``, so Triton
  snapshots that buffer and restores it before each trial.

  Measured against CPU autograd on shapes tuned fresh in their own
  process, the depthwise weight gradient:

  .. list-table::
     :header-rows: 1

     * - shape
       - before
       - after
     * - ``C=48, L=176``
       - 1.60e-01
       - 8.52e-04
     * - ``C=80, L=208``
       - 4.86e-01
       - 2.43e-03
     * - ``C=112, L=144``
       - 3.90e-01
       - 2.90e-03

  The remaining difference is TF32 in the surrounding ``Linear``
  layers, not the kernel: with ``torch.set_float32_matmul_precision
  ('highest')`` it falls to 2.03e-06. The ``linear1`` and ``linear2``
  gradients were never affected, since they do not pass through that
  buffer.

  This was documented as a known wart with "warm up the kernel before
  reading gradients" as the workaround, and estimated at ~7e-2; it is
  larger than that and it is now fixed. It mattered most for
  attribution, which is gradient-based — a fresh process attributing
  one shape hits this path exactly once, with nothing after it to
  notice.

* Applying an EMA snapshot to a model already in eval mode left the
  Cheri Block's cached MLP weights holding the *previous* weights, so on
  CUDA the depthwise convolution ran on the EMA snapshot while the MLP
  ran on whatever was loaded before it. :meth:`cherimoya.EMA.apply_shadow`
  and :meth:`~cherimoya.EMA.restore` wrote through ``Tensor.data``, which
  is precisely the assignment that does not advance a tensor's version
  counter, and ``CheriBlock.train(False)`` materializes bf16 casts of the
  MLP weights that only ``train()``/``eval()`` and a ``load_state_dict``
  post-hook refreshed. Measured on a 9-layer, 128-filter model at fp32
  on CUDA, the documented ``model.eval(); ema.apply_shadow(model)``
  sequence produced profile logits **1.9e-2** away from the same weights
  evaluated with a fresh cache, against an output scale of 0.77 — about
  2.5% of the signal. It is now exactly 0.

  ``cherimoya fit`` was not affected in practice: ``tangermeme.predict``
  calls ``model.eval()`` inside ``_preserve_model_state``, after the
  swap, which happened to rebuild the cache. Direct users of
  :class:`~cherimoya.EMA` were, and so was any code that writes a weight
  in place while in eval mode — an optimizer step, a hand-edited tensor.

  Two changes. ``EMA`` no longer writes through ``.data``, so the swap is
  visible to anything watching the version counter; both methods were
  already under ``no_grad``, so nothing else about them changes. And
  ``CheriBlock`` records the version counters its cache was built from
  and falls back to an inline cast when they have moved, which is the
  same path a block already takes when the inference kernel is reached
  without a prior ``.eval()``. Calling ``.eval()`` again rebuilds the
  cache and restores the fast path.

  Only the fp32 cache branch was affected. Under ``torch.autocast`` the
  block's input dtype does not match the cached cast, so the inline path
  was already being taken and fp16/bf16 results were correct.

* The version check added by the entry above broke the compiled eval
  forward on torch 2.12. Dynamo cannot compare ``Tensor._version``
  values without a graph break, and a break inside the block loop sends
  all of ``Cherimoya._forward_impl`` to eager and compiles each
  ``CheriBlock`` on its own, once per dilation, until it hits
  ``torch._dynamo hit config.recompile_limit (8)``, a warning printed
  during validation in ``cherimoya fit``. Results stayed correct,
  but the eval forward was 6-17% slower than a single graph (0.59 vs
  0.66 ms for 32 filters, 1.34 vs 1.42 ms for 64, 2.88 vs 3.09 ms for
  128 at fp32, 1.77 vs 2.08 ms for 128 under bf16 autocast; batch 64, 9
  layers). torch 2.13 traces the comparison and was not affected.

  ``CheriBlock`` now skips the check under ``torch.compile`` and trusts
  its cache, and ``Cherimoya.forward``, which runs outside the compiled
  region, rebuilds the cache of any block whose weights moved before
  dispatching. The EMA swap is still seen, now without falling back to
  the inline cast. A ``CheriBlock`` compiled on its own, or a
  ``Cherimoya`` wrapped in a user's own ``torch.compile``, no longer
  detects in-place weight writes made after ``.eval()``; call
  ``.eval()`` again after such a write.

* A hand-written ``pipeline`` JSON that omitted ``motifs`` raised
  ``KeyError: 'motifs'`` at the seqlet annotation step — after the model
  had already been trained. ``pipeline.run`` reads
  ``parameters["motifs"]`` unguarded for the annotation, the MoDISco
  report and the marginalization step, but ``motifs`` was not a declared
  default, so only a JSON emitted by ``cherimoya pipeline-json`` (which
  always writes the key) had it. ``motifs`` is now a top-level pipeline
  default, documented in the CLI reference, and may be omitted.

* A ``pipeline`` JSON that omitted ``model`` was rejected with ``Must
  provide value for 'model'``, even though ``pipeline.run`` treats a
  null model as "train one" and that is the only thing the key does.
  ``model`` and ``motifs`` are now both omittable.

* ``merge_parameters`` now says what to write when a required key is
  missing. ``null`` is accepted and an absent key is not, which was not
  guessable from ``Must provide value for 'x'``; the message now adds
  "Set it to null if this step is supposed to produce it."

* The CLI reference claimed that any key missing from a JSON falls back
  to its default. That was false for every key whose default is
  ``null``, which is most of the input paths. The "Common conventions"
  section now states which keys must be present and which are genuinely
  optional.

* ``cherimoya attribute`` never passed ``in_window`` to ``extract_loci``,
  so every run extracted ``tangermeme``'s own default of 2114bp no
  matter what the JSON said. A model trained at a different input
  window was therefore fed the wrong window — and the key was declared
  in the schema and documented in the CLI reference the whole time. It
  is now the window that is actually extracted.

* The attributed slice was a hard-coded ``mid - 200, mid + 200`` with no
  key controlling it. It is now ``attr_window``, defaulting to 400 so
  existing runs produce byte-identical output, and validated against
  ``in_window`` rather than silently producing an out-of-range slice.
  ``cherimoya seqlets`` reads this width back off the saved arrays, so
  changing it needs no matching setting there.

* ``attribute_parameters.out_window`` is removed. The step extracts
  sequence only, never signal, so there was no output window to size. A
  JSON that still sets it is passed through and ignored.

Robustness
~~~~~~~~~~

* ``Cherimoya.fit`` now warns when an epoch produces no full batch. The
  loop skips any batch whose size is not exactly ``batch_size``, so a
  ``batch_size`` that disagrees with the DataLoader's own silently
  skips *every* batch: the run completes for the full ``max_epochs``
  having taken no optimizer step, saves a checkpoint and returns a best
  correlation, with the only evidence a nan in two columns of the log.
  A single empty epoch remains a supported outcome — a training set
  smaller than one batch produces it — so this warns rather than
  raising.

* ``Cherimoya.fit`` rejects a missing ``X_valid`` or ``y_valid`` with a
  message naming them. Validation is what selects the saved checkpoint
  and what the returned correlation is computed from, so it is not
  optional; passing None used to fail inside ``tangermeme.predict``
  with an error mentioning neither argument. A vestigial
  ``y_valid_counts`` computation, guarded on ``X_valid is not None``
  and never read, is removed.

* :class:`~cherimoya.io.PeakNegativeSampler` rejects
  ``negative_ratio > 0`` with an empty negative set at construction.
  Those slots can only be filled from the negative set, so the sampler
  used to raise ``IndexError`` partway into the first epoch, naming
  neither the ratio nor the set.

* ``spearman_corr``'s docstring said it used a dense ordering. It uses
  ``argsort().argsort()``, which is an ordinal ranking — every element
  gets a distinct rank and ties are broken by position rather than
  shared.

CLI
~~~

* ``default_pipeline_parameters['marginalize_parameters']`` declared
  ``output_folder`` while ``cherimoya marginalize`` reads
  ``output_filename``, so setting it in a pipeline JSON was a silent
  no-op and the report landed in the default location anyway. The
  declared key is now ``output_filename``, which is what the CLI
  reference already documented. ``modisco_report_parameters`` keeps its
  ``output_folder``; that one is read.

* Removed ``count_loss_weight`` from the pipeline's ``fit_parameters``
  and from ``merge_parameters``'s omittable list. Nothing read it —
  not ``fit``, not the model, not the loss. ``loss_weights`` is the key
  that sets fixed profile and count weights.

* Documented why ``default_fit_parameters['reverse_complement_average']``
  is not dead, since it reads that way: ``fit`` never uses it, but
  deepcopies its parameters into the evaluate JSON it generates when
  training finishes, and ``evaluate`` does read it.
  

* ``cherimoya marginalize``'s ``shuffle`` did not sample. ``extract_loci``
  stops as soon as it has ``n_loci`` usable sequences, i.e. it returns
  the first ``n_loci`` rows of the file, and the shuffle ran *after*
  that — so it permuted a set already chosen by file order and the
  truncation that followed was a no-op. Every marginalization report was
  built from the top of the background BED, in a seed-dependent order,
  which is the one thing ``shuffle`` exists to avoid. The extraction is
  no longer capped when shuffling, so the sample is drawn from the whole
  file. **Reports produced with** ``shuffle: true`` **will now use
  different background loci**; the unshuffled path is unchanged.

  Drawing a sample means reading the population, so the shuffled path
  now holds the full locus set in memory. The unshuffled path still
  stops at ``n_loci``.

* ``calculate_performance_measures`` dropped ``signal_groups`` when
  recursing to compute the ``within_peak_`` measures, so for a
  multi-group model those fell through to the legacy "sum every channel
  into one total" count target while the outer measures pooled counts
  per group. The two then described different quantities under names
  that read as the same measure on different rows, and because
  ``pearson_corr`` broadcasts the collapsed target back up to one value
  per prediction column, the wrong numbers also arrived in the right
  shape. No in-repo caller passes ``labels``, so no CLI output changes;
  this affects external callers of the function.

* ``labels`` was an undocumented parameter of
  ``calculate_performance_measures``. It now has a ``Parameters`` entry,
  including the detail that ``auprc`` and ``auroc`` are scored against
  the first count output only and so describe group 0 rather than the
  whole model when there is more than one group.

* ``"dry_run": true`` crashed with ``FileNotFoundError`` on any pipeline
  configured with a motif database. The seqlet annotation step guards
  the ``ttl`` subprocess behind ``dry_run`` but read that subprocess's
  output with ``pandas.read_csv`` outside the guard, so the dry run
  looked for an annotation file it had deliberately not produced. Since
  running with a motif database is the common case, the documented way
  to check a config before committing GPU time to it did not work. The
  tally is now inside the same guard.

* ``ExpectedCountsWrapper(ControlWrapper(model))`` raised
  ``AttributeError: 'ControlWrapper' object has no attribute
  'signal_groups'``. :class:`~cherimoya.ControlWrapper` is documented as
  the inner wrapper the output wrappers are layered on top of, and
  ``cherimoya attribute`` builds exactly that stack, but
  ``torch.nn.Module`` does not forward attribute lookups to submodules,
  so the one output wrapper that reads the model's grouping could not be
  used over it. ``ControlWrapper`` now exposes ``signal_groups`` from
  the model it wraps. :class:`~cherimoya.ProfileWrapper` and
  :class:`~cherimoya.LogCountWrapper` were unaffected — they read no
  model configuration.

* ``evaluate``, ``attribute`` and ``marginalize`` accept ``compile`` and
  ``compile_mode``, passed through to :meth:`cherimoya.Cherimoya.load`.
  Both default to ``load``'s own values, so nothing changes for a JSON
  that does not set them. Setting either at the top level of a
  ``pipeline`` JSON reaches every step that loads a model, the same way
  ``dtype`` and ``device`` do.

  The troubleshooting page and the bundled skill both recommend
  ``compile=False`` when a run hits a ``torch.compile`` or CUDA-graph
  error, and the DeepLIFT documentation recommends it for attribution
  because Inductor cannot trace the backward hooks. None of that was
  reachable from the CLI, which always loaded with the default
  ``compile=True, compile_mode='max-autotune'``.

* ``"skip": true`` ended the whole run rather than the step.
  ``cherimoya pipeline`` calls each subcommand's ``run(args)``
  in-process, and ``fit``, ``evaluate``, ``attribute``, ``seqlets`` and
  ``marginalize`` all honoured ``skip`` with ``sys.exit()``, which
  terminates the interpreter. They now return, so the pipeline moves on
  to the next step — which is what the key is documented to do. The
  marginalization guard in ``pipeline`` and the tail of
  ``pipeline-json`` returned the same way.

* ``cherimoya pipeline-json`` now requires ``-s``, ``-i``, ``-n`` and
  ``-o``. None were enforced, so omitting one either wrote a JSON full
  of nulls, produced ``None_*`` filenames several steps later, or
  failed with ``TypeError: expected str, bytes or os.PathLike object,
  not NoneType`` from ``open(None)``. **A script that relied on
  omitting one of these will now fail at parse time** with a message
  naming the missing flag.

* Removed a dead ``add_parser`` in ``commands/install_skill.py``. The
  real parser is built in ``__main__.py``; the duplicate was never
  called and could only drift.

Training
~~~~~~~~

* **The Kendall loss weights can be replaced by constants.**
  :meth:`cherimoya.Cherimoya.fit` takes a ``loss_weights`` tuple, exposed
  as ``loss_weights`` in the fit and pipeline JSONs, which replaces the
  learned ``lw0`` / ``lw1`` with fixed values and stops the ``lw_*``
  optimizer receiving gradient. The default is ``None``, which keeps the
  existing behaviour.

  When set, the profile loss is first divided by **each signal group's own**
  batch-mean read depth. That division is the point: ``lw0`` and ``lw1`` are
  ``Parameter(torch.ones(n_groups))``, so on a multi-task model the Kendall
  mechanism learns one weight per experiment, and the profile MNLL is a sum
  of per-read log-likelihoods that scales with read depth. Dividing every
  group by one pooled number would rescale them all equally and leave their
  weights relative to each other untouched. The count MSE needs no such
  division: it is computed on ``log1p`` counts, where depth is an additive
  shift the model absorbs into its bias.

  ``cherimoya fit`` reports which scheme is in force. With ``verbose``
  set it printed ``SGD Optimizer (lw): lr=..., wd=..., momentum=...``
  unconditionally, which describes an optimizer that takes no effective
  step once ``loss_weights`` is given. It now prints ``Fixed Loss
  Weights: profile=..., count=...`` instead when the weights are fixed.

  ``(1.333, 0.274)`` reproduces the operating point the learned weights
  reach. On single-experiment models this is free: +0.0001 median count
  Pearson over 44 accessibility experiments (95% CI [-0.0012, +0.0011]) and
  within 0.005 on two TF panels of 22 and 26. On 24 four-experiment
  multi-task models it is +0.0021 per group over a pooled divisor
  (68 of 92 groups) and +0.0014 over the learned weights (59 of 92).
  ``lw0`` and ``lw1`` remain on the model, so checkpoints are unaffected.

* **A minimum optimizer step count now overrides** ``max_epochs``.
  ``min_total_steps`` is a new CLI parameter, ``20000`` by default in
  ``default_fit_parameters`` and in the ``fit_parameters`` block of
  ``default_pipeline_parameters``. An epoch is one pass over the peaks,
  so ``max_epochs`` alone buys a number of optimizer steps proportional
  to how many peaks an experiment has -- 20 epochs is 280 steps for an
  experiment with 14 batches of peaks and 54,000 for one with 2,700.
  ``cherimoya fit`` now raises ``max_epochs`` until the run reaches
  ``min_total_steps``, and lays the warmup and cosine decay out over the
  raised value so the schedule stretches with the run rather than
  decaying inside the original budget. The new epoch count and total
  step count are printed when ``verbose`` is set. Experiments that
  already clear the floor are untouched, and ``min_total_steps: null``
  disables it. :meth:`cherimoya.Cherimoya.fit` is unchanged.

* **Early stopping is now off by default.** ``early_stopping`` is
  ``None`` in ``default_fit_parameters`` and in the ``fit_parameters``
  block of ``default_pipeline_parameters``; it was ``5``. A run with
  the default parameters now trains all ``max_epochs`` and keeps the
  epoch with the best validation count Pearson, rather than halting
  after five epochs without an improvement. The learning rate
  schedule is laid out over ``max_epochs`` and the validation metric
  is measured on the EMA weights, so a patience counter over that
  metric was ending runs partway through the cosine decay.
  :meth:`cherimoya.Cherimoya.fit` already defaulted to ``None``; only
  the CLI defaults disagreed. Set ``early_stopping`` to an integer in
  the fit or pipeline JSON to get the old behavior.

Reproducibility
~~~~~~~~~~~~~~~

* Training is now seeded by default. ``random_state`` defaults to ``0``
  in ``default_fit_parameters`` and at the top level of
  ``default_pipeline_parameters``, and :class:`cherimoya.Cherimoya` takes
  a ``random_state`` that seeds its initialization from a local
  ``torch.Generator``. The seed previously reached only the
  peak/negative sampler, so two runs with the same ``random_state`` saw
  the same examples in the same order but started from different
  weights — the larger of the two sources of run-to-run variance was
  unseeded. **Rerunning an unchanged fit JSON now rebuilds the same
  model rather than producing an independent replicate; vary**
  ``random_state`` **to get replicates.**

* ``random_state: null`` no longer means "run unseeded". ``cherimoya
  fit`` draws a seed, prints it whether or not ``verbose`` is set, and
  stores it back into the parameters it deepcopies into the generated
  evaluate JSON. The drawn seed used to be created inside
  :class:`~cherimoya.io.PeakNegativeSampler` and never printed, logged,
  or saved, so a run made with the old default could not be repeated
  even in principle.

* A fit JSON that omits ``random_state`` is now accepted. ``merge_parameters``
  rejects a missing key whose default is ``None`` unless the key is in a
  small whitelist, and ``random_state`` was not in it, so a hand-written
  JSON that left the seed out failed with ``Must provide value for
  'random_state'`` instead of falling back to a default.

* ``cherimoya marginalize`` now uses the ``random_state`` it documents.
  The locus shuffle called ``numpy.random.shuffle`` directly, so the seed
  in ``default_marginalize_parameters`` — and the ``0`` printed for it in
  the CLI reference — had no effect, and every report was built from a
  different sample of background loci.

* The bundled Claude Code agent skill gains a ``random_state`` entry in
  its training vocabulary and a note in the pipeline reference that a
  rerun of an unchanged JSON is not a replicate. Re-run ``cherimoya
  install-skill --force`` to pick them up.

* The reproducibility claims in the README and in the architecture page
  now state the CUDA limit. The fused convolution + normalization kernel
  accumulates its per-example statistics with relaxed atomic adds, so the
  order of that floating-point reduction varies between launches: two
  seeded GPU runs share an initialization and an example order but are
  not bitwise equal, and training compounds the difference rather than
  holding it at rounding scale. CPU runs with the same seed are bitwise
  identical, across separate processes and thread counts.

Logging
~~~~~~~

* The **Training MNLL** and **Training Count MSE** columns of
  ``{name}.log`` are now averaged over the epoch's batches. They
  previously held whatever the last full batch of the epoch produced,
  a single-batch estimate noisy enough that the two training columns
  could look flat or non-monotonic while the model was improving.
  Expect the columns in a new log to sit at a different level, and to
  move far more smoothly, than in one written before this change; the
  validation columns are unchanged. Thanks to Ethan Armand for the
  report in issue #19.

* An epoch in which the loader yields no full batch now writes a row
  with nan in those two columns instead of raising. The training loop
  skips any batch smaller than ``batch_size``, so a dataset smaller
  than one batch used to end the run with a ``NameError`` from the
  logging code rather than a row showing that nothing trained.

Attribution
~~~~~~~~~~~

* The fused dilated convolution + per-example norm inside
  :class:`cherimoya.CheriBlock` now lives on a
  :class:`~cherimoya.cheri.FusedDilatedConvNorm` submodule
  (``block.conv``) rather than being called inline. Attribution methods
  that walk the module tree — DeepLIFT and SHAP in particular — now have
  a concrete node to hook, which is a prerequisite for correct DeepLIFT
  support. The kernel, the conditions under which each forward path
  dispatches, and the numerics are all unchanged. The class is public;
  import it from ``cherimoya.cheri``, since registering a rule for it
  means naming the type.

* Note that the inference megakernel calls the fused op directly rather
  than through ``block.conv``, so hooks on that submodule fire on the CPU
  and Triton training paths but not under ``no_grad`` on CUDA.
  Attribution is unaffected, since it runs with gradients enabled.

* Nothing is registered against the new node by default. An attribution
  method only acts on a module it has been given a rule for, so runs that
  do not mention the class behave exactly as they did before — the node
  exists to be opted into, and adding it changes no existing result.

* **The count head's control-track log stays an inline** ``torch.log``.
  An earlier revision of this work wrapped it in a module for symmetry
  with the convolution, and that module has been removed again: it could
  not affect an attribution. That log's input is the summed control
  tracks, which attribution holds fixed between a sequence and its
  references, so the difference across the node is exactly zero and the
  rescale rule has no multiplier to correct — registering a rule for it
  was measured to leave attributions bitwise identical. It would only
  become a useful hook point for attributions taken with respect to the
  control tracks themselves, which is not a supported path.

Compatibility
~~~~~~~~~~~~~

* **Existing checkpoints are unaffected — bit-for-bit.** The depthwise
  weight is still written to and read from ``state_dict`` under its
  historical ``conv_weight`` key, in its original position, even though
  the parameter now lives at ``block.conv.conv_weight``. Checkpoints
  round-trip between this version and earlier ones in both directions,
  and loading a pre-existing model reproduces identical forward output,
  input gradients, and parameter gradients. Loading accepts either
  spelling, so a state dict assembled by hand or from a
  ``named_parameters()`` walk loads as well.

* The parameter's *name* does change even though its checkpoint key does
  not. ``named_parameters()`` now reports ``blocks.N.conv.conv_weight``
  where it reported ``blocks.N.conv_weight``, and the module tree gains a
  ``blocks.N.conv`` node per block. Code that
  matches parameter names by substring is unaffected — including the
  Muon / AdamW / SGD split in ``cherimoya fit``, whose ``conv_weight``
  exclusion is a substring test and still routes every parameter to the
  optimizer it went to before. What needs the new name is anything that
  looks the parameter up *exactly* by its ``named_parameters()`` name —
  ``dict(model.named_parameters())["blocks.0.conv_weight"]`` now raises
  ``KeyError``, as does any per-parameter map (learning rates, weight
  decay, a hand-rolled EMA) built before the change and keyed by name.
  Attribute-path lookups are fine: ``model.get_parameter`` resolves
  through the alias property, so both spellings return the parameter.

* Block initialization draws from the RNG in the same order as before, so
  a given seed still rebuilds an identical block.

* ``CheriBlock.conv_weight`` remains available as a read-only property
  aliasing ``block.conv.conv_weight``, and reads return the same object.
  Assignment through it now raises instead of silently leaving the
  submodule holding the old parameter — assign to
  ``block.conv.conv_weight`` instead.

Packaging
~~~~~~~~~

* The ``tangermeme`` floor was ``>=0.2.3``, which no release satisfying
  it can actually run: Cherimoya uses ``extract_loci(return_mask=...)``,
  ``io._interleave_loci``, ``predict``'s dtype/device handling,
  ``seqlet.recursive_seqlets``, ``utils.example_to_fasta_coords``,
  ``match.extract_matching_loci`` and ``saturation_mutagenesis``. A
  fresh resolve that picked an old tangermeme failed with
  ``TypeError``/``ImportError`` deep in a subcommand rather than with a
  version error at install time. Raised to ``>=1.4.0``, the version the
  test suite is run against, with a comment in ``pyproject.toml``
  recording the policy so it does not drift again.

* Removed the one-week ``exclude-newer`` window from ``[tool.uv]`` in
  ``pyproject.toml``. It hid every release younger than a week from the
  resolver, so ``uv sync`` could not satisfy a floor on a just-released
  dependency such as ``tangermeme>=1.5.0``. ``uv.lock`` now pins
  tangermeme 1.5.0.

Documentation
~~~~~~~~~~~~~

* The bundled Claude Code agent skill now writes its cross-references as
  complete skill-root-relative paths (``references/cli.md``) rather than
  bare filenames (``cli.md``). A bare name does not say which directory
  the file is in, so an agent following a pointer had to search for the
  target first. All 26 mentions across the nine reference files were
  converted and every target verified to exist. No guidance changed.
  Re-run ``cherimoya install-skill --force`` to pick up the corrections.

* The three forward paths were documented as agreeing to "~1e-5
  max-abs" in the README, on the landing page, and in the architecture,
  benchmarks and ``cherimoya.cheri`` pages. Measured on the default
  9-layer, 128-filter model over a batch of 4 sequences of 2114 bp,
  worst of three seeds, against a profile-logit scale of 0.73:

  .. list-table::
     :header-rows: 1

     * - input dtype
       - CPU vs training kernel
       - CPU vs megakernel
       - training kernel vs megakernel
     * - fp32
       - 2.5e-04
       - 2.1e-04
       - 1.4e-04
     * - fp16 (autocast)
       - 5.9e-04
       - 5.9e-04
       - 4.9e-04
     * - bf16 (autocast)
       - 5.0e-03
       - 5.0e-03
       - 3.9e-03

  So the published figure was optimistic by roughly 20x at fp32 and
  500x at bf16, and the test suite never enforced it — the tolerances
  that exist are 1e-4 for a single block, 5e-3 for gradient parity and
  5e-2 for the whole model. Every page now carries the measured numbers
  and the configuration that produced them.

  ``cheri.py`` also contradicted itself: the module and ``CheriBlock``
  docstrings said ~1e-5 while two comments in the same file said ~1e-2.
  All four now say the same measured thing.

  The earlier figure remains in the v0.2.0 changelog entry below, which
  is a record of what was claimed at the time rather than a current
  statement.

* The receptive field was documented as 1115 bp. Measured, it is
  **1117 bp**: the 21-bp stem reaches 10 bases each side, the dilated
  stack 511, and the 75-bp profile head 37, for a half-width of 558.
  The architecture page's own derivation already summed to 558 while
  stating the ``46`` trimming constant, which is 47 minus one — so
  ``trimming`` is 557 against a half-width of 558, and the outermost
  output position on each side reads one base of zero padding rather
  than having "full context" as the page claimed. The constant is not
  changed: it would change every model's output window.

* ``docs/conf.py`` hard-coded ``release = '0.2.0'`` while
  ``pyproject.toml`` said ``0.2.1``. It now reads the installed
  metadata.

* The README described the Kendall weighting as "one learnable weight
  per output track"; there are two per signal *group*, ``lw0`` for the
  profile term and ``lw1`` for the counts term. It also described
  "minimal weight decay on the Muon-routed projection weights", which
  are in fact the only weights that get any — ``muon_wd`` is 0.03 and
  ``adam_wd`` is 0.

* ``docs/development.rst``'s repository layout omitted
  ``cherimoya/wrappers.py``, ``cherimoya_cli/skills/`` and
  ``cherimoya_cli/commands/install_skill.py``.

Tooling
~~~~~~~

* ``test_evaluate_single_group_value_equals_legacy_full_mean`` failed
  intermittently, on one leg of the CI matrix at a time. It ran
  ``evaluate`` — which predicts in batches through
  ``tangermeme.predict`` — and then recomputed the same thing with a
  single unbatched ``model(X)``, asserting the two agreed to ``1e-4``.
  The paths agree only to within a few float32 ULPs, and
  ``profile_spearman`` ranks with ``argsort().argsort()``: one swapped
  pair moves the metric far more than the float difference that caused
  it, so no tolerance was safe.

  The test now predicts the way ``evaluate`` does, at the same batch
  size, and pins ``compile`` on both sides — ``Cherimoya.load
  (compile=True)`` and ``compile=False`` are not bit-identical even
  under ``TORCH_COMPILE_DISABLE=1``, which was a second source of
  divergence worth about 1.5e-08. With both matched the two sides see
  bit-identical predictions, so every metric agrees by construction
  rather than by luck, and the residual measures 0 across eight seeds.

* The models in ``tests/commands/test_evaluate.py`` are seeded.
  ``_build_and_save`` never passed ``random_state``, so every run drew
  different weights and each assertion in the file passed or failed on
  the draw.

* The CLI tests are now named after the modules they cover, mirroring
  the package layout with the top-level package name elided the way
  ``cherimoya/io.py`` maps to ``tests/test_io.py``:
  ``tests/test_cli_utils.py`` becomes ``tests/test_utils.py``, and
  ``tests/test_evaluate_cli.py``, ``tests/test_fit_wiring.py`` and
  ``tests/test_install_skill.py`` move under a new ``tests/commands/``
  subpackage as ``test_evaluate.py``, ``test_fit.py`` and
  ``test_install_skill.py``. Renames only — no test content changed and
  the suite count is unchanged. ``cherimoya_cli/utils.py`` is a
  top-level module rather than a command, so its test stays at the
  ``tests/`` root. Each file's docstring still records the slice of its
  module it covers, which the filename alone does not: ``test_fit.py``
  exercises parameter wiring and optimizer routing without training,
  and ``test_evaluate.py`` covers the TSV output shape.

* Continuous integration gained two jobs. ``docs`` runs the Sphinx
  build with ``-W``, so a broken ``:doc:`` or ``:ref:`` link fails the
  build rather than shipping; nothing checked the documentation before.
  ``lint`` runs ``ruff`` restricted to syntax errors, undefined names
  and broken comparisons — deliberately narrow, because the full rule
  set reports findings on this tree that are worth fixing separately
  from adding the gate.

* ``docs/development.rst`` now states what CI does not cover: no hosted
  runner has a GPU, so the CUDA and Triton paths — including the
  three-way forward parity that is the repository's central numerical
  invariant — are verified only by running ``pytest -m "cuda or
  triton"`` and the ``compat`` sweep by hand before merging.

* Added tests for ``cherimoya negatives``, which had no coverage at all.
  They pin the flag-to-kwarg forwarding — most flags are renamed on the
  way into ``extract_matching_loci`` — the headerless BED output that
  feeds straight into ``fit`` as a locus file, and the argparse
  defaults, which live only in the parser and so can drift from the CLI
  reference unchecked.

* ``cherimoya attribute`` and ``cherimoya seqlets`` are now tested
  across the seam between them, in
  ``tests/commands/test_attribute_to_seqlets.py``. ``attribute`` saves
  a slice centred on the locus midpoint and ``seqlets`` converts a
  position in that saved array back to a genomic coordinate by assuming
  exactly that; nothing checked the two against each other. Each
  command's own tests pin the arithmetic against literal offsets, so
  editing those offsets to match a changed ``attribute`` leaves them
  green — the 857bp shift is the bug this class produces. The new tests
  run both commands for real and recover the slice offset by locating
  the saved array inside the extracted window, so neither side
  re-derives the other's arithmetic.

* The ``compile`` and ``compile_mode`` keys are now tested as reaching
  ``Cherimoya.load`` from all three subcommands that load a model.
  Previously only ``evaluate`` had a forwarding test, while the
  key-declaration test was parametrized over all three, which read as
  three-way coverage. All three already forwarded the keys correctly.

* Tests that assert only that a removed key is still removed are gone,
  along with the four near-identical pipeline-JSON builders the CLI
  tests each carried; shared fixtures now live in
  ``tests/commands/conftest.py``. The dead-key checks for
  ``marginalize_parameters.output_folder`` and
  ``fit_parameters.count_loss_weight`` are replaced by one invariant
  over every step the pipeline forwards — each declared key must be one
  the receiving subcommand reads — which covers both of those and
  ``annotation_parameters``, which nothing checked. Two assertions that
  could not fail for the reason they were written were strengthened to
  compare values rather than shapes.

* ``docs/development.rst`` listed ``tests/test_fit_wiring.py`` and
  ``tests/test_cli_utils.py``, which were renamed away in v0.2.1, and
  omitted ``tests/commands/`` entirely. The table now matches the
  tree.

* The bundled skill's ``SKILL.md`` frontmatter is now tested, in
  ``tests/commands/test_install_skill.py``. Claude Code reads only the
  frontmatter before deciding to load a skill, so a missing ``name`` or
  ``description``, a ``name`` that differs from the ``cherimoya`` install
  directory, or a description over the 1024-character limit stops the skill
  from loading or triggering, and no other test would notice. The check
  parses the frontmatter directly because PyYAML is not a dependency.

v0.2.1
------

Bug fixes
~~~~~~~~~

* Fixed an ``IndexError`` in ``cherimoya attribute`` when ``extract_loci``
  drops loci near a chromosome end. The ambiguous-base filter is now
  projected back into peak space so the saved index array stays aligned
  with ``X``.

Tooling
~~~~~~~

* Pinned the ``uv`` resolver to a one-week ``exclude-newer`` window via
  ``[tool.uv]`` in ``pyproject.toml`` and updated ``uv.lock`` for
  reproducible dependency resolution.

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
