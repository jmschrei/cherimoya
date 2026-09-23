CLI Reference
=============

Reference for every ``cherimoya`` subcommand, every command-line flag,
and every key of every JSON parameter file. Pulled from
``cherimoya_cli.defaults`` and the per-subcommand ``argparse`` setup;
update these tables when the source defaults change.

For a walkthrough of how the pieces fit together see
:doc:`tutorials/cli_pipeline`.


Common conventions
------------------

* Every subcommand except ``pipeline-json`` and ``negatives`` is
  driven by a JSON file passed with ``-p``. Keys missing from the JSON
  fall back to the corresponding default in
  ``cherimoya_cli.defaults``.
* Most JSON schemas accept ``"skip": true`` to no-op the step. The
  ``pipeline`` JSON accepts ``"dry_run": true`` to print/emit the
  per-step JSONs without running any subprocess.
* List-valued keys (``signals``, ``controls``, ``loci``, ``negatives``,
  ``training_chroms``, ``validation_chroms``, ``chroms``) accept
  multiple values. Single-string scalars are coerced to a one-element
  list internally in some places.
* Path-valued keys can be remote URLs (``http://``, ``https://``,
  ``s3://``, ``gs://``). Remote paths are streamed by ``bam2bw`` and
  ``tangermeme.io`` and skipped by the pre-flight existence check
  inside ``cherimoya pipeline``.


cherimoya pipeline-json
-----------------------

Emit a fully-populated pipeline JSON from a small number of CLI
pointers.

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Flag
     - Type
     - Description
   * - ``-s, --sequences``
     - path
     - Reference genome FASTA.
   * - ``-i, --inputs``
     - path (repeatable)
     - Signal file (BAM/SAM/fragment file/bigWig). Repeat for multiple
       replicates.
   * - ``-c, --controls``
     - path (repeatable)
     - Optional control file. Repeat for multiple replicates.
   * - ``-p, --peaks``
     - path (repeatable)
     - Optional BED of peak coordinates. If omitted, MACS3 calls
       peaks.
   * - ``-neg, --negatives``
     - path (repeatable)
     - Optional BED of GC-matched negatives. If omitted, the pipeline
       samples them.
   * - ``-n, --name``
     - str
     - Suffix used in intermediate filenames.
   * - ``-u, --unstranded``
     - flag
     - Treat signal as unstranded (single output track).
   * - ``-f, --fragments``
     - flag
     - Treat input as fragment files, not aligned reads.
   * - ``-ps, --pos_shift``
     - int
     - Shift applied to + strand reads (bp). Default 0.
   * - ``-ns, --neg_shift``
     - int
     - Shift applied to - strand reads (bp). Default 0.
   * - ``-m, --motifs``
     - path
     - MEME-format motif database. When set, TF-MoDISco report,
       tomtom-lite annotation, and marginalization are run.
   * - ``-o, --output``
     - path
     - Output JSON path.
   * - ``-pe, --paired_end``
     - flag
     - Treat input as paired-end. Affects MACS3 file format
       (``BAMPE``) and ``bam2bw`` fragment reconstruction.
   * - ``-sf, --scale_factor``
     - float
     - Multiplier on the raw read counts. Default 1 (no scaling).


cherimoya pipeline
------------------

Run an end-to-end pipeline from a JSON file.

CLI flags:

* ``-p, --parameters`` (required) — path to the pipeline JSON.

JSON schema (top-level keys, with defaults from
``default_pipeline_parameters``):

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``in_window``
     - 2114
     - Input window size (bp).
   * - ``out_window``
     - 1000
     - Output window size (bp).
   * - ``name``
     - ``null``
     - Suffix for intermediate filenames; required.
   * - ``model``
     - ``null``
     - Optional path to an existing ``.torch`` checkpoint. If set,
       skip the training step and use this model for downstream
       stages.
   * - ``dtype``
     - ``"float32"``
     - Tensor dtype for inference; can be ``"bfloat16"`` etc.
   * - ``device``
     - ``"cuda"``
     - Torch device for inference and training.
   * - ``batch_size``
     - 512
     - Batch size for inference stages (attribution, evaluation).
   * - ``verbose``
     - ``true``
     - Print per-step progress.
   * - ``random_state``
     - 0
     - Base RNG seed, inherited by the fit and marginalize steps.
       Seeds the model's initialization and the sampler's draw order.
       Set ``null`` to have one drawn, printed, and recorded instead.
   * - ``exclusion_lists``
     - ``null``
     - BED file(s) of regions to exclude.
   * - ``sequences``
     - ``null``
     - Reference genome FASTA. Required.
   * - ``loci``
     - ``null``
     - BED of peaks. If null, MACS3 calls peaks.
   * - ``negatives``
     - ``null``
     - BED of negatives. If null, GC-matched negatives are sampled.
   * - ``signals``
     - ``null``
     - Signal-track specification (BAM or bigWig files). Required.
       Accepts either a flat list — in which case each entry is its
       own one-channel (unstranded) group — or a structured list whose
       entries are each a ``str`` (one-channel group) or a
       ``list[str]`` (multi-channel group, e.g. a stranded
       ``(+, -)`` pair). Example: ``["atac.bw",
       ["ctcf.+.bw", "ctcf.-.bw"]]`` declares one unstranded ATAC group
       and one stranded CTCF group. See the note in
       ``cherimoya_cli/defaults.py`` for full semantics.
   * - ``controls``
     - ``null``
     - Optional list of control files. Same grouping rule as ``signals``.
   * - ``skip``
     - ``false``
     - If ``true``, the whole pipeline is a no-op.
   * - ``dry_run``
     - ``false``
     - If ``true``, write all per-step JSONs but do not run any
       subprocess.
   * - ``preprocessing_parameters``
     - (sub-dict, below)
     - Settings for MACS3 peak calling and ``bam2bw``.
   * - ``fit_parameters``
     - (sub-dict, below)
     - Training parameters.
   * - ``attribute_parameters``
     - (sub-dict, below)
     - Attribution parameters.
   * - ``seqlet_parameters``
     - (sub-dict, below)
     - Seqlet calling parameters.
   * - ``annotation_parameters``
     - (sub-dict, below)
     - tomtom-lite annotation parameters.
   * - ``modisco_motifs_parameters``
     - (sub-dict, below)
     - TF-MoDISco motif discovery parameters.
   * - ``modisco_report_parameters``
     - (sub-dict, below)
     - TF-MoDISco report parameters.
   * - ``marginalize_parameters``
     - (sub-dict, below)
     - Marginalization parameters.


preprocessing_parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``unstranded``
     - ``false``
     - Produce a single unstranded bigWig instead of a ``+ / -`` pair.
   * - ``fragments``
     - ``false``
     - Treat input as fragment files.
   * - ``paired_end``
     - ``false``
     - Treat input as paired-end; affects MACS3 format.
   * - ``pos_shift``
     - 0
     - + strand shift (bp).
   * - ``neg_shift``
     - 0
     - - strand shift (bp).
   * - ``scale_factor``
     - 1
     - Multiplier on raw counts.
   * - ``read_depth``
     - ``false``
     - Pass ``-r`` to ``bam2bw`` to scale by sequencing depth.
   * - ``callpeaks_format``
     - ``null``
     - MACS3 ``-f`` value. ``null`` auto-detects from the input file
       extension and ``paired_end`` flag.
   * - ``callpeaks_gsize``
     - ``"hs"``
     - MACS3 ``-g`` value (effective genome size). Use ``"mm"`` for
       mouse, a numeric value for other organisms.
   * - ``callpeaks_q``
     - 0.05
     - MACS3 q-value cutoff.
   * - ``verbose``
     - ``true``
     - Print per-step progress.


fit_parameters
~~~~~~~~~~~~~~

These keys are merged with ``default_fit_parameters`` before training.
Unspecified keys fall back to the fit-level defaults.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``n_filters``
     - 128
     - Backbone channel width.
   * - ``n_layers``
     - 9
     - Number of Cheri Blocks.
   * - ``expansion``
     - 2
     - MLP expansion factor inside each Cheri Block.
   * - ``residual_scale``
     - 0.15
     - Fixed residual scalar.
   * - ``batch_size``
     - 64
     - Training batch size.
   * - ``muon_lr``
     - 0.025
     - Muon learning rate.
   * - ``muon_wd``
     - 0.03
     - Muon weight decay.
   * - ``adam_lr``
     - 0.001
     - AdamW learning rate.
   * - ``adam_wd``
     - 0.0
     - AdamW weight decay.
   * - ``lw_lr``
     - 0.001
     - SGD learning rate for the Kendall uncertainty weights
       (``lw0``, ``lw1``).
   * - ``lw_wd``
     - 0.0
     - SGD weight decay for the Kendall uncertainty weights.
   * - ``lw_momentum``
     - 0.9
     - SGD momentum for the Kendall uncertainty weights.
   * - ``n_warmup_epochs``
     - 2
     - Number of epochs over which the LR is linearly warmed up from
       1% of its target before cosine decay begins.
   * - ``negative_ratio``
     - 0.25
     - Negatives per peak per epoch.
   * - ``num_workers``
     - 1
     - Async prefetch workers for the data loader.
   * - ``early_stopping``
     - ``null``
     - Stop after N consecutive epochs with no validation count
       Pearson improvement. ``null`` trains the full ``max_epochs``.
   * - ``max_jitter``
     - 500
     - Maximum jitter (bp) for peak centers at training time.
   * - ``reverse_complement``
     - ``true``
     - Augment training with reverse complements.
   * - ``reverse_complement_average``
     - ``false``
     - Evaluation-time RC averaging.
   * - ``max_epochs``
     - 20
     - Maximum training epochs. Raised at run time when it would buy
       fewer than ``min_total_steps`` optimizer steps.
   * - ``loss_weights``
     - ``null``
     - Fixed ``[w0, w1]`` for the profile and count terms, replacing the
       learned Kendall weights ``lw0`` / ``lw1``. When set, the profile
       loss is divided by each signal group's own batch-mean read depth
       first, and the ``lw_*`` optimizer becomes inert.
       ``[1.333, 0.274]`` reproduces the operating point the learned
       weights reach. ``null`` keeps the Kendall weights. Under
       ``verbose``, the two weights are printed in place of the
       ``lw_*`` optimizer's hyperparameters.
   * - ``min_total_steps``
     - 20000
     - Minimum optimizer steps for the run. An epoch is one pass over the
       peaks, so ``max_epochs`` alone buys a step count proportional to
       how many peaks an experiment has; this raises ``max_epochs`` until
       the run reaches this many steps. The learning rate schedules are
       laid out over the raised value, so they stretch with it. ``null``
       disables the floor.
   * - ``training_chroms``
     - hg38 default (chr2, chr4, chr5, chr7, chr9-22, chrX, chrY)
     - Chromosomes used for training.
   * - ``validation_chroms``
     - ``["chr8", "chr20"]``
     - Held-out chromosomes for validation.
   * - ``in_window`` / ``out_window``
     - 2114 / 1000
     - Input / output window sizes (bp).
   * - ``summits``
     - ``false``
     - Center loci on narrowPeak summit column.
   * - ``dtype``
     - ``"float32"``
     - Training dtype (``"bfloat16"`` enables autocast).
   * - ``device``
     - ``"cuda"``
     - Training device.
   * - ``random_state``
     - 0
     - Base RNG seed. See :ref:`what a seed fixes <reproducibility>`.
       Left ``null`` here, the pipeline's top-level value is used.


attribute_parameters
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Key
     - Default
     - Description
   * - ``batch_size``
     - 512
     - Inference batch size.
   * - ``chroms``
     - training + validation chroms
     - Chromosomes to attribute.
   * - ``output``
     - ``"counts"``
     - Attribute to counts or profile (``"profile"``).
   * - ``ohe_filename``
     - ``"attributions.ohe.npz"``
     - Output: one-hot encoded inputs.
   * - ``attr_filename``
     - ``"attributions.attr.npz"``
     - Output: per-base hypothetical importance.
   * - ``idx_filename``
     - ``"attributions.idx.npy"``
     - Output: boolean mask back to the original loci list.
   * - ``dtype`` / ``device``
     - ``"float32"`` / ``"cuda"``
     - Inference dtype and device.


seqlet_parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``threshold``
     - 0.01
     - Recursive seqlet p-value threshold.
   * - ``min_seqlet_len`` / ``max_seqlet_len``
     - 4 / 25
     - Minimum and maximum seqlet length (bp).
   * - ``additional_flanks``
     - 3
     - Flanking bases retained on each side.
   * - ``ohe_filename`` / ``attr_filename`` / ``idx_filename``
     - inherit from ``attribute_parameters``
     - Inputs from the attribute step.
   * - ``output_filename``
     - ``"seqlets.bed"``
     - Output BED.


annotation_parameters
~~~~~~~~~~~~~~~~~~~~~

tomtom-lite (``ttl``) annotation runs only when ``motifs`` is set on
the top-level JSON.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``motifs``
     - inherit
     - MEME-format motif database.
   * - ``sequences``
     - inherit
     - Reference genome FASTA.
   * - ``seqlet_filename``
     - inherit
     - Seqlet BED from the seqlets step.
   * - ``n_score_bins``
     - 100
     - ``ttl -s``.
   * - ``n_median_bins``
     - 1000
     - ``ttl -m``.
   * - ``n_target_bins``
     - 100
     - ``ttl -a``.
   * - ``n_cache``
     - 250
     - ``ttl -c``.
   * - ``reverse_complement``
     - ``true``
     - Scan motifs in both orientations.
   * - ``n_jobs``
     - -1
     - Parallel workers; -1 uses all cores.
   * - ``output_filename``
     - ``"seqlets_annotated.bed"``
     - Output BED.


modisco_motifs_parameters / modisco_report_parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``n_seqlets``
     - 100000
     - Number of seqlets passed to ``modisco motifs``.
   * - ``output_filename``
     - ``"{name}_modisco_results.h5"``
     - HDF5 output of ``modisco motifs``.
   * - ``output_folder``
     - ``"{name}_modisco/"``
     - Directory output of ``modisco report``.
   * - ``motifs``
     - inherit
     - Motif database passed to ``modisco report -m`` (optional).


marginalize_parameters
~~~~~~~~~~~~~~~~~~~~~~

Skipped entirely when the top-level ``motifs`` is null.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``loci``
     - inherit ``negatives``
     - Background loci to insert motifs into.
   * - ``n_loci``
     - 100
     - Number of background loci per motif.
   * - ``attributions``
     - ``false``
     - Compute attributions on the inserted motif.
   * - ``batch_size``
     - 512
     - Inference batch size.
   * - ``shuffle``
     - ``false``
     - Draw the ``n_loci`` background loci at random from the whole
       file rather than taking the first ``n_loci`` rows. Because a
       sample cannot be drawn without seeing the population, this reads
       every locus in ``loci`` into memory before selecting; the
       unshuffled path stops at ``n_loci`` and does not.
   * - ``random_state``
     - 0
     - RNG seed for the locus shuffle. Left ``null`` here, the
       pipeline's top-level value is used.
   * - ``minimal``
     - ``true``
     - Use the minimal marginalization output format.
   * - ``output_filename``
     - ``"{name}_marginalize/"``
     - Output directory.


cherimoya fit
-------------

CLI flags:

* ``-p, --parameters`` (required) — path to a fit JSON.

JSON schema: the ``fit_parameters`` table above, plus the input
keys ``sequences``, ``loci``, ``negatives``, ``signals``,
``controls``, ``exclusion_lists``, and ``performance_filename``
(default ``"performance.tsv"``). On completion, ``fit`` also writes
the resulting ``evaluate`` JSON and invokes the evaluate step.


cherimoya evaluate
------------------

CLI flags:

* ``-p, --parameters`` (required) — path to an evaluate JSON.

JSON schema:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Description
   * - ``model``
     - ``null``
     - Path to a saved ``.torch`` checkpoint.
   * - ``sequences``
     - ``null``
     - Reference genome FASTA.
   * - ``loci``
     - ``null``
     - BED of evaluation loci.
   * - ``controls``
     - ``null``
     - Optional list of control bigWigs (must match training). Same
       grouping rule as ``signals`` below.
   * - ``signals``
     - ``null``
     - Signal bigWigs to score against (must match training). Accepts
       the same flat-or-grouped form as ``fit``'s ``signals``. The
       per-group count pooling used to compute count metrics is
       recovered from the loaded model's checkpoint, so passing the
       structured form is recommended but not required.
   * - ``chroms``
     - ``["chr8", "chr20"]``
     - Held-out chromosomes.
   * - ``in_window`` / ``out_window``
     - 2114 / 1000
     - Window sizes (must match training).
   * - ``batch_size``
     - 512
     - Inference batch size.
   * - ``reverse_complement_average``
     - ``false``
     - Run predictions on RC inputs and average the results.
   * - ``device`` / ``dtype``
     - ``"cuda"`` / ``"float32"``
     - Inference device and dtype.
   * - ``exclusion_lists``
     - ``null``
     - Optional regions to exclude.
   * - ``performance_filename``
     - ``"performance.tsv"``
     - TSV with one row per signal group.

The TSV columns are
``profile_mnll``, ``profile_jsd``, ``profile_pearson``,
``profile_spearman``, ``count_pearson``, ``count_spearman``,
``count_mse``. The file has one data row per signal group, in
``signal_groups`` order — for a single-group model (the default) this
is a single row holding the same per-group mean that
``calculate_performance_measures`` returns; for a multi-group model
row ``i`` corresponds to ``signal_groups[i]``. Profile metrics are
the mean of the metric over (validation loci × the group's
channels); count metrics are read directly from the per-group
``(n_groups,)`` tensors. See :doc:`multi_task` for an in-depth
description.


cherimoya attribute
-------------------

CLI flags:

* ``-p, --parameters`` (required) — path to an attribute JSON.

JSON schema: the ``attribute_parameters`` table above, plus
``model``, ``sequences``, ``loci``, ``exclusion_lists``, and
``in_window`` / ``out_window``.


cherimoya seqlets
-----------------

CLI flags:

* ``-p, --parameters`` (required) — path to a seqlets JSON.

JSON schema: the ``seqlet_parameters`` table above, plus ``chroms``
and ``loci`` (needed to convert example-relative seqlet coordinates
back to genome coordinates) and ``exclusion_lists``.

The emitted BED is in the same coordinate system as ``loci``. Seqlet
positions are relative to the attribution array, which covers a slice
centred on each locus rather than the full extraction window, so every
seqlet falls inside the slice that ``attribute`` scored.


cherimoya marginalize
---------------------

CLI flags:

* ``-p, --parameters`` (required) — path to a marginalize JSON.

JSON schema: the ``marginalize_parameters`` table above, plus
``sequences``, ``model``, and ``motifs``.


cherimoya negatives
-------------------

Sample GC-matched negative regions for a peak file. All flags are
direct CLI arguments (no JSON):

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Flag
     - Type
     - Description
   * - ``-i, --peaks``
     - path (required)
     - Peak BED.
   * - ``-f, --fasta``
     - path
     - Reference genome FASTA.
   * - ``-b, --bigwig``
     - path
     - Optional signal bigWig (used to set a minimum-counts threshold
       on negatives via ``--beta``).
   * - ``-o, --output``
     - path (required)
     - Output BED.
   * - ``-l, --bin_width``
     - float
     - GC bin width to match. Default 0.02.
   * - ``-n, --max_n_perc``
     - float
     - Maximum fraction of ``N`` bases allowed per locus. Default 0.1.
   * - ``-a, --beta``
     - float
     - Multiplier on the minimum peak counts when filtering negatives
       by signal. Default 0.5.
   * - ``-w, --in_window``
     - int
     - Window over which GC content is calculated. Default 2114.
   * - ``-x, --out_window``
     - int
     - Non-overlapping stride. Default 1000.
   * - ``-v, --verbose``
     - flag
     - Print per-step progress.


cherimoya install-skill
-----------------------

Install the bundled Cherimoya agent skill for `Claude Code
<https://claude.com/claude-code>`_ into your skills directory, creating
``cherimoya/`` inside it. The skill teaches the assistant to drive this CLI
and the Python API — working out which inputs you have, choosing
assay-appropriate settings, calling the right subcommands, and interpreting
outputs — and to ask clarifying questions when an input is ambiguous.

CLI flags:

* ``-d, --directory`` — skills directory to install into. Default
  ``~/.claude/skills``.
* ``--symlink`` — symlink the packaged skill instead of copying it, so
  in-place edits are reflected without reinstalling. Breaks if the install
  location moves.
* ``-f, --force`` — overwrite an existing installation at the destination.

Restart Claude Code (or reload skills) to pick it up.
