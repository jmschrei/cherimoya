Recipe: TF ChIP-seq
===================

This recipe trains a stranded Cherimoya model on transcription factor
ChIP-seq with input controls. ChIP-seq data is typically single-end
and stranded, and the standard practice is to model the + and -
strand coverage as two separate output tracks.


Inputs
------

* Reference genome FASTA (e.g. ``hg38.fa``).
* One or more BAM files of aligned ChIP-seq reads.
* One or more BAM files of aligned input controls.
* A motif database in MEME format (e.g. ``JASPAR_2024.meme``).
* Optional: a BED of peak coordinates. If not provided, MACS3 will
  call peaks.


Generate the pipeline JSON
--------------------------

.. code-block:: bash

   cherimoya pipeline-json \
       -s hg38.fa \
       -i ctcf_rep1.bam -i ctcf_rep2.bam \
       -c input_rep1.bam -c input_rep2.bam \
       -m JASPAR_2024.meme \
       -n ctcf \
       -o ctcf.pipeline.json

If you already have peak coordinates, add ``-p ctcf_peaks.narrowPeak``.

The defaults populated into the JSON are:

* Stranded output (``unstranded: false``).
* No read shift (``pos_shift: 0``, ``neg_shift: 0``).
* Single-end (``paired_end: false``, ``fragments: false``).
* MACS3 q-value 0.05 with auto-detected file format (``BAM`` for BAM
  inputs, ``BAMPE`` when ``paired_end: true``).

These are appropriate for typical TF ChIP-seq. If your ChIP-seq is
paired-end, override with ``-pe`` at this step or set
``"paired_end": true`` in the JSON.


Run the pipeline
----------------

.. code-block:: bash

   cherimoya pipeline -p ctcf.pipeline.json

Steps invoked, in order:

1. MACS3 peak calling on the ChIP BAMs with the input BAMs as
   controls (output: ``ctcf_peaks.narrowPeak``).
2. ``bam2bw`` converts the ChIP and input BAMs to stranded bigWigs
   (``ctcf.+.bw``, ``ctcf.-.bw``, ``ctcf.control.+.bw``,
   ``ctcf.control.-.bw``).
3. GC-matched negative sampling (``ctcf.negatives.bed``).
4. Train a 9-layer 128-filter Cherimoya model with
   ``signal_groups=[2]`` (one stranded ``(+, -)`` group) and
   ``n_control_tracks=2``.
5. Compute count attributions via DeepLIFT/SHAP on the
   validation chromosomes.
6. Call seqlets, annotate with tomtom-lite against
   ``JASPAR_2024.meme``.
7. Run TF-MoDISco motif discovery and generate the HTML report.
8. Marginalize each motif at the center of negative loci and
   generate the marginalization report.


Common overrides
----------------

If you want to deviate from defaults, edit the JSON before running
``cherimoya pipeline``. The most commonly overridden keys:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key path
     - Default
     - When to change
   * - ``fit_parameters.n_filters``
     - 128
     - Smaller (64) for faster experiments; larger (192) for very
       complex assays.
   * - ``fit_parameters.n_layers``
     - 9
     - Reduce to shrink receptive field (``RF = 46 + sum(2^i)``);
       9 layers → 1115 bp.
   * - ``fit_parameters.max_epochs``
     - 50
     - Reduce for quick smoke tests; increase only if the validation
       count Pearson is still climbing at epoch 50.
   * - ``fit_parameters.training_chroms`` / ``validation_chroms``
     - hg38 default split (chr8/chr20 validation)
     - For non-hg38 references, replace with the appropriate
       chromosome list.
   * - ``preprocessing_parameters.callpeaks_gsize``
     - ``"hs"``
     - Set to ``"mm"`` for mouse, or a numeric effective genome size
       for other organisms.
   * - ``preprocessing_parameters.callpeaks_q``
     - 0.05
     - Loosen (0.1) for low-yield experiments, tighten (0.01) for
       very confident calls.


Outputs
-------

See the "Outputs" table in :doc:`../tutorials/cli_pipeline` for the
full list. The two most useful artifacts for downstream analysis are:

* ``ctcf.torch`` — the trained model, loadable with
  :meth:`cherimoya.Cherimoya.load`.
* ``ctcf_modisco/`` — the TF-MoDISco HTML report showing discovered
  motifs and their seqlet support.
