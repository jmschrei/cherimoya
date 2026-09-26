Recipe: DNase-seq
=================

This recipe trains a Cherimoya model on DNase-seq, which measures
chromatin accessibility via DNase I hypersensitivity. DNase-seq is
typically single-end and is most commonly modeled as a single
unstranded output track, although stranded variants exist.

If your data is paired-end ATAC-seq instead, see :doc:`atacseq`.


Inputs
------

* Reference genome FASTA (e.g. ``hg38.fa``).
* A BAM file of aligned single-end DNase-seq reads.
* A motif database in MEME format.
* Optional: a BED of peak coordinates. If not provided, MACS3 will
  call peaks.


Generate the pipeline JSON
--------------------------

.. code-block:: bash

   cherimoya pipeline-json \
       -s hg38.fa \
       -i dnase.bam \
       -m JASPAR_2024.meme \
       -n dnase_experiment \
       -o dnase.pipeline.json \
       -u

Flag-by-flag:

* ``-u`` — unstranded output (single signal track). The most common
  setup for DNase-seq.
* No ``-pe`` (single-end reads).
* No ``-f`` (the input is a BAM of aligned reads, not a fragment
  file).
* No shift by default. If your protocol calls for a DNase I-specific
  shift (some pipelines apply +1 / 0 to mark cut sites), set
  ``-ps`` / ``-ns`` here.

For a stranded variant, omit ``-u``. Cherimoya will then produce two
output tracks (+ and - strand) and the trained model will have
``signal_groups=[2]`` (one stranded group).


Run the pipeline
----------------

.. code-block:: bash

   cherimoya pipeline -p dnase.pipeline.json

Steps invoked, in order:

1. MACS3 peak calling on the BAM with no controls (output:
   ``dnase_experiment_peaks.narrowPeak``).
2. ``bam2bw`` converts the BAM to an unstranded bigWig
   (``dnase_experiment.bw``).
3. GC-matched negative sampling (``dnase_experiment.negatives.bed``).
4. Train a 9-layer 128-filter Cherimoya model with
   ``signal_groups=[1]`` and ``n_control_tracks=0``.
5. Compute count attributions via DeepLIFT/SHAP on the
   validation chromosomes.
6. Call seqlets, annotate with tomtom-lite.
7. Run TF-MoDISco motif discovery.
8. Marginalize each motif at the center of negative loci.


Notes
-----

* DNase-seq has historically been processed by many slightly
  different protocols. If you trust the upstream pipeline that
  produced the BAM, you can leave the shift parameters at 0; if you
  produced the BAM yourself and want footprint-resolution
  predictions, a small ``+1 / 0`` shift sometimes improves results.
* Like ATAC-seq, DNase-seq peaks can vary significantly in summit
  height. The same ``max_counts`` / ``max_jitter`` advice from the
  :doc:`atacseq` recipe applies.


Outputs
-------

See the "Outputs" table in :doc:`../tutorials/cli_pipeline`. The
trained model is at ``dnase_experiment.torch`` and the TF-MoDISco
report is at ``dnase_experiment_modisco/``.
