Recipe: Differential / Conditional Analysis
===========================================

This recipe covers comparing two (or more) conditions — treated vs.
control, knockout vs. wildtype, time point A vs. B. The standard
pattern in Cherimoya is to train one model per condition, then
compare their predictions on a shared set of loci. This is simpler
to set up than a single multi-output model and gives cleaner
attribution and marginalization results.

If you have only a single condition, use the assay-specific recipe
(:doc:`chipseq_tf`, :doc:`atacseq`, or :doc:`dnaseq`) instead.


Inputs
------

* Reference genome FASTA.
* Per-condition signal BAMs (with replicates pooled or treated as
  separate ``-i`` files).
* Per-condition control BAMs (for ChIP-seq) or none (for ATAC/DNase).
* A motif database in MEME format.


Step 1: train one model per condition
--------------------------------------

Generate one pipeline JSON per condition and run them (see
:doc:`../cli`):

.. code-block:: bash

   cherimoya pipeline-json \
       -s hg38.fa \
       -i condA_rep1.bam -i condA_rep2.bam \
       -c condA_input.bam \
       -m JASPAR_2024.meme -n condA -o condA.pipeline.json

   cherimoya pipeline-json \
       -s hg38.fa \
       -i condB_rep1.bam -i condB_rep2.bam \
       -c condB_input.bam \
       -m JASPAR_2024.meme -n condB -o condB.pipeline.json

   cherimoya pipeline -p condA.pipeline.json
   cherimoya pipeline -p condB.pipeline.json

Each run produces a model checkpoint (``condA.torch`` /
``condB.torch``), per-track bigWigs, attributions, seqlets, and a
TF-MoDISco report scoped to that condition's peaks.

Use the same ``training_chroms`` / ``validation_chroms`` in both
JSONs so the held-out evaluation is comparable.


Step 2: build a shared locus set
--------------------------------

Decide what regions to compare across. The natural choices, in
increasing order of conservatism:

* **Union** of condition-A and condition-B peaks — catches gains and
  losses but includes weakly-supported regions.
* **Intersection** — only regions called as peaks in both
  conditions; conservative but misses pure gains/losses.
* **Union over a reference annotation** (e.g. promoters,
  GENCODE-defined TSSes) — gives a biologically interpretable set
  that doesn't depend on the peak calls.

A typical union with ``bedtools``:

.. code-block:: bash

   cat condA_peaks.narrowPeak condB_peaks.narrowPeak | \
       sort -k1,1 -k2,2n | \
       bedtools merge -i - > shared_loci.bed


Step 3: predict from both models on the shared set
--------------------------------------------------

.. code-block:: python

   from cherimoya import Cherimoya
   from tangermeme.io import extract_loci
   from tangermeme.predict import predict

   model_A = Cherimoya.load("condA.torch", device="cuda")
   model_B = Cherimoya.load("condB.torch", device="cuda")
   model_A.eval(); model_B.eval()

   X, _ = extract_loci(
       sequences="hg38.fa",
       loci="shared_loci.bed",
       chroms=["chr8", "chr20"],     # the held-out chromosomes used during training
       in_window=2114, out_window=1000, max_jitter=0,
       ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
       return_mask=True,
   )

   y_prof_A, y_counts_A = predict(model_A, X, batch_size=64, device="cuda")
   y_prof_B, y_counts_B = predict(model_B, X, batch_size=64, device="cuda")

   # Differences in log counts (predicted log fold change).
   delta_log_counts = y_counts_B - y_counts_A

Loci with large positive ``delta_log_counts`` are predicted to gain
signal in condition B; large negative values are predicted losses.


Step 4: identify differential motifs
------------------------------------

For motif-level differences, run marginalization on both models and
compare. Each model's pipeline already runs marginalization on its
own negative loci; to compare directly, run marginalization on a
**shared** background:

.. code-block:: bash

   cherimoya marginalize -p marginalize_A.json
   cherimoya marginalize -p marginalize_B.json

with the two JSONs differing only in ``model`` and ``output_filename``,
but identical in ``loci`` (the shared background) and ``motifs``. The
delta in per-motif marginalization scores between A and B is a
direct estimate of which motifs cause the condition-specific signal.


Caveats
-------

* **Held-out chromosomes must match.** A model trained with
  ``chr8``/``chr20`` as validation cannot be compared with one
  trained with ``chr1``/``chr12`` as validation on common loci —
  some of those loci were in the second model's training set. Use
  the same split in both JSONs.
* **Replicate-to-replicate noise is the floor.** Before treating
  ``delta_log_counts`` as biological signal, compare to the
  technical-replicate baseline by training two models on independent
  replicates of the same condition and computing the same delta.
  Biological deltas should be larger than the replicate baseline.
* **The two models share no parameters.** Each model is fit
  independently and the comparison happens only at prediction time.
  Multi-output single-model training is possible by passing
  ``-i condA.bw -i condB.bw`` to one pipeline (Cherimoya treats each
  signal as a separate output track), but in practice per-condition
  models give cleaner attribution and motif results.
