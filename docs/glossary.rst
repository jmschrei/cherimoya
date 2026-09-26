Glossary
========

A quick reference for terms that appear throughout the documentation.
Definitions are scoped to how the word is used inside Cherimoya, not
to the broader literature.


Sequence-to-function (S2F) model
   A neural network that takes a DNA sequence as input and predicts a
   biological readout (e.g. read coverage from a sequencing assay) at
   each base or over the whole sequence. Cherimoya is an S2F model.

One-hot encoded DNA
   A DNA sequence of length ``L`` represented as a ``(4, L)`` tensor
   where row 0 is the indicator for A, row 1 for C, row 2 for G, row 3
   for T, and any ``N`` base is all zeros. ``tangermeme.utils.one_hot_encode``
   produces this from a Python string; ``tangermeme.io.extract_loci``
   produces this from a FASTA file plus a BED of loci.

Profile and counts
   Cherimoya predicts two things per input: a **profile** (per-base
   logits over the output window — what the read coverage looks like
   *shape*-wise) and a **counts** scalar or per-track count (the log
   of the total number of reads expected in the window). The profile
   is normalized into a probability distribution via softmax for the
   MNLL loss; the counts are predicted in log space.

Signal group
   A set of output channels that share an orientation and are treated
   as one biological modality. An unstranded track (e.g. ATAC, DNase)
   is a single-channel group; a stranded BPNet-style ``(+, -)`` pair
   is a two-channel group. Each group emits one count prediction (so
   the two strands of a pair share a single per-locus count) and its
   channels swap among themselves under reverse-complement
   augmentation while groups stay independent of one another. The
   model's ``signal_groups=[1]`` default is a single unstranded
   track; ``[2]`` is a stranded pair; ``[1, 2]`` is an unstranded
   track co-trained with a stranded pair.

Input window vs output window vs trimming
   The model receives a longer sequence (``in_window``, default 2114
   bp) than it predicts over (``out_window``, default 1000 bp). The
   difference, half on each side, is the **trimming** — the
   context the model uses to predict the central window. The default
   trimming of 557 bp is one base short of the receptive field's
   half-width, so the two outermost output positions read a single
   base of zero padding.

Receptive field
   The number of input bases that can influence a single output
   prediction. For Cherimoya's default 9-layer backbone it is 1117 bp:
   the 21-bp input stem (10 bp each side), the dilated stack
   (``1 + 2 + … + 256 = 511``) and the 75-bp profile head (37 bp each
   side) give a half-width of 558.

Saturation mutagenesis (in silico)
   The exhaustive *in-silico* variant scan: for each position in a
   region, evaluate the model on all four single-nucleotide
   substitutions and record the predicted delta. Each base position
   gets a 4-dim vector of "hypothetical importance scores", and
   multiplying by the actual one-hot sequence yields the importance
   of the base that *is* there. Cherimoya's ``attribute`` subcommand
   wraps ``tangermeme.saturation_mutagenesis.saturation_mutagenesis``
   when ``algorithm`` is ``"saturation_mutagenesis"``; its default is
   DeepLIFT/SHAP.

Hypothetical vs actual importance
   *Hypothetical* importance is the score every possible base would
   have at each position (shape ``(4, L)`` per example). *Actual*
   importance is the score the base that is actually there has,
   computed by elementwise multiplying the hypothetical scores with
   the one-hot encoding and summing across the base axis (shape
   ``(L,)`` per example). Both are useful for different downstream
   analyses.

Seqlet
   A contiguous subsequence with high attribution scores. Cherimoya
   uses ``tangermeme.seqlet.recursive_seqlets`` to extract seqlets
   from the actual-importance signal; the resulting BED file is the
   input to TF-MoDISco.

TF-MoDISco
   *Transcription Factor MOtif Discovery from Importance SCOres.* An
   algorithm that clusters seqlets across a dataset to discover
   recurring motif patterns. Cherimoya's pipeline calls TF-MoDISco
   as a downstream step and produces both an HDF5 of patterns and an
   HTML report.

tomtom-lite (``ttl``)
   A fast motif-similarity tool used to label discovered seqlets
   against a known motif database in MEME format (typically JASPAR).
   Distinct from full TomTom; the ``-lite`` variant uses precomputed
   score binning to score thousands of seqlets quickly.

Marginalization
   A causal evaluation of motif effects: insert a known motif from a
   MEME database into the center of negative (non-peak) sequences and
   measure the predicted change in profile and counts. Produces one
   measurement per (motif, locus) pair.

MEME format
   The plain-text motif database format used by the MEME suite. A
   motif is encoded as a position probability matrix. JASPAR releases
   their database in this format.

GC-matched negatives
   Non-peak control regions selected to have the same GC-content
   distribution as the peaks. Cherimoya uses these as negative
   training examples; ``tangermeme.match.extract_matching_loci``
   does the matching. Without GC-matching, the model can learn to
   predict "peak vs background" via GC alone, which is uninformative.

Muon optimizer
   A second-order-flavored optimizer that orthogonalizes its update
   matrix before applying it. Cherimoya uses Muon for the 2D
   projection weights inside Cheri Blocks (the ``linear1`` and
   ``linear2`` weights) and AdamW for everything else.

Kendall-Gal uncertainty weighting
   A technique for combining multi-task losses without a fixed
   hyperparameter: each task gets a learnable scalar that scales its
   loss, with a log-squared regularizer that prevents the scalar from
   going to zero. Cherimoya uses two such scalars (``lw0`` for
   profile loss, ``lw1`` for counts loss) and freezes them once their
   gradients become negligible.

EMA (exponential moving average)
   A shadow copy of every model parameter that is updated as
   ``shadow = decay * shadow + (1 - decay) * parameter`` after each
   optimizer step. Cherimoya uses decay 0.999 by default. The shadow
   weights produce smoother validation curves and are what gets
   saved to the ``.torch`` checkpoint.

Tn5 transposase / +4 / −4 shift
   ATAC-seq uses the Tn5 transposase, which cuts and inserts adapters
   at preferred positions. The cut site lies 4 bp into each end of
   the resulting fragment. Pipelines correct for this by shifting +
   strand alignments by +4 bp and − strand alignments by −5 bp (or
   −4, used by Cherimoya for symmetry). The correction reflects where
   Tn5 *cut*, which is the biologically meaningful event, rather than
   where the read aligns.

bam2bw
   The streaming BAM/SAM/fragment-file → bigWig converter used by
   the Cherimoya pipeline. It reads the input file (local or remote),
   counts reads (or fragments, with optional shifts) into per-base
   coverage bins, and emits one or two bigWig files (stranded or
   unstranded). Cherimoya invokes it automatically when a pipeline
   step receives BAM/SAM/fragment input; it can also be used
   standalone.
