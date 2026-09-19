# Concepts — the vocabulary in one place

A plain-language primer for a user new to sequence modeling. Load this when a
term below is blocking understanding; for depth, the docs glossary at
<https://cherimoya.readthedocs.io> is authoritative.

## What the model does

Cherimoya is a **sequence-to-function** model: given a stretch of DNA (one-hot
encoded, `(batch, 4, length)`), it predicts a genomic assay's coverage over that
stretch. Training teaches it which sequence patterns drive the measured signal.

## Profile vs. counts (the two output heads)

Every forward returns `(profile, log-count)` — two complementary views of the
same locus:
- **Profile** — the *shape* of the signal at base-pair resolution across the
  output window (default 1000 bp): where within the region the reads pile up.
- **Counts** — the *total* signal in the window (one number per group), trained
  against `log(count + 1)`. This is the headline "how much signal is here"
  prediction; `count_pearson` on held-out data is the usual quality metric.

A model can be good at one and not the other — hence separate metrics.

## Peaks, negatives, controls

- **Peaks (`loci`)** — genomic regions of interest, the **positive** examples
  (where the assay shows enrichment). Supplied as a BED, or called by MACS3 from
  the signal.
- **Negatives** — **background** regions the model also trains on so it learns
  what *absence* of signal looks like. GC-matched to the peaks; `negative_ratio`
  (default 0.25) sets how many per peak per epoch.
- **Controls** — an unenriched track (ChIP input / IgG) that captures
  background/bias. A model trained *with* controls must be used *with* them
  thereafter, or the count head sees garbage.

## Strandedness

Some assays distinguish the `+` and `-` DNA strands (a `(+, -)` pair of tracks);
others don't (one unstranded track). This is a per-assay property, not
inferable from a filename — see `references/assay-defaults.md`. Getting it
wrong is silent; see the grouping footgun in `references/input-files.md`.

## Attribution, seqlets, motifs

The "what did the model learn" chain, in order:
- **Attribution** — per-base importance scores: which bases the model relied on
  for its prediction. Cherimoya computes these by **saturation mutagenesis
  (ISM)** — mutating each base and measuring the change in prediction.
- **Seqlets** — short, contiguous high-importance stretches pulled out of the
  attributions (lengths ~4–25 bp): candidate functional elements.
- **Motifs** — recurring sequence patterns (e.g. a transcription-factor binding
  site). **TF-MoDISco** clusters seqlets into *de novo* motifs; a supplied MEME
  motif database lets those discovered motifs be *named* against known ones.
- **Marginalization** — a complementary check: insert a known motif into
  background sequences and measure how much the model's prediction responds — how
  much it "cares" about that motif.

## Training vocabulary

- **Held-out (validation) chromosomes** — whole chromosomes kept out of training
  (default hg38 split: chr8, chr20) so metrics measure generalization to unseen
  sequence, not memorization. Must match your genome build.
- **EMA (exponential moving average)** — a smoothed running average of the model
  weights kept during training; it validates better than the raw weights.
  Cherimoya **saves the EMA weights**, so a reloaded model reproduces the EMA
  validation numbers in the log, not the mid-epoch training loss (see
  `references/interpreting-outputs.md`).
- **Reverse-complement augmentation** — training also shows the model each
  sequence's reverse complement (on by default); for stranded data this swaps the
  `+`/`-` tracks, which is why stranded pairs must be grouped correctly.
- **Early stopping** — training halts after a set number of epochs (default 5)
  with no improvement in validation count Pearson, keeping the best checkpoint.
- **Random state (seed)** — fixes the model's initial weights and the order the
  sampler draws examples in, so a run can be repeated. Defaults to `0`; set it
  to `null` and one is drawn, printed, and recorded in the evaluate JSON. Two
  runs with the same seed are bitwise identical on CPU. On GPU they share an
  initialization and an example order but diverge as training compounds the
  last-bit differences from the fused kernel's atomic reductions — a seed
  repeats the setup of a run, it does not make GPU training deterministic.

For every default value, see `references/cli.md` or `cherimoya_cli/defaults.py`.
