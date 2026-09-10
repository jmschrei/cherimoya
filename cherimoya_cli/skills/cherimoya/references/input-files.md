# Identifying the user's input files

Before building a pipeline you need to know *what the user actually has*. This
is a recognition guide: given a file (often just an extension), what is it, which
pipeline slot does it fill, and what to confirm. For assay-driven settings
(shifts, strandedness), see `references/assay-defaults.md`.

## The pipeline slots

A training run has, at most, these inputs:

| Slot | Flag (`pipeline-json`) | JSON key | Required? |
|---|---|---|---|
| Reference genome | `-s` | `sequences` | **Yes** |
| Signal | `-i` (repeatable) | `signals` | **Yes** |
| Control | `-c` (repeatable) | `controls` | No |
| Peaks | `-p` (repeatable) | `loci` | No — else MACS3 calls them |
| Negatives | `-neg` (repeatable) | `negatives` | No — else sampled |
| Motif database | `-m` | `motifs` | No — enables annotation + marginalization |
| Exclusion list | *(none; JSON only)* | `exclusion_lists` | No — BED of regions to drop from train/validation |

If a required slot is empty, ask the user for it. For each empty optional slot,
tell the user what the pipeline will do instead. `exclusion_lists` has no flag —
set it directly in the JSON to mask regions (e.g. ENCODE blacklist).

## Recognizing file types

### Reference genome — `sequences`
- **`.fa`, `.fasta`, `.fa.gz`** — the genome the reads were aligned to.
- Must match the **genome build** of the signal and peaks (hg38 ≠ hg19 ≠ mm10).
  The single most important thing to confirm; a mismatch trains a
  silently-wrong model.
- Chromosome names in the FASTA must match `training_chroms` /
  `validation_chroms` in the JSON (e.g. `chr8` vs `8`).

### Signal — `signals`
The measured coverage the model learns to predict. Two forms:
- **Aligned reads / fragments: `.bam`, `.sam`, `.bed`, `.bed.gz`, `.tsv`,
  `.tsv.gz`** — converted to bigWig by `bam2bw` (step 0.2). Fragment files
  (10x-style) need the `-f`/`fragments` flag.
- **Coverage tracks: `.bw`, `.bigwig`** — already processed; used directly,
  conversion skipped.

Confirm with the user:
- **Reads or fragments?** A `.bam` almost always holds **aligned reads**; a
  `.tsv`/`.tsv.gz` (often `.bed`/`.bed.gz`) usually holds **fragments**. Decides
  `-f` and `-pe` — see the reads-vs-fragments table in
  `references/assay-defaults.md`.
- **Stranded or unstranded?** ChIP/ATAC/DNase are usually unstranded (one
  track); many TF/initiation assays are stranded (`+`/`-` pair). Decides `-u`
  and the grouping (below). See `references/assay-defaults.md`.
- **Replicates?** How they combine depends on input type. **BAM/SAM/fragment**
  replicates passed as multiple `-i` are merged into one bigWig by `bam2bw` (one
  pooled track). **Pre-made bigWig** replicates are *not* pooled — a flat list is
  read as N independent unstranded groups (see the grouping footgun below), so
  pool them upstream first. (Peak calling pools all replicates as MACS3
  treatments in both cases.)

> **Check flags against the filetype.** A mismatch is silent:
> - `-f` (fragments) with a `.bam` of aligned reads, or a fragment `.tsv`
>   *without* `-f`, misparses the input.
> - `-pe` only does something for a paired-end read BAM (sets MACS3's `BAMPE`);
>   it is **ignored for fragment files**, so `-f` with `-pe` is a contradiction —
>   the intent was `-f` alone.
> If flags and filetype disagree, stop and ask which is correct.

### Control — `controls`
- Same file types as signal. For ChIP-seq this is the **input / IgG control**
  (unenriched DNA). A model trained *with* controls must be evaluated and used
  *with* the same controls, or the count head sees garbage.

### Peaks — `loci`
- **`.narrowPeak`, `.bed`, `.broadPeak`** — genomic regions of interest
  (positives). If absent, MACS3 calls them from the signal at q < 0.05.
- `summits` in `fit_parameters` centers loci on the narrowPeak summit column.
  Requires a true narrowPeak (10-column) file — don't set `summits` with a
  `.broadPeak` or plain `.bed`, which have no summit column.

### Negatives — `negatives`
- **`.bed`** — background regions the model also trains on. If absent, the
  pipeline samples GC-matched negatives (step 0.3). `negative_ratio` (default
  0.25) sets negatives per peak per epoch.

### Motif database — `motifs`
- **`.meme`** — MEME-format known motifs (e.g. JASPAR, HOCOMOCO). Optional;
  passing it enables tomtom-lite seqlet annotation and marginalization and lets
  the TF-MoDISco report *name* discovered motifs. Without it, TF-MoDISco still
  discovers motifs *de novo* but leaves them unnamed.

## The stranded-signal footgun (important)

`signals` (and `controls`) accept a flat or nested list, and the shape changes
the meaning:

- **Flat list** → each entry is its own **independent unstranded** track:
  `["a.bw", "b.bw"]` = two separate unstranded outputs.
- **Nested list** → the inner list is one **multi-channel group**, e.g. a
  stranded `(+, -)` pair: `[["ctcf.+.bw", "ctcf.-.bw"]]` = one stranded group.

Getting this wrong is silent: a stranded pair written flat as
`["plus.bw", "minus.bw"]` becomes two unstranded tracks and disables the `+/-`
swap during reverse-complement augmentation. Wrap stranded pairs:
`[["plus.bw", "minus.bw"]]`.

When the pipeline converts BAMs itself (step 0.2) it writes the grouped form for
you; you assemble it by hand only for pre-made bigWigs. Full semantics: the
header comment in `cherimoya_cli/defaults.py`.

## Remote files

Any path can be a remote URL (`http://`, `https://`, `s3://`, `gs://`); the
pipeline streams it via `bam2bw` / `tangermeme.io` without downloading, and the
pre-flight check skips it. Credentialed buckets need the usual environment
variables (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`) set first.
