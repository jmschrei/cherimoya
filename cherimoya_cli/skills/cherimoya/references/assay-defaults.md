# Assay-specific settings

The assay determines a handful of preprocessing options that change the biology
of what gets modeled: strandedness, fragments vs reads, paired-end handling, and
read shifts. None can be inferred from a filename, so **confirm the assay and
each setting with the user** rather than assuming. The tables below are starting
points to *propose and confirm*, not values to apply silently.

All map to `pipeline-json` flags and to `preprocessing_parameters` in the JSON.

## The knobs

| Flag | JSON key | Meaning |
|---|---|---|
| `-u` | `unstranded` | Produce one unstranded track instead of a `(+, -)` pair. |
| `-f` | `fragments` | Input is fragment files, not aligned reads. |
| `-pe` | `paired_end` | Input is paired-end; affects MACS3 format (`BAMPE`). |
| `-ps N` | `pos_shift` | Shift `+` strand reads by N bp. |
| `-ns N` | `neg_shift` | Shift `-` strand reads by N bp. |
| `-sf X` | `scale_factor` | Multiply raw counts by X (default 1 = no scaling). |
| — | `callpeaks_gsize` | MACS3 effective genome size: `"hs"` human, `"mm"` mouse. |
| — | `callpeaks_q` | MACS3 q-value cutoff (default 0.05). Loosen to 0.1 for low-yield; tighten to 0.01 for high-confidence. |

Every shift defaults to **0** and strandedness defaults to **stranded**
(`unstranded=false`), so doing nothing gives a stranded, unshifted, read-based
run — *wrong* for ATAC and DNase. Hence confirming the assay matters.

## Reads vs. fragments (`-f` and `-pe`)

For ATAC-seq and DNase-seq especially, the biggest thing to get right is whether
the input holds **aligned reads** or **fragments**; it decides `-f` and `-pe`.
These are *not* interchangeable and depend only on the file, not the assay:

| Input | Typical file | `-f` | `-pe` |
|---|---|---|---|
| **Fragment file** | `.tsv`, `.tsv.gz` (also `.bed`/`.bed.gz`) | **yes** | **no** |
| **Paired-end reads** | `.bam` (paired-end ATAC) | no | **yes** |
| **Single-end reads** | `.bam` (typical DNase) | no | no |

Why (from the pipeline source):
- **`-f` (fragments)** is what routes a fragment file correctly: MACS3 uses
  `FRAG` format and `bam2bw` parses fragment intervals rather than reads.
- **`-pe` (paired-end)** exists *only* to switch MACS3 to `BAMPE` for a
  paired-end **BAM of reads**. A fragment file is already `FRAG`, so **`-pe` has
  no effect on fragments** — don't add it there.

So: a fragment file gets `-f` and *not* `-pe`; a paired-end read BAM gets `-pe`
and *not* `-f`. If unsure which a user has, ask — or infer from the extension and
confirm (see the flag/filetype check in `references/input-files.md`).

## Starting points by assay (confirm before applying)

### ATAC-seq (unstranded, +4 / −4 Tn5 shift)
Unstranded, Tn5 insertion conventionally corrected with a **+4 / −4** shift. The
rest depends on reads vs. fragments:

- **Paired-end read BAM** (`atac.bam`): `-u -ps 4 -ns -4 -pe`
- **Fragment file** (`atac.fragments.tsv.gz`): `-u -ps 4 -ns -4 -f`
  (`-f`, not `-pe` — see the table.)
- **Ask whether the data is already shifted.** Many pipelines (10x/CellRanger/
  ArchR-derived fragments) pre-apply the Tn5 shift; applying it twice is wrong.
  If already shifted, drop `-ps`/`-ns` (keep `-u` and the reads/fragments flag) —
  or, if the upstream shift used the older +4/−5 convention, set the shifts to
  the relative offset rather than zero.

### DNase-seq (unstranded, usually single-end, no shift)
Typically a single-end read BAM modeled as one unstranded track: `-u`
- No `-pe` (single-end), no `-f` (aligned reads, not fragments).
- No shift by default; some protocols apply a small `+1 / 0` cut-site shift
  (`-ps 1 -ns 0`) for footprint work — confirm.
- For a paired-end DNase read BAM, add `-pe`; for a stranded variant, omit `-u`.

### ChIP-seq (TF or histone)
- **TF ChIP-seq is stranded by default** (`+` and `-` coverage modeled as two
  tracks), single-end, no shift, **with an input control**:
  `-i chip.bam -c input.bam`
  Do **not** pass `-u` — the default (`unstranded: false`) is what you want.
- `-i` is the ChIP/IP signal; `-c` the unenriched input control. Confirm the
  user has the matched input — a control-trained model must be used with controls
  thereafter.
- If **paired-end**, add `-pe`. Histone ChIP is *sometimes* modeled unstranded
  (`-u`) — a confirm-with-user choice, not a default.

### Stranded assays (many TF profiling / initiation assays, PRO-seq, CAGE)
- Leave strandedness on (omit `-u`); the pipeline emits a `(+, -)` pair as one
  stranded group.
- For pre-made bigWigs, remember the nested grouping
  (`[["plus.bw","minus.bw"]]`) from `references/input-files.md`.

### Non-human genome
- Set `callpeaks_gsize` to `"mm"` for mouse (or a numeric effective size), **and**
  replace the hg38 `training_chroms` / `validation_chroms` with names/splits
  valid for that genome. The chromosome default is the most commonly missed one
  for non-human data.

## When you're unsure

If the user can't say whether the data is shifted, stranded, or paired-end,
**ask one question** rather than pick a default — an unnecessary read shift or
wrong strandedness degrades the model with no error message. A safe fallback for
a totally-unknown coverage track is unstranded, unshifted (`-u`), which at least
won't double-apply a correction.

Full recipes with worked commands: the ChIP-seq, ATAC-seq, and DNase-seq pages
under <https://cherimoya.readthedocs.io/en/latest/> (recipes section).
