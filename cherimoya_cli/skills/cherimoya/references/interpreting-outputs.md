# Reading what the pipeline produced

Maps every output file to a plain-language answer to "did it work?", "is the
model good?", and "what did it learn?" `{name}` is the `-n` value from step 1.

## "Did it work / is it any good?" — the model and its metrics

### `{name}.performance.tsv` — the scorecard
One row per signal group, seven columns: `profile_mnll`, `profile_jsd`,
`profile_pearson`, `profile_spearman`, `count_pearson`, `count_spearman`,
`count_mse`. Computed on the **held-out validation chromosomes** (default chr8,
chr20).

Headline number: **`count_pearson`** — how well predicted per-peak total signal
correlates with truth on held-out chromosomes.
- Good models on a solid dataset land well above 0.5, often 0.7–0.9+.
- Near 0 means it didn't learn — see "stuck Pearson" in
  `references/troubleshooting.md` (common causes: validation chromosomes
  with no peaks, dropped controls, an uninformative signal).

`profile_pearson` / `profile_jsd` describe how well the profile **shape**
(base-pair resolution) was learned, separate from total counts.

### `{name}.log` — per-epoch training curve
One row per epoch (train/validation metrics). Shows whether validation count
Pearson climbed and where early stopping kicked in. **Compare final results
against the EMA validation numbers here, not the mid-epoch training loss** (see
the checkpoint note below).

### `{name}.detailed.log`
`.log` plus per-group `ProfilePearson_g{i}` / `CountPearson_g{i}` columns —
only relevant for multi-group (multi-task) models.

### `{name}.torch` vs `{name}.final.torch` — which checkpoint to use
Both are **EMA-applied** snapshots (an exponential moving average of the
weights, which validates better than the raw training weights):
- **`{name}.torch`** — the **best validation count Pearson** epoch. Downstream
  steps load this; use it for analysis. Load with `Cherimoya.load`.
- **`{name}.final.torch`** — the EMA snapshot at the **final** epoch; interesting
  only if you want the end-of-training state. Because these are EMA weights, a
  reloaded model reproduces the *EMA* validation row in the log, not any
  mid-epoch loss — expected, not a bug.

## Intermediate data files

| File | What it is |
|---|---|
| `{name}_peaks.narrowPeak` | Peaks MACS3 called (when you didn't supply peaks). |
| `{name}.negatives.bed` | GC-matched background regions (when you didn't supply negatives). |
| `{name}.+.bw` / `{name}.-.bw` / `{name}.bw` | bigWig coverage `bam2bw` made (stranded pair / unstranded). |
| `{name}.control.*.bw` | Same, for controls. |

## "What did the model learn?" — interpretation outputs

### Attributions — `{name}.attributions.*`
Per-base importance from saturation mutagenesis, over the central 400 bp of each
locus, as three aligned files:
- `{name}.attributions.ohe.npz` — one-hot input sequences (that 400 bp span).
- `{name}.attributions.attr.npz` — hypothetical importance scores.
- `{name}.attributions.idxs.npy` — boolean mask back to the original loci list
  (which loci survived N-filtering).

Don't eyeball these as files; load and plot them (see
`references/using-tangermeme.md`) or let the seqlet/MoDISco steps
consume them.

### Seqlets — `{name}.seqlets.bed`
Short, high-importance stretches pulled from the attributions, in genome
coordinates — the candidate functional elements the model found.

### Annotated seqlets — `{name}.seqlets_annotated.bed` + `{name}.motif_seqlet_count.tsv`
Only with a motif database. tomtom-lite labels each seqlet with its closest
known motif; the `.tsv` tallies matches per motif — a quick "which TFs did the
model rely on" summary.

### TF-MoDISco — `{name}_modisco_results.h5` + `{name}_modisco/`
*De novo* motifs aggregated across all seqlets. The `.h5` holds the patterns;
`{name}_modisco/` is a browsable **HTML report** (open `index.html`) — usually
the most satisfying thing to show a user, with names when a motif database was
supplied.

### Marginalization — `{name}_marginalize/`
Only with a motif database. Inserts each known motif into background sequences
and measures the model's predicted response — an HTML-plus-CSV report of how
much the model "cares" about each motif.

## Per-step JSON snapshots

`{name}.{pipeline,fit,evaluate,attribute,seqlets,marginalize}.json` record the
exact parameters each step ran with — the record of what happened, and they let
you re-run any single step (`cherimoya <step> -p {name}.<step>.json`).
