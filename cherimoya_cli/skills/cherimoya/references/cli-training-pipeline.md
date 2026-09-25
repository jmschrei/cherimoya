# Training a model with the CLI pipeline

The end-to-end workflow is two commands: `cherimoya pipeline-json` builds a
config from a few pointers, and `cherimoya pipeline` runs every step from it.
This is the path for "train a Cherimoya model on my data."

First complete the checks in `SKILL.md` (assay, genome build, which inputs
exist). This file assumes that's done.

## Step 1 — build the pipeline JSON

`cherimoya pipeline-json` takes a few CLI pointers and emits a fully-populated
JSON, filling everything you don't specify from `cherimoya_cli/defaults.py`.

```bash
cherimoya pipeline-json \
    -s genome.fa \          # reference genome FASTA (REQUIRED)
    -i signal.bam \         # signal file; repeat -i for replicates (REQUIRED)
    -c control.bam \        # optional control; repeat -c for replicates
    -p peaks.narrowPeak \   # optional BED of peaks (else MACS3 calls them)
    -neg negatives.bed \    # optional BED of negatives (else sampled)
    -m motifs.meme \        # optional MEME motif database
    -n my_run \             # name suffix used in every output filename
    -o my_run.pipeline.json # where to write the JSON
```

Assay-specific flags (`-u` unstranded, `-f` fragments, `-pe` paired-end,
`-ps`/`-ns` read shifts, `-sf` scale factor) are in
`references/assay-defaults.md` — get them right, because they change the
biology, not just the bookkeeping.

**Watch the `-p` flag.** In `pipeline-json`, `-p` means **peaks**; in every
other subcommand it means the **parameters JSON**. Don't carry the meaning
across.

You need `-s` (genome), `-i` (signal), `-n` (name), and `-o` (output JSON);
`-c`/`-p`/`-neg`/`-m` are optional. All four are enforced by argparse, so a
missing one is a clear error rather than a `None_*` filename — but still ask
the user for any that's absent rather than inventing one.

## Step 2 — (optional) edit the JSON

The JSON is both the run config and a permanent record. Edit the file
`pipeline-json` emitted rather than writing one from scratch: keys whose
default is `null` must be *present*, so a from-scratch JSON fails with
`Must provide value for 'x'. Set it to null if this step is supposed to
produce it.` Writing `null` is how you say "MACS3 / the negatives step makes
this later". Only `controls`, `model`, `motifs`, `exclusion_lists`,
`early_stopping` and `loss_weights` may be left out entirely.

Common novice edits — always explain the change you make:

- **Not hg38?** The default `fit_parameters.training_chroms` /
  `fit_parameters.validation_chroms` are hg38 names. Update them to the user's
  genome or validation is silently empty. (For mouse, also set
  `preprocessing_parameters.callpeaks_gsize` to `"mm"`.)
- **Out of GPU memory?** Lower `fit_parameters.batch_size` (64 → 32/16) or
  `fit_parameters.n_filters` (128 → 64), or set `fit_parameters.dtype` to
  `"bfloat16"`.
- **Small dataset?** See dataset-size guidance in
  `references/troubleshooting.md`.
- **Want replicates?** `random_state` defaults to `0`, so rerunning the same
  JSON rebuilds the *same* model rather than an independent replicate. Give
  each replicate its own seed.
- **Skip a step:** most sub-dicts accept `"skip": true`.
- **Dry run:** set top-level `"dry_run": true` to write all per-step JSONs
  without running anything — useful to show the user what will run before
  spending GPU time.

Every key and default is tabulated in `references/cli.md` and, exhaustively, at
<https://cherimoya.readthedocs.io/en/latest/cli.html>.

## Step 3 — run it

```bash
cherimoya pipeline -p my_run.pipeline.json
```

`pipeline` first runs a **pre-flight existence check** on every local input path
and hard-fails with a list of everything missing before any expensive work.
Remote paths (`http://`, `https://`, `s3://`, `gs://`) are skipped and streamed.
If a path it complains about is something the pipeline will create later, set
that key to `null`.

## What the pipeline does, in order

Each step is driven by a sub-dict of the JSON. The fit, attribute, seqlets, and
marginalize steps each write a JSON snapshot (`{name}.fit.json`, etc.) so they
can be re-run in isolation; the preprocessing, annotation, and MoDISco steps run
inline and write no snapshot.

| Step | Runs when | Produces |
|---|---|---|
| 0.1 MACS3 peak calling | `loci` is `null` | `{name}_peaks.narrowPeak` |
| 0.2 `bam2bw` signal → bigWig | signals aren't already bigWig (`.sam/.bam/.bed[.gz]/.tsv[.gz]`) | `{name}.+.bw`/`{name}.-.bw` (stranded) or `{name}.bw` (unstranded); controls → `{name}.control.*.bw` |
| 0.3 negative sampling | `negatives` is `null` | `{name}.negatives.bed` |
| 1 train | always (unless `model` set) | `{name}.torch`, `{name}.final.torch`, `{name}.log`, `{name}.detailed.log`, `{name}.performance.tsv` |
| 2 attribute | always | `{name}.attributions.{ohe,attr}.npz`, `{name}.attributions.idxs.npy` |
| 3.1 seqlets | always | `{name}.seqlets.bed` |
| 3.2 tomtom-lite annotation | `motifs` is set | `{name}.seqlets_annotated.bed`, `{name}.motif_seqlet_count.tsv` |
| 4.1/4.2 TF-MoDISco | always | `{name}_modisco_results.h5`, `{name}_modisco/` |
| 5 marginalization | `motifs` is set | `{name}_marginalize/` |

Worth telling a user up front:

- **Pass an existing `model` path in the JSON and training is skipped** — that
  checkpoint is used for all downstream steps.
- **TF-MoDISco runs even without a motif database** (it discovers motifs *de
  novo*); the database only *names* what it found. The motif-gated steps (tomtom
  annotation, marginalization) are what get skipped without `-m`, so the run
  ends after TF-MoDISco.
- `{name}.torch` is the best-validation checkpoint the downstream steps load;
  see `references/interpreting-outputs.md` for `.torch` vs `.final.torch`.

## Re-running a single step

For the steps that wrote a JSON (fit, attribute, seqlets, marginalize):

```bash
cherimoya fit        -p my_run.fit.json
cherimoya evaluate   -p my_run.evaluate.json
cherimoya attribute  -p my_run.attribute.json
cherimoya seqlets    -p my_run.seqlets.json
cherimoya marginalize -p my_run.marginalize.json
```

Two things that make re-running work:
- After step 0.2 the pipeline rewrites `signals`/`controls` in the fit JSON to
  the produced bigWig paths, so re-running `fit` works off the bigWigs even
  though the input was a BAM.
- `fit` automatically runs `evaluate` on completion and emits
  `my_run.evaluate.json` — there's no separate evaluate step in the pipeline,
  but you can re-run it standalone above.

`negatives` is a standalone subcommand with its own flags (no JSON); see
`references/cli.md`.
