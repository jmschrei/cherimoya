# CLI subcommand map

`cherimoya` has one subcommand per pipeline stage. Most people only need
`pipeline-json` + `pipeline` (see `references/cli-training-pipeline.md`);
the rest re-run a single stage. This maps a goal to a subcommand. For
exhaustive flag and JSON-key tables, defer to
<https://cherimoya.readthedocs.io/en/latest/cli.html>.

## Goal → subcommand

| Goal | Subcommand |
|---|---|
| Build a pipeline config from raw-data pointers | `pipeline-json` |
| Run the whole thing end-to-end | `pipeline` |
| Just train (and auto-evaluate) a model | `fit` |
| Score an existing model on held-out data | `evaluate` |
| Compute per-base attributions | `attribute` |
| Call seqlets from attributions | `seqlets` |
| Measure a model's response to inserted motifs | `marginalize` |
| Sample GC-matched negative regions | `negatives` |

## Two flag conventions

- **`pipeline-json` and `negatives` take direct CLI flags** (no JSON).
- **Every other subcommand is driven by a JSON file passed with `-p,
  --parameters`.** Missing keys fall back to `cherimoya_cli/defaults.py`.
- **Short flags are overloaded:** `-p` means `--peaks` in `pipeline-json` but
  `--parameters` elsewhere, and `-i` means `--inputs` (signal) in `pipeline-json`
  but `--peaks` in `negatives`. Don't carry a flag's meaning across subcommands.

Most JSON schemas accept `"skip": true` to no-op a step; the pipeline JSON also
accepts `"dry_run": true` to emit per-step JSONs without running anything.

## `pipeline-json` flags

`-s` sequences (FASTA) · `-i` inputs/signal (repeatable) · `-c` controls
(repeatable) · `-p` peaks (repeatable) · `-neg` negatives (repeatable) ·
`-n` name · `-o` output JSON · `-m` motifs (MEME) · `-u` unstranded ·
`-f` fragments · `-pe` paired-end · `-ps` pos_shift · `-ns` neg_shift ·
`-sf` scale_factor. `-s`, `-i`, `-n`, `-o` are all effectively required
(argparse enforces none, but a missing one fails later or yields `None_*`
filenames).

## `negatives` flags (standalone)

`-i` peaks (required) · `-o` output BED (required) · `-f` fasta ·
`-b` bigwig (sets a min-counts threshold via `--beta`) · `-l` bin_width
(0.02) · `-n` max_n_perc (0.1) · `-a` beta (0.5) · `-w` in_window (2114) ·
`-x` out_window (1000) · `-v` verbose.

## Key defaults (from `defaults.py`)

Quote these; don't guess others — read `defaults.py` or the docs.

- Windows: `in_window` 2114, `out_window` 1000.
- Model: `n_filters` 128, `n_layers` 9, `expansion` 2.
- Training: `batch_size` 64, `max_epochs` 20, `early_stopping` 5 (epochs of no
  validation count-Pearson improvement), `n_warmup_epochs` 2,
  `negative_ratio` 0.25, `reverse_complement` true, `num_workers` 1.
- Inference stages: `batch_size` 512.
- Device/dtype: `cuda` / `float32`.
- Split (hg38): `validation_chroms` = chr8, chr20; everything else (minus
  chr1/chr3/chr6 held out of the default list) trains. **Change these for
  non-hg38 genomes.**
- Peak calling: `callpeaks_gsize` `"hs"`, `callpeaks_q` 0.05.
- Seqlets: `threshold` 0.01, lengths 4–25 bp, `additional_flanks` 3.
