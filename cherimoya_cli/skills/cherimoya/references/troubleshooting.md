# Troubleshooting a Cherimoya run

Common failures, the symptom each produces, and what to change. Match the
symptom here first. Authoritative version:
<https://cherimoya.readthedocs.io/en/latest/troubleshooting.html>.

## Prerequisites (check first)

- **GPU.** Training and high-throughput inference want a CUDA GPU. A pure-PyTorch
  CPU fallback exists for everything except the inference megakernel, so the
  package runs on CPU, but **training on CPU is impractical at any realistic
  scale** — warn the user before starting. CPU is fine for interactive use.
- **Triton** is a hard dependency and installs from PyPI automatically on
  standard Linux + CUDA. Unusual toolchains may need a manual Triton install.

## "CUDA out of memory" at the start of training

Cheapest fixes first:
1. Lower `fit_parameters.batch_size` (64 → 32 → 16 → 8).
2. Set `fit_parameters.dtype` to `"bfloat16"` (bf16 autocast).
3. Shrink the model: `fit_parameters.n_filters` (128 → 96).

The default (batch 64, 2114 bp window, 9-layer/128-filter model) fits
comfortably on a 16 GB GPU. Reducing batch size is cheapest; don't change model
complexity without user input.

## "CUDA out of memory" in the attribute step

Under DeepLIFT/SHAP (the default `algorithm`), `attribute_parameters.batch_size`
counts sequence-reference pairs, each run forward and backward. On the default
model at 2114 bp, 64 pairs peaked at 15.5 GB and 512 at 124 GB. Lower
`batch_size` (64 → 32 → 16); with a fixed `random_state` the attributions are
the same at any batch size, up to float rounding.

## "Convergence deltas too high" during attribution

DeepLIFT/SHAP scores for an example and a reference should sum to the difference
in their predictions; the warning fires when a pair misses by more than
`warning_threshold` (default 0.001, absolute, in the units of the attributed
output). The known cause is Python code calling tangermeme's `deep_lift_shap`
without `additional_nonlinear_ops=attribution_ops()` (from
`cherimoya.deep_lift_shap`). `cherimoya attribute` always registers them.

## "The first iteration is very slow, then it speeds up"

Not a bug — **Triton autotune** sweeping kernel configs on the first call, then
caching the winner for the process. The first inference call autotunes
separately. Nothing to fix.

## "Training loss is NaN"

By likelihood:
1. **A peak with zero counts** makes the multinomial log-likelihood `-inf`. The
   `min_counts`/`max_counts` filters are `PeakGenerator` arguments **not exposed
   through the CLI JSON**, so the CLI fix is to **remove empty/zero-count peaks
   from the peak BED upstream** before training. (`min_counts` is only reachable
   by writing a custom Python loop with `cherimoya.io.PeakGenerator`.)
2. **Mismatched strand counts** — stranded data passed flat (or unstranded data
   as a pair) gives `y` the wrong shape. Set `verbose=true` and check the
   train/validation shapes printed at startup. Confirm a stranded `(+, -)` pair
   is nested (`[["plus.bw","minus.bw"]]`), not flat (see
   `references/input-files.md`).

## "Stranded predictions come almost entirely from one strand"

A stranded `(+, -)` model (TF ChIP, PRO-cap, etc.) whose reconstructed
`ExpectedCountsWrapper` profile has nearly all its signal on one strand,
even though the observed data has comparable coverage on both (offset by
~100-300 bp). Fixed in the **Unreleased** release: the profile loss now
normalizes each signal group's channels **jointly** (one multinomial over
both strands + length) instead of per-strand, so the strand balance is
trained. Models trained with an older release have an uncalibrated
strand offset baked into their weights — **retrain** with a current
release to fix. Unstranded ATAC/DNase models are unaffected (the change
is bit-identical for single-channel groups). See the CHANGELOG entry
"Loss (breaking for stranded/multi-channel models)".

## "Validation Pearson is stuck near zero"

1. **Validation chromosomes contain no peaks.** Check `Validation Set Size` at
   startup — if ~0, your `validation_chroms` don't intersect your peaks. Common
   when peaks are subset to one chromosome, or when the genome isn't hg38 and the
   default chr8/chr20 split is wrong.
2. **Controls were silently dropped.** A control-trained model evaluated without
   controls collapses. The evaluate step must list the same `controls` as fit.
3. **Train and validation signals differ** (e.g. different replicates) — a much
   harder generalization problem than across-chromosome.
4. **The signal is genuinely uninformative.** Sanity-check on a known-good target
   (e.g. CTCF in K562 from ENCODE); if that converges, the issue is upstream of
   Cherimoya.

## "Is my dataset even big enough?"

- **Peaks:** <2,000 → expect underfitting; 2,000–10,000 → workable, reduce
  `n_filters` to 64-96; 10,000+ → defaults fine; 50,000+ → larger models
  (`n_filters=192`/`256`) worth trying.
- **Depth:** <~50 reads/peak → profile metrics noise-dominated (pool replicates
  before peak calling); hundreds → normal; thousands (ATAC/DNase) →
  signal-limited, not data-limited.

## "MACS3 hangs or returns no peaks"

- A large BAM can take ~10 min in MACS3 — normal.
- Empty peak file on a small BAM → `callpeaks_q` too strict; try 0.1 or 0.5.
- Wrong auto-detected format → set `preprocessing_parameters.callpeaks_format`
  explicitly (e.g. `BAMPE` for paired-end).

## "bam2bw couldn't open a remote URL"

Streaming needs the remote store to support range requests. Public ENCODE HTTPS
BAMs, S3-presigned URLs, and standard GCS objects work. Credentialed buckets
need `AWS_*` / `GOOGLE_APPLICATION_CREDENTIALS` set first.

## "torch.compile / CUDA-graph error at inference time"

A `torch._dynamo`/`torch._inductor` traceback from inside `Cherimoya.forward`, or
"accessing tensor output of CUDAGraphs that has been overwritten". The forward is
wrapped in `torch.compile(mode='max-autotune')` by default. **If you don't
immediately recognize the error, load with `compile=False`** — numerically
identical, at the cost of the compile speedup:

```python
model = Cherimoya.load("checkpoint.torch", device="cuda", compile=False)
# or keep autotuned kernels, skip only the CUDA graph:
model = Cherimoya.load("checkpoint.torch", device="cuda",
                       compile_mode='max-autotune-no-cudagraphs')
```

For fastest inference, call `model.eval()` before predicting so the megakernel
reuses its bf16 weight cast.

## "Cherimoya.load rejects a checkpoint" (`KeyError: 'config'` or a `weights_only` `RuntimeError`)

The checkpoint was saved with the legacy `torch.save(model, ...)` path from
before v0.1.0. It's not loadable by the current config-plus-state-dict loader;
retrain with the current release. See `references/using-in-python.md` for
the save/load format.

## "The loaded model predicts differently than training reported"

Not a mismatch. The saved checkpoint holds the **EMA-applied** weights, which
produced the best validation numbers. Compare against the **EMA validation** row
in `{name}.log`, not the mid-epoch training loss. See the checkpoint note in
`references/interpreting-outputs.md`.
