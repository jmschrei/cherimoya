# Using Cherimoya from Python

For programmatic use rather than the CLI. Most end-users don't need this, but
"where is my trained model and how do I load it later" is a common follow-up.

## The public API

Only these symbols are public (`cherimoya/__init__.py`); everything else is
private and may change between versions:

- `Cherimoya` — the model.
- `CheriBlock` — the building block.
- `EMA` — the exponential-moving-average wrapper used during training.
- `ControlWrapper`, `ProfileWrapper`, `LogCountWrapper`,
  `ExpectedCountsWrapper` — output wrappers for analysis (see
  `references/using-tangermeme.md`).

Some module-level symbols are also public though not re-exported (the full
list is in the docs' development page):
- `attribution_ops`, imported from `cherimoya.deep_lift_shap`, which every
  DeepLIFT/SHAP call on a Cherimoya model passes as `additional_nonlinear_ops`.
- `cherimoya.training.fit` and `cherimoya.training.CherimoyaModule` — training
  (see "Training in Python" below).
- `cherimoya.io.PeakGenerator`, `PeakNegativeSampler` and
  `ShardedEpochSampler` — the training data.

## Constructing and calling a model

```python
import torch
from cherimoya import Cherimoya

model = Cherimoya(n_filters=128, n_layers=9).cuda()

# X: one-hot DNA, shape (batch, 4, length). Length is arbitrary; the CLI and
# quickstart use 2114, which yields the 1000 bp output window below.
X = torch.randn(2, 4, 2114, device="cuda")
with torch.no_grad():
    y_profile, y_counts = model(X)

y_profile.shape   # (2, 1, 1000)  — (batch, sum of group channels, out_window)
y_counts.shape    # (2, 1)        — (batch, n_groups)
```

The forward **always returns a `(profile, log-count)` tuple.** The count head is
trained against `log(count + 1)`.

### Signal groups (single-task vs stranded vs multi-task)
`signal_groups` describes the output tracks (mirrors the CLI `signals`
grouping):
- `signal_groups=[1]` (default) — one unstranded track (ATAC, DNase, …).
- `signal_groups=[2]` — one stranded `(+, -)` group; two profile channels
  sharing one count prediction.
- `signal_groups=[1, 2]` — multi-task: unstranded plus stranded; `y_counts` has
  shape `(batch, 2)`.

### Control tracks
Models trained with controls expect a control tensor every forward:
`model(X, X_ctl)`. For analysis tools that pass only a sequence, wrap in
`ControlWrapper` (see `references/using-tangermeme.md`) so a zero control is
synthesized.

## Saving and loading — the checkpoint format

Checkpoints are a **config + `state_dict` bundle, not a pickled module.** This
survives source-layout changes and is safe to load with `weights_only=True`.

```python
model.save("my_model.torch")

model = Cherimoya.load("my_model.torch")                 # CPU by default
model = Cherimoya.load("my_model.torch", device="cuda")  # onto a GPU
```

For fastest inference call `model.eval()` first.

### Run inference under `torch.no_grad()` (megakernel)

Cherimoya has a fused **inference megakernel** roughly **2× faster** than the
default (training) forward, but it dispatches only when **the input is on a CUDA
device and gradients are disabled** (GPU-only — CPU inputs always take the
fallback). Wrap prediction in `torch.no_grad()` (or `torch.inference_mode()`);
`model.eval()` alone does *not* disable grad, so without the wrapper the slower
default kernel runs:

```python
model = model.eval()
with torch.no_grad():                # required for the fast megakernel
    y_profile, y_counts = model(X)
```

(The megakernel also requires `expansion * n_filters` to be a multiple of 16,
which holds for the defaults: `2 * 128 = 256`.)

### Choosing the `compile` setting — ask the user

`Cherimoya.load` compiles the forward via two arguments: `compile` (bool,
default `True`) and `compile_mode` (str, default `'max-autotune'`). This is a
speed-vs-robustness trade-off — **ask the user before loading** rather than
assuming the default:

| Setting | How to load | Speed | Robustness |
|---|---|---|---|
| `max-autotune` (default) | `Cherimoya.load(path, device="cuda")` | Fastest | Can hit `torch.compile` / CUDA-graph errors on some setups |
| `max-autotune-no-cudagraphs` | `Cherimoya.load(path, device="cuda", compile_mode='max-autotune-no-cudagraphs')` | Middle — keeps autotuned kernels, drops CUDA-graph capture | Avoids the CUDA-graph errors |
| no compile | `Cherimoya.load(path, device="cuda", compile=False)` | Slowest (no compile speedup) | Most robust — sidesteps all compile/CUDA-graph foot-guns |

All three are **numerically equivalent** and still run Cherimoya's Triton
inference kernels; only the `torch.compile` wrapping differs. With
`compile=False` the `compile_mode` value is ignored.

Rules of thumb:
- **Interactive / exploratory work, or after compile errors:** `compile=False`
  is safest — the lost speedup (~10–20% on the megakernel) rarely matters at that
  scale and it rules out an unrecognized `torch.compile`/CUDA-graph traceback
  (see `references/troubleshooting.md`).
- **Large-scale / high-throughput:** the default `max-autotune` is worth it; fall
  back to `max-autotune-no-cudagraphs` on a CUDA-graph error, and to
  `compile=False` only if that still fails.

The same two settings are JSON keys on `evaluate`, `attribute` and
`marginalize`, and at the top level of a `pipeline` JSON where one value reaches
every step that loads a model: `{"compile": false}`,
`{"compile_mode": "max-autotune-no-cudagraphs"}`. Reach for them when a CLI run
hits a compile or CUDA-graph traceback. `attribute` is the exception: it
defaults to `compile: false` (neither algorithm ran faster compiled) and never
compiles under DeepLIFT/SHAP.

### Things to tell users about checkpoints

- **Saved weights are the EMA shadow.** `cherimoya.training.fit(...)` applies
  the EMA average before saving, so a loaded model reproduces the **EMA
  validation** numbers in the log — not any mid-epoch training loss. Expected,
  not drift.
- From a pipeline run, load **`{name}.torch`** (best validation count Pearson);
  `{name}.final.torch` is the final-epoch EMA snapshot. See
  `references/interpreting-outputs.md`.
- **Legacy `torch.save(model, ...)` checkpoints (pre-0.1.0) are not loadable**
  and must be retrained. Cherimoya is under active development and may break
  checkpoint compatibility between versions — pin the version you train with if
  you need to reload later.

## Training in Python

For most "train on my data" requests the CLI pipeline
(`references/cli-training-pipeline.md`) is the better tool. When the user needs
Python, `cherimoya.training.fit` trains with PyTorch Lightning and writes the
same files as `cherimoya fit` (`{name}.torch`, `{name}.final.torch`,
`{name}.log`, `{name}.detailed.log`, next to `model.name`):

```python
from cherimoya import Cherimoya
from cherimoya.io import PeakGenerator
from cherimoya.training import fit

# The dataset inside the DataLoader, not the DataLoader: fit builds its own.
training_data = PeakGenerator(peaks=..., negatives=..., sequences=...,
    signals=..., random_state=0).dataset

# The CLI's schedule: 2 warmup epochs, cosine decay over the rest.
steps_per_epoch = -(-len(training_data) // 64)

model = Cherimoya(n_filters=128, n_layers=9, name="my_run", random_state=0)
trainer = fit(model, training_data, X_valid, y_valid, X_ctl_valid=None,
    max_epochs=20, dtype='float32', accelerator='gpu', devices=1,
    batch_size=64, verbose=True, n_warmup_steps=2 * steps_per_epoch,
    n_decay_steps=18 * steps_per_epoch)

trainer.checkpoint_callback.best_model_score   # best validation count Pearson
```

Things to know:
- The optimizers and schedules are built inside from keyword arguments
  (`muon_lr`, `adam_lr`, `lw_lr`, their `_wd`s, `loss_weights`, `ema_decay`,
  `n_warmup_steps`, `n_decay_steps`). The schedule defaults are **not** the
  CLI's, hence the two step counts above; `cherimoya fit` also raises
  `max_epochs` to reach `min_total_steps` (20000) first.
- `dtype` is `'float32'`, `'bfloat16'` or `'float16'` (Lightning's `32-true`,
  `bf16-mixed`, `16-mixed`); anything else raises.
- After `fit`, `model` holds the EMA weights **on the CPU** (Lightning moves it
  there); `.cuda()` it or `Cherimoya.load` the checkpoint for GPU inference.
- `devices` > 1 trains with DDP, with the same rules as the CLI (global
  `batch_size`, first N visible GPUs), plus: it must run as a **script**, not in
  a notebook; the script is re-run once per GPU from the top, so pass an integer
  `random_state` to `PeakGenerator` (so every rank draws the same examples) and
  guard post-training work with `if trainer.is_global_zero:`.

Walkthrough with the CLI's schedule:
<https://cherimoya.readthedocs.io/en/latest/tutorials/python_api.html>.
