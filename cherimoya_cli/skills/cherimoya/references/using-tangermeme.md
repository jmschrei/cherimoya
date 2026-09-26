# Analyzing a Cherimoya model with tangermeme

Post-training analysis — attributions via saturation mutagenesis (ISM, the
method Cherimoya uses), marginalization, variant-effect scoring, and sequence
design — lives in **tangermeme**, not Cherimoya. (DeepLIFT/SHAP via
`deep_lift_shap` is an alternative on a wrapped model, and needs two rules
registered — see below.) Cherimoya's only job is to expose the right single
tensor from its `(profile, log-count)` output.

**If a `tangermeme` skill is available, invoke it for the actual analysis.**
This file covers only the Cherimoya-specific step — choosing and applying the
output wrapper — and does not duplicate tangermeme's API. Read the tangermeme
skill/docs for the analysis call; don't guess its signature.

## The wrappers (from `cherimoya`)

tangermeme's tools expect a model returning a **single tensor**, but Cherimoya
returns `(profile, log-count)`. Pick the wrapper for what you want to attribute
or optimize:

| Wrapper | Returns | Use for |
|---|---|---|
| `ProfileWrapper` | scalar per example from profile logits (weighted softmax) | profile **shape** |
| `LogCountWrapper` | the log-count prediction | **total signal** (counts) at a locus |
| `ExpectedCountsWrapper` | expected reads per base pair | per-position expected counts |
| `ControlWrapper` | raw `(profile, log-count)`, synthesizing a zero control if needed | the **inner** wrapper for control-trained models |

**`LogCountWrapper` (counts) is the recommended default; use `ProfileWrapper`
when you care about profile shape.** `ControlWrapper` is the inner layer the
profile/count wrappers sit on: for a control-trained model whose analysis tool
passes only a sequence, it supplies a zero control of the right
shape/dtype/device automatically. For a model with no controls it's a
pass-through, so wrapping is always safe.

For a multi-group model (`signal_groups` with more than one entry),
`LogCountWrapper` returns every group's count and `ProfileWrapper` softmaxes
all groups together. Pass `group=i` to either one to attribute group `i` alone;
`cherimoya attribute` takes the same index as `output_group`.

## Pattern

```python
from cherimoya import Cherimoya
from cherimoya import ControlWrapper
from cherimoya import LogCountWrapper

model = Cherimoya.load("my_run.torch", device="cuda")
model = model.eval()

# Attribute the counts (usual default); ControlWrapper handles a
# control-trained model transparently. Swap in ProfileWrapper for shape.
wrapper = LogCountWrapper(ControlWrapper(model))

# Hand `wrapper` to tangermeme (saturation_mutagenesis, deep_lift_shap,
# variant effect, ledidi design, ...).
```

## DeepLIFT/SHAP needs two rules registered

ISM needs nothing. `deep_lift_shap` does: it corrects the module types it knows
and silently treats the rest as linear, and two of Cherimoya's layers are
neither known nor linear. Without them the attributions carry no guarantee that
they sum to the change in the prediction, and for the profile head the error
exceeds the prediction itself. Always pass `attribution_ops()`:

```python
from tangermeme.deep_lift_shap import deep_lift_shap
from cherimoya.deep_lift_shap import attribution_ops

X_attr = deep_lift_shap(wrapper, X, references=references,
    additional_nonlinear_ops=attribution_ops())
```

Two more things it needs. Load the model with `compile=False` — DeepLIFT's
backward hooks replace gradients, which Inductor cannot trace, so a compiled
model graph-breaks and runs slightly slower for the same answer. And
`attribution_ops()` requires `tangermeme >= 1.5.0`.

One GPU caveat: the convergence delta is not a reliable check there. On a
small model it can land three orders of magnitude higher in some processes than
in others without the attributions moving, and a model with no Cherimoya
kernel in it does the same. Validate on CPU, or against a model rewritten with
`torch.nn.LayerNorm`, rather than trusting a GPU delta. The wrong first
backward per `(n_filters, length)` (jmschrei/cherimoya#50) is fixed, so no
warm-up backward is needed.

## What lives where

- **Cherimoya** — the model and the four wrappers above. Use these, not
  bpnet-lite's.
- **tangermeme** — `predict`, `saturation_mutagenesis`, `deep_lift_shap`,
  `variant_effect`, `ersatz` (motif insertion), `seqlet.recursive_seqlets`,
  `io.extract_loci`, MEME/bigWig/VCF loaders, plotting. Don't reimplement.
- **ledidi** — gradient-based sequence design against a wrapped model, when the
  user wants to *design* rather than *analyze*.

The CLI already wires these for the standard flow (`attribute`, `seqlets`,
`marginalize` in `references/cli-training-pipeline.md`). Reach for tangermeme
directly only for something the pipeline doesn't do — a custom attribution
target, variant scoring, or bespoke design objective.
