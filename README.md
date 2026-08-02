<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/cherimoya.png">

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/cherimoya?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/cherimoya)
[![PyPI Version](https://img.shields.io/pypi/v/cherimoya.svg)](https://pypi.org/project/cherimoya/)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/jmschrei/cherimoya/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://cherimoya.readthedocs.io)


> [!IMPORTANT]
> Cherimoya is under active development and may introduce breaking changes between versions. Pin the version you train with if you need to reload checkpoints later.

Cherimoya is a compact deep learning model for predicting genomic modalities measured by high-throughput sequencing experiments, such as transcription factor binding, chromatin accessibility, transcription initiation, and many others, directly from DNA sequence. Cherimoya builds upon the ChromBPNet model through a backbone made up of new Cheri Block units, more sophisticated optimization, and custom GPU kernels to accelerate training and inference. In addition to the model, this repository provides an end-to-end CLI for training and using these models, including a pipeline command that takes BAM files through peak calling, training, attribution, and motif discovery in a single command. The default 9-layer model is **~610K parameters** and runs a full forward in **under a millisecond per batch on an H200**, while delivering state-of-the-art performance.

<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/cheri-model.png">

### Design highlights

The backbone is built from **Cheri Blocks** — each a depthwise dilated convolution followed by per-example layer normalization and a channel-mixing MLP, allowing spatial and channel information to be aggregated cheaply and at separate stages of each block. Training uses a tuned three-way optimizer split: **Muon** for projection weights, **SGD** for the Kendall uncertainty weights, **AdamW** for everything else. The profile and counts losses are combined via **Kendall-Gal uncertainty weighting** with one learnable weight per output track, replacing the usual fixed loss weight with one the model balances on its own. An **exponential moving average** of the parameters is maintained during training and used at evaluation, smoothing both the validation curve and the final predictions. Several **stability-first** choices keep deep stacks well-behaved: a small fixed residual scale at initialization, no biases in the blocks, minimal weight decay on the Muon-routed projection weights, and a small warmup before cosine decay. Details of the architecture and the training recipe were arrived at via agent-driven autoresearch exploration of the design space. See [the architecture docs](https://cherimoya.readthedocs.io/en/latest/architecture.html) for more information.

### Installation

```bash
pip install cherimoya          # or: uv pip install cherimoya
```

From source:

```bash
git clone https://github.com/jmschrei/cherimoya.git
cd cherimoya && pip install -e .
```

Or pull the prebuilt Docker image, published to the GitHub Container Registry on every push to `main`:

```bash
docker pull ghcr.io/jmschrei/cherimoya:latest      # or pin a version: ghcr.io/jmschrei/cherimoya:0.2.0
```

GPU acceleration requires Triton and a CUDA-capable device; a pure-PyTorch CPU fallback is available for everything except the inference megakernel. See [the installation guide](https://cherimoya.readthedocs.io/en/latest/installation.html) for Triton compatibility notes.

### What you can do with Cherimoya

- Train a sequence-to-function model on [TF ChIP-seq](https://cherimoya.readthedocs.io/en/latest/recipes/chipseq_tf.html), [ATAC-seq](https://cherimoya.readthedocs.io/en/latest/recipes/atacseq.html), [DNase-seq](https://cherimoya.readthedocs.io/en/latest/recipes/dnaseq.html), or any signal that can be expressed as a stranded or unstranded coverage track. Multi-task models that share a backbone across several modalities — for example ATAC co-trained with several stranded TFs — are also supported; see [the multi-task guide](https://cherimoya.readthedocs.io/en/latest/multi_task.html).
- Compute per-base attribution scores via [*in silico* saturation mutagenesis](https://cherimoya.readthedocs.io/en/latest/tutorials/attribution.html).
- Call seqlets and discover *de novo* motifs with [TF-MoDISco](https://cherimoya.readthedocs.io/en/latest/tutorials/attribution.html#tf-modisco-motif-discovery).
- Annotate seqlets against a known motif database via [tomtom-lite](https://cherimoya.readthedocs.io/en/latest/tutorials/attribution.html#tomtom-lite-annotation).
- [Marginalize](https://cherimoya.readthedocs.io/en/latest/tutorials/variant_effect.html#motif-marginalization-cli) the contribution of inserted motifs in counterfactual sequence designs.
- [Score variants](https://cherimoya.readthedocs.io/en/latest/tutorials/variant_effect.html) by predicting their effects on the underlying profile and counts.
- Reproduce a training run bit-for-bit from a seed — the peak/negative sampler is a pure function of `(seed, epoch, index)`, and `num_workers > 1` is purely a speed optimization that produces the same batch sequence as `num_workers = 1`.
- Stream remote BAM, BED, and FASTA inputs directly without downloading them first.

### The Cheri Block

<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/cheri-block.png">

Each block performs a 3-tap dilated depthwise convolution, a per-example layer normalization, a linear expansion to `expansion × n_filters` channels, a GELU non-linearity, a contraction back to `n_filters` channels, and a residual connection scaled by a small fixed constant (`residual_scale`, default `0.15`). The convolution and normalization are fused into a custom Triton kernel; under `torch.no_grad()` the entire block (including the MLP) collapses into a second fused megakernel for inference. The default 9-layer model uses dilations `1, 2, 4, ..., 256`, giving a receptive field of 1115 bp and a 2114 → 1000 bp input/output by default. See [the architecture docs](https://cherimoya.readthedocs.io/en/latest/architecture.html) for receptive field math, kernel internals, and the rationale for each design choice.

### Performance

Per-call latency (ms) on an NVIDIA H200 for a single Cheri Block at `N=512, L=1024, C=128, dilation=4`, in `.eval()` mode. The inference megakernel is automatically dispatched under `torch.no_grad()`; calling `.eval()` first lets it reuse a precomputed bf16 weight cast across calls instead of recomputing every call, which matters more at small batches (see the benchmarks page).

| dtype | training-fwd | megakernel |
|---|---|---|
| fp32 | 1.337 | **0.498** |
| bf16 | 0.707 | **0.347** |
| fp16 | 0.706 | **0.347** |

All paths agree on the fp32 model output to within ~1e-5 max-abs, so existing trained checkpoints produce numerically equivalent predictions through training-fwd and the megakernel paths. A pure-PyTorch CPU fallback is also available for development and one-off evaluation on a laptop. See [the benchmarks page](https://cherimoya.readthedocs.io/en/latest/benchmarks.html) for small-batch breakdowns and full methodology.

### End-to-end CLI pipeline

<img src="https://github.com/jmschrei/cherimoya/blob/main/imgs/pipeline.png" width=70%>

The CLI strings the full pipeline — peak calling, signal extraction, training, attribution, seqlet calling, motif discovery — into a single reproducible run. Each step is parameterized through a JSON file, which serves both as a runtime config and a permanent record of what was run. The user-supplied JSON is merged with sensible defaults, so practical configs are short.

**Step 1: generate a pipeline JSON from raw data pointers.** Provide a reference genome, one or more signal files, optional controls, a BED of positive loci, and a motif database. For stranded ChIP-seq with input controls (full recipe [here](https://cherimoya.readthedocs.io/en/latest/recipes/chipseq_tf.html)):

```bash
cherimoya pipeline-json \
    -s hg38.fa -p peaks.narrowPeak \
    -i chipseq_rep1.bam -i chipseq_rep2.bam \
    -c input_rep1.bam -c input_rep2.bam \
    -m JASPAR_2024.meme -n my_experiment -o pipeline.json
```

Note: `-i` is the ChIP signal (IP reads) and `-c` is the unenriched-DNA input control (optional).

For unstranded paired-end ATAC-seq with the standard +4/−4 fragment shift (full recipe [here](https://cherimoya.readthedocs.io/en/latest/recipes/atacseq.html)):

```bash
cherimoya pipeline-json \
    -s hg38.fa -p peaks.narrowPeak \
    -i fragments.bam -m JASPAR_2024.meme \
    -n atac_experiment -o pipeline.json \
    -ps 4 -ns -4 -u -pe
```

Any input path can be remote (S3, HTTPS, etc.); the pipeline streams reads through `bam2bw` directly.

**Step 2: edit the JSON if you want to override defaults** — model width, training/validation chromosomes, seqlet p-value threshold, MoDISco settings, anything. Then run:

```bash
cherimoya pipeline -p pipeline.json
```

This calls peaks with MACS3, samples GC-matched negatives, trains a Cherimoya model, computes attributions via saturation mutagenesis, calls seqlets, annotates them with tomtom-lite, and runs TF-MoDISco. The outputs land in the working directory: a `.torch` model checkpoint and training log, per-track bigWigs, a saturation-mutagenesis attribution array (`.npz`), a seqlet table with tomtom-lite annotations, and a TF-MoDISco results H5. Each sub-step writes its own JSON snapshot so individual stages can be re-run in isolation with the `negatives`, `fit`, `evaluate`, `attribute`, `marginalize`, or `seqlets` subcommands. See [the CLI reference](https://cherimoya.readthedocs.io/en/latest/cli.html) for the full command list and JSON schema.

### Python API and saving/loading

For programmatic use, the public API is `Cherimoya` (the model), `CheriBlock` (the building block), `EMA` (the parameter exponential-moving-average wrapper used during training), and four output wrappers — `ControlWrapper`, `ProfileWrapper`, `LogCountWrapper`, and `ExpectedCountsWrapper` — that expose a single tensor from the model's `(profile, log-count)` output for attribution and design tools. See the [Python API tutorial](https://cherimoya.readthedocs.io/en/latest/tutorials/python_api.html) for an end-to-end training walkthrough:

```python
from cherimoya import Cherimoya

model = Cherimoya(n_filters=128, n_layers=9).cuda()
y_profile, y_counts = model(X)              # X: (N, 4, L) one-hot DNA
```

Models are saved as a config + state_dict bundle, not a pickled module. This format is robust to source-layout changes and safe to load with `weights_only=True`:

```python
model.save("my_model.torch")
model = Cherimoya.load("my_model.torch")              # CPU by default
model = Cherimoya.load("my_model.torch", device="cuda")
```

Older checkpoints saved with `torch.save(model, ...)` are not compatible with `Cherimoya.load` and must be retrained. The CLI subcommands and `model.fit(...)` use this format internally. See [the save/load guide](https://cherimoya.readthedocs.io/en/latest/tutorials/save_load.html) for full semantics (including that the saved weights are the EMA snapshot) and [the Python API reference](https://cherimoya.readthedocs.io/en/latest/api/model.html) for the full `fit()` and `predict()` signatures.

### Claude Code skill

Cherimoya ships an agent skill for [Claude Code](https://claude.com/claude-code) that teaches the assistant to drive the CLI pipeline and Python API on your behalf — working out which inputs you have, choosing assay-appropriate settings, calling the right subcommands, and interpreting the outputs. It uses progressive disclosure: a short router plus topic-specific reference files (training pipeline, input files, assay defaults, interpreting outputs, troubleshooting, CLI reference, Python usage, tangermeme analysis, and a concepts primer) that load only when relevant. The skill is built to ask a clarifying question when an input is ambiguous rather than guess, and to explain in plain language which defaults it applied and why — so it stays useful even if you're new to sequence modeling.

Install it with the bundled subcommand:

```bash
cherimoya install-skill
```

This copies the skill into `~/.claude/skills/cherimoya`. Options:

- `-d, --directory DIR` — install into a different skills directory (default `~/.claude/skills`).
- `--symlink` — symlink the packaged skill instead of copying it, so in-place edits to the installed package are reflected without reinstalling.
- `-f, --force` — overwrite an existing installation at the destination.

Restart Claude Code (or reload skills) to pick it up, then just describe what you want — for example, *"I have a ChIP-seq BAM and a genome FASTA here, train a Cherimoya model on them"* — and the skill guides the run, asking about anything it needs.

### Documentation

Full documentation, including tutorials, architecture details, and API reference, is at [cherimoya.readthedocs.io](https://cherimoya.readthedocs.io). New to the terminology? See the [glossary](https://cherimoya.readthedocs.io/en/latest/glossary.html). Hitting an error? See the [troubleshooting page](https://cherimoya.readthedocs.io/en/latest/troubleshooting.html). The [changelog](https://cherimoya.readthedocs.io/en/latest/CHANGELOG.html) tracks user-visible changes between versions.

### Citation

If you use Cherimoya in published work, please cite the repository. A formal preprint is forthcoming.

### License

MIT. See [`LICENSE`](https://github.com/jmschrei/cherimoya/blob/main/LICENSE).
