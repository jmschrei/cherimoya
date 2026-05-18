Cherimoya
=========

.. image:: https://img.shields.io/pypi/v/cherimoya.svg
   :target: https://pypi.org/project/cherimoya/
   :alt: PyPI Version

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/jmschrei/cherimoya/blob/main/LICENSE
   :alt: License


.. image:: ../imgs/cherimoya.png
   :width: 1000px
   :align: center
   :alt: Cherimoya logo

|

**A compact deep learning model for predicting genomic profile data from DNA
sequence.**

Cherimoya predicts genomic modalities — transcription factor binding,
chromatin accessibility, and transcription initiation — directly from DNA
sequence. It pairs a lightweight ConvNeXt-style backbone with custom Triton
GPU kernels for both training and inference, and ships with an end-to-end
CLI that takes BAM files through peak calling, training, attribution, and
motif discovery in a single command.

.. admonition:: Under Active Development

   Cherimoya is still evolving and may introduce breaking changes between
   versions. Pin the version you train with if you need to reload
   checkpoints later.

.. warning::

   **Hitting a** ``torch.compile`` **or CUDA-graphs error at inference
   time?** Cherimoya's forward is wrapped in
   ``torch.compile(mode='max-autotune')`` by default, and some usage
   patterns (notably loading several model instances in one process)
   can trip the CUDA-graph runtime. Load with ``compile=False`` to
   bypass it — forward outputs are numerically equivalent. See
   :ref:`torch_compile_cudagraph_errors` for the full explanation and
   examples. The same advice applies to any other ``torch.compile`` or
   CUDA-graph error you don't immediately recognize: the safest fix is
   ``Cherimoya.load(..., compile=False)``.


Where to start
--------------

* **Bioinformaticians** running Cherimoya on their data: read
  :doc:`installation`, then :doc:`tutorials/cli_pipeline` for an
  end-to-end walkthrough, then pick the recipe that matches your assay
  in :doc:`recipes/chipseq_tf`, :doc:`recipes/atacseq`, or
  :doc:`recipes/dnaseq`. Comparing across conditions?
  :doc:`recipes/differential`.
* **Researchers** using Cherimoya from Python: read
  :doc:`installation`, then :doc:`tutorials/python_api`, then explore
  :doc:`tutorials/attribution`, :doc:`tutorials/variant_effect`, and
  :doc:`tutorials/save_load`.
* **Developers** integrating Cherimoya or contributing to it: read
  :doc:`development` for repo layout and the test suite, then
  :doc:`architecture` and the :doc:`api/model` / :doc:`api/cheri`
  reference pages.

If a term is unfamiliar, :doc:`glossary` defines everything used in
the rest of these docs. If something is going wrong, start with
:doc:`troubleshooting`.


Design highlights
-----------------

* **Cheri Blocks**. A dilated depthwise convolution fused with a
  per-example layer normalization and a channel-mixing MLP, implemented
  as a custom Triton kernel. The default 9-layer model is ~340K
  parameters with a 1115 bp receptive field.
* **Three forward paths, one set of weights**. A CPU fallback, a
  Triton fwd+bwd kernel for training, and a fwd-only megakernel for
  inference, all numerically equivalent up to ~1e-5 max-abs.
* **Dual-optimizer training**. Muon for 2D projection weights, AdamW
  for everything else, with hyperparameters tuned via large-scale
  sweeps.
* **Learned loss balancing**. Kendall-Gal uncertainty weighting with
  two learnable scalars replaces a fixed profile/counts loss weight.
* **EMA at evaluation**. An exponential moving average of the
  parameters is maintained during training and used at evaluation,
  smoothing both the validation curve and the final predictions.
* **Stability-first defaults**. Small fixed residual scale at
  initialization, no biases inside Cheri Blocks, no weight decay on
  Muon-routed weights, and a 5-epoch warmup before cosine decay.

See :doc:`architecture` for the full story and :doc:`benchmarks` for
measured numbers.

---

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   glossary

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   architecture
   benchmarks
   troubleshooting
   development
   CHANGELOG

.. toctree::
   :maxdepth: 2
   :caption: Command-Line Interface

   tutorials/cli_pipeline
   cli
   recipes/chipseq_tf
   recipes/atacseq
   recipes/dnaseq
   recipes/differential

.. toctree::
   :maxdepth: 2
   :caption: Python API

   tutorials/python_api
   tutorials/attribution
   tutorials/variant_effect
   tutorials/save_load

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/model
   api/cheri
   api/io
   api/losses
   api/performance
