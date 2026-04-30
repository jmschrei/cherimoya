Changelog
=========

v0.1.0
------

Model
~~~~~

* Replaced the learnable channel-wise scaling with a fixed constant.
* Added an exponential moving average (EMA) of model weights during training.
* Changed the final profile convolution to ``kernel_width=1``.
* Set the default model size to 96 filters.
* Tuned the Muon and AdamW learning rates and weight decay values for
  improved convergence.
* Best-model selection now monitors the validation profile loss rather than
  the total validation loss.

Training
~~~~~~~~

* Default ``max_jitter`` for fitting lowered from 500 to 50 (applied across
  the package's fitting defaults).

Packaging and tooling
~~~~~~~~~~~~~~~~~~~~~

* Migrated from ``setup.py`` to ``pyproject.toml`` with ``uv`` support.
* Refactored the CLI from a monolithic script into the ``cherimoya_cli``
  modular package.
* Raised the minimum Python version to 3.10 and minimum PyTorch to 2.9.
* Added ``macs3``, ``bam2bw``, ``bpnet-lite``, ``triton``, and ``joblib``
  as dependencies.
* Added a Sphinx documentation site hosted on Read the Docs.

v0.0.1
------

* Initial release of the Cherimoya model and pipeline. 
* Includes the `CheriBlock` architecture and custom kernels.
* Features a dual-optimizer training strategy (AdamW + Muon).
* Implements a full end-to-end processing and modeling pipeline.
