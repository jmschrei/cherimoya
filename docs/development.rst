Development
===========

This page is for contributors and integrators: how the source tree
is organized, how to run the tests, and the conventions the
codebase follows. End users do not need to read this.


Repository layout
-----------------

::

   cherimoya/
   ├── cherimoya/                  # The Python package
   │   ├── __init__.py             # Public API re-exports: Cherimoya, CheriBlock, EMA
   │   ├── cherimoya.py            # Cherimoya model + EMA wrapper + fit/save/load
   │   ├── cheri.py                # CheriBlock + Triton kernels + dispatcher
   │   ├── io.py                   # PeakGenerator + PeakNegativeSampler
   │   ├── losses.py               # Profile MNLL + log1pMSE mixture loss
   │   └── performance.py          # Evaluation metrics
   ├── cherimoya_cli/              # The CLI entry-point package
   │   ├── __main__.py             # Argparse driver and subcommand registry
   │   ├── defaults.py             # All default JSON parameter dicts
   │   ├── utils.py                # JSON merging and parameter helpers
   │   └── commands/               # One file per subcommand
   │       ├── pipeline.py
   │       ├── pipeline_json.py
   │       ├── batch.py
   │       ├── fit.py
   │       ├── evaluate.py
   │       ├── attribute.py
   │       ├── seqlets.py
   │       ├── marginalize.py
   │       └── negatives.py
   ├── tests/                      # Pytest suite (see below)
   ├── docs/                       # Sphinx docs (this site)
   ├── imgs/                       # Architecture / pipeline diagrams
   ├── bench_kernels.py            # Standalone forward-path benchmark
   └── pyproject.toml              # Build, deps, and tooling config

Two top-level packages: ``cherimoya`` is the model and data plumbing,
``cherimoya_cli`` is the command-line tool. They are independent —
``cherimoya_cli`` imports ``cherimoya``, never the reverse.


Public vs. private API
----------------------

The convention is the standard Python one: anything prefixed with an
underscore is private, and may change or be removed without notice.
Explicitly:

* **Public** symbols, re-exported from ``cherimoya.__init__``:
  :class:`~cherimoya.Cherimoya`, :class:`~cherimoya.CheriBlock`,
  :class:`~cherimoya.cherimoya.EMA`, and the four model wrappers
  :class:`~cherimoya.wrappers.ControlWrapper`,
  :class:`~cherimoya.wrappers.ProfileWrapper`,
  :class:`~cherimoya.wrappers.LogCountWrapper` and
  :class:`~cherimoya.wrappers.ExpectedCountsWrapper`.
* **Public** module-level symbols:
  :func:`~cherimoya.io.PeakGenerator`,
  :class:`~cherimoya.io.PeakNegativeSampler`,
  :func:`~cherimoya.cheri.fused_dilated_conv_norm`,
  :class:`~cherimoya.cheri.FusedDilatedConvNorm`,
  :class:`~cherimoya.cheri.FusedDilatedConvNormFunc`,
  :func:`~cherimoya.performance.calculate_performance_measures` and
  its component metrics, :func:`~cherimoya.losses._mixture_loss`
  (despite the underscore — it is the trainer's loss function and
  the API is stable).

  :class:`~cherimoya.cheri.FusedDilatedConvNorm` is public because
  naming the class *is* the interface: an attribution method that wants
  to treat the fused convolution specially has to identify it by type,
  so the name and import path are a compatibility surface even though
  nothing in this package calls the class from outside
  :class:`~cherimoya.CheriBlock`.
* **Private** and may change: anything else, including the Triton
  kernels (``_fwd_*``, ``_bwd_*``, ``_fwd_inf_*``), the CPU fallback
  (``_cheri_conv_norm_cpu``), the CheriBlock weight cache
  (``_w_cache``), the ``torch.log`` module wrapper in the count head
  (``cherimoya.cherimoya._Log``), and the model's checkpoint-payload
  helper (``_init_kwargs``).

  ``_Log`` is deliberately *not* public, unlike its counterpart above.
  It exists for the same reason — giving attribution methods a node in
  the module tree — but it wraps a single elementwise call to
  ``torch.log``, which the standard DeepLIFT rescale rule already
  handles correctly. There is no reason for caller code to name the
  type, so it stays private and may be renamed or removed.


Development install
-------------------

For development, install in editable mode with the ``docs`` extra:

.. code-block:: bash

   git clone https://github.com/jmschrei/cherimoya.git
   cd cherimoya
   pip install -e .[docs]

The ``docs`` extra adds ``sphinx``, ``furo``, and
``sphinx-copybutton``, which you need to build this documentation
locally:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build

The build produces ``docs/_build/index.html``. Read the Docs runs the
same command with the same dependency set.


Running the tests
-----------------

The test suite lives in ``tests/`` and uses pytest.

.. code-block:: bash

   pytest tests/

Test files:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Covers
   * - ``tests/test_cheri.py``
     - Cheri Block forward parity (CPU vs training Triton vs
       inference megakernel), backward parity against CPU
       autograd, weight-cache invalidation, dtype matrix.
   * - ``tests/test_model.py``
     - Full Cherimoya forward/backward parity, no_grad ==
       grad-enabled equivalence, EMA-applied save/load round trip.
   * - ``tests/test_io.py``
     - ``PeakGenerator`` and ``PeakNegativeSampler`` reproducibility,
       per-epoch determinism, multi-worker equivalence.
   * - ``tests/test_ema.py``
     - EMA update/apply/restore semantics.
   * - ``tests/test_losses.py``
     - ``_mixture_loss`` shapes and edge cases.
   * - ``tests/test_performance.py``
     - Evaluation-metric correctness.
   * - ``tests/test_fit_wiring.py``
     - End-to-end fit step on tiny data: confirms optimizers,
       schedulers, EMA, and checkpoint paths are wired correctly.
   * - ``tests/test_cli_utils.py``
     - JSON merge and default-handling helpers.

Markers:

* ``@pytest.mark.cuda`` — requires a CUDA device; skipped on
  CPU-only hosts.
* ``@pytest.mark.triton`` — requires both a CUDA device and a
  Triton install.

Both markers are wired through ``tests/conftest.py``, which also
disables ``torch.compile`` for the suite so tests don't pay the
several-minute autotune cost on every run.

To run only the CPU-safe subset:

.. code-block:: bash

   pytest tests/ -m "not cuda and not triton"

To run only the GPU parity tests:

.. code-block:: bash

   pytest tests/ -m "cuda or triton"


Benchmarking
------------

``bench_kernels.py`` at the repo root is a standalone script that
times the three forward paths and checks they all agree within
machine precision. It is intentionally not packaged with the
install. Run it with:

.. code-block:: bash

   python bench_kernels.py

See :doc:`benchmarks` for the published numbers and the measurement
methodology.


Coding conventions
------------------

* **Tabs, not spaces.** The codebase uses tab indentation throughout.
* **Channels-last layout** ``(N, L, C)`` is used inside the Cheri
  Block backbone. The input stem and output heads do the necessary
  transpositions. New blocks should follow the same convention.
* **fp32 for normalization statistics** even under bf16 autocast.
  Both the CPU fallback and the Triton kernels accumulate ``sum`` /
  ``sq_sum`` in fp32; this is load-bearing for stability and
  shouldn't be changed casually.
* **Triton autotune keys.** Kernels are keyed by ``(C, L)`` so the
  same configuration is reused across batches with the same shapes.
  Adding a new kernel that depends on a new shape parameter should
  add that parameter to the key.
* **No public bias terms inside Cheri Blocks.** The input stem,
  profile head, and count head use biases; the block layers do not.
  This is intentional (see :doc:`architecture`).
