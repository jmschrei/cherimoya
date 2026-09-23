Installation
============

Requirements
------------

Cherimoya requires:

- Python ≥ 3.10
- PyTorch ≥ 2.9
- A CUDA-capable GPU is **strongly recommended** for training and
  high-throughput inference. A pure-PyTorch CPU fallback path is
  provided for everything except the inference megakernel, so the
  package can be installed and the model can be run on machines
  without a GPU. Training on CPU is impractical at any realistic
  scale.

.. note::

   Cherimoya's custom GPU kernels are written in `Triton
   <https://github.com/triton-lang/triton>`_. ``triton`` is a hard
   dependency and is installed automatically from PyPI. On Linux with a
   modern CUDA toolkit and a recent PyTorch wheel, this works without
   any extra steps. On unusual configurations (custom CUDA versions,
   non-x86 hosts) Triton may need to be installed against your specific
   toolchain — see the Triton README for details.


Install from PyPI
-----------------

.. code-block:: bash

   pip install cherimoya

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/jmschrei/cherimoya.git
   cd cherimoya
   pip install -e .

Install with uv
---------------

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package manager that
can be used as a drop-in replacement for pip:

.. code-block:: bash

   uv pip install cherimoya

   # or, from source:
   git clone https://github.com/jmschrei/cherimoya.git
   cd cherimoya
   uv pip install -e .


Dependencies
------------

The following packages are installed automatically. Pinned lower-bounds
are taken from ``pyproject.toml``. They record the versions the test
suite is run against rather than the oldest release that happens to
work, so an older pin may still function — it is simply not something
Cherimoya checks.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``torch`` (≥ 2.9)
     - Tensor framework and autograd.
   * - ``triton`` (≥ 3.5.1)
     - Custom GPU kernels for Cheri Blocks (fwd+bwd) and the inference
       megakernel.
   * - ``numpy`` (≥ 1.14), ``scipy`` (≥ 1.0), ``pandas`` (≥ 1.3.3)
     - Numerical computing and tabular data handling.
   * - ``h5py`` (≥ 3.7)
     - HDF5 I/O (TF-MoDISco results, attribution arrays).
   * - ``tqdm`` (≥ 4.64.1)
     - Progress bars for data loading and training.
   * - ``tangermeme`` (≥ 1.4.0)
     - Sequence loading, attribution (saturation mutagenesis), and
       seqlet extraction primitives. The floor is tight because
       Cherimoya uses a wide slice of this package:
       ``extract_loci(return_mask=...)``, ``io._interleave_loci``,
       ``predict``'s dtype/device handling,
       ``seqlet.recursive_seqlets``, ``utils.example_to_fasta_coords``
       and ``match.extract_matching_loci``.
   * - ``bpnet-lite`` (≥ 1.0.0)
     - The multinomial NLL profile loss (``MNLLLoss``) and training
       ``Logger`` used during fitting, and ``marginalization_report``
       used by the marginalize subcommand.
   * - ``macs3``
     - Peak calling, invoked by the ``pipeline`` subcommand.
   * - ``bam2bw`` (≥ 0.4.1)
     - BAM/SAM/fragment file → bigWig conversion; used by the
       ``pipeline`` subcommand and supports remote input URLs.
   * - ``modisco`` (≥ 2.0)
     - TF-MoDISco motif discovery, invoked by the ``pipeline``
       subcommand.
   * - ``seaborn`` (≥ 0.11.2)
     - Plotting used by the marginalization report.


Optional dependencies
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``scikit-learn``
     - Required only for the ``auroc`` / ``auprc`` measures inside
       :func:`cherimoya.performance.calculate_performance_measures`.
       The rest of the package does not import it.

Documentation build dependencies (``sphinx``, ``furo``,
``sphinx-copybutton``) are listed under the ``docs`` extra in
``pyproject.toml`` and can be installed with ``pip install -e .[docs]``.


Verifying the installation
--------------------------

.. code-block:: python

   import cherimoya
   print(cherimoya.__version__)

   from cherimoya import Cherimoya
   model = Cherimoya(n_filters=128, n_layers=9)
   print("Parameters:", sum(p.numel() for p in model.parameters()))

The default 9-layer model has roughly 610K parameters. The package will
import and the model will instantiate without a GPU; a CUDA device is
only needed when you call ``.cuda()`` or pass tensors that live on a
GPU.


Hardware expectations
---------------------

The default 9-layer, 128-filter Cherimoya model is small (~610K
parameters). The dominant memory cost during training is the
activations and the optimizer state, not the parameters; both scale
linearly with ``batch_size``, ``in_window``, and ``n_filters``.

In practice:

* The default training configuration (``batch_size=64``,
  ``in_window=2114``, ``n_filters=128``, 9 layers) fits comfortably on
  a 16 GB GPU.
* Inference at ``batch_size=512`` (the CLI default) fits on the same
  hardware.
* If you have less VRAM, reduce ``batch_size`` first; reducing
  ``n_filters`` is a secondary lever.

Training time scales linearly with peak count and epochs and is
dominated by the dataloader for typical configurations. For a
ChIP-seq target with a few tens of thousands of peaks and the default
50-epoch schedule, full training is a tens-of-minutes operation on a
modern data-center GPU. See :doc:`benchmarks` for measured forward
times.


Smoke test
----------

The fastest way to confirm a working install end-to-end is to
instantiate a model, run one forward pass on random input, and check
that the output shapes are right:

.. code-block:: python

   import torch
   from cherimoya import Cherimoya

   model = Cherimoya(n_filters=128, n_layers=9)
   if torch.cuda.is_available():
       model = model.cuda()

   X = torch.randn(2, 4, 2114)
   if torch.cuda.is_available():
       X = X.cuda()

   with torch.no_grad():
       y_profile, y_counts = model(X)

   print(y_profile.shape)   # torch.Size([2, 1, 1000])
   print(y_counts.shape)    # torch.Size([2, 1])

If both shape prints match, your install is working through every
forward path the model uses. The first call on GPU will spend a
few seconds in Triton autotune — see
:doc:`troubleshooting` if it stays slow.

To run the full CLI pipeline end-to-end on real data, pick a small
ChIP-seq target from ENCODE (CTCF in K562, for example), download
the BAM and matched input BAM, and follow :doc:`recipes/chipseq_tf`.
A short ChIP-seq dataset with ~30k peaks completes the full
peak-calling-through-MoDISco pipeline in tens of minutes on a single
modern GPU.
