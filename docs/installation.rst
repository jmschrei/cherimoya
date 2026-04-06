Installation
============

Requirements
------------

Cherimoya requires:

- Python ≥ 3.10
- PyTorch ≥ 2.6 (with CUDA support recommended)
- A CUDA-capable GPU (for the fused Triton kernel)

.. warning::
   Cherimoya relies on `Triton <https://github.com/triton-lang/triton>`_ for its custom
   GPU kernels. Due to its low-level nature, Triton requires a hardware-dependent
   installation that varies based on your GPU architecture and CUDA version.
   Please ensure your hardware is compatible and that you have the appropriate
   drivers and PyTorch-compatible CUDA toolkit installed.

Install from PyPI
-----------------

.. code-block:: bash

   pip install cherimoya

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/jmschrei/cherimoya.git
   cd cherimoya
   pip install -e .

Install with uv
---------------

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package manager that can be used
as a drop-in replacement for pip.

.. code-block:: bash

   uv pip install cherimoya

Install from source with uv:

.. code-block:: bash

   git clone https://github.com/jmschrei/cherimoya.git
   cd cherimoya
   uv pip install -e .

Dependencies
------------

The following packages are installed automatically:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``torch``
     - Deep learning framework
   * - ``triton``
     - Custom GPU kernel compilation
   * - ``numpy``, ``scipy``, ``pandas``
     - Numerical computing and data handling
   * - ``h5py``
     - HDF5 file I/O
   * - ``tangermeme``
     - Genomic sequence utilities and attribution
   * - ``bpnet-lite``
     - Loss functions and logging from BPNet
   * - ``macs3``
     - Peak calling
   * - ``bam2bw``
     - BAM/SAM to bigWig conversion

Optional Dependencies
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``scikit-learn``
     - Required only for AUROC/AUPRC evaluation metrics
   * - ``modisco``
     - TF-MoDISco motif discovery
   * - ``seaborn``
     - Visualization

Verifying the Installation
--------------------------

.. code-block:: python

   import cherimoya
   print(cherimoya.__version__)

   from cherimoya import Cherimoya
   model = Cherimoya(n_filters=64, n_layers=9)
   print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
