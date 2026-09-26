Variant Effect Prediction
=========================

Cherimoya predicts a base-pair-resolution profile and total counts
from sequence alone, so quantifying the predicted effect of a variant
reduces to running the model on the reference and alternate sequences
and comparing the outputs. This page documents the common patterns,
both from the CLI and from Python.


Saturation mutagenesis (CLI)
----------------------------

The simplest *exhaustive* variant scan is the ``attribute`` subcommand
with ``"algorithm": "saturation_mutagenesis"``, which performs in-silico
saturation mutagenesis over the central 400 bp of each input sequence —
every possible single-nucleotide substitution, scored by its effect on
the predicted log counts (or profile). The subcommand's default
algorithm is DeepLIFT/SHAP, so the key has to be set:

.. code-block:: bash

   cherimoya attribute -p attribute_params.json

Example JSON:

.. code-block:: json

   {
       "model": "my_model.torch",
       "sequences": "hg38.fa",
       "loci": "peaks.narrowPeak",
       "chroms": ["chr2", "chr4", "chr5"],
       "algorithm": "saturation_mutagenesis",
       "output": "counts",
       "batch_size": 512,
       "device": "cuda",
       "ohe_filename": "attributions.ohe.npz",
       "attr_filename": "attributions.attr.npz",
       "idx_filename": "attributions.idx.npy"
   }

``output`` can be ``"counts"`` (recommended for most analyses) or
``"profile"``. The output array has shape ``(n_examples, 4, 400)`` and
contains hypothetical importance scores per base per channel —
equivalent to the predicted delta when substituting that nucleotide at
that position.

This is the CLI equivalent of running ``tangermeme.saturation_mutagenesis``
on each locus.


Single-prediction inference (Python)
------------------------------------

For a one-off prediction at a single sequence (e.g. ref vs alt), use
``tangermeme.predict.predict``:

.. code-block:: python

   import torch
   from cherimoya import Cherimoya
   from tangermeme.predict import predict

   model = Cherimoya.load("my_model.torch", device="cuda")
   model.eval()

   # X_ref, X_alt: (N, 4, 2114) one-hot tensors.
   y_profile_ref, y_counts_ref = predict(model, X_ref, batch_size=64, device="cuda")
   y_profile_alt, y_counts_alt = predict(model, X_alt, batch_size=64, device="cuda")

   delta_counts = y_counts_alt - y_counts_ref


Exhaustive ISM (Python)
-----------------------

For an in-silico saturation mutagenesis sweep over a region, use
``tangermeme.saturation_mutagenesis.saturation_mutagenesis``:

.. code-block:: python

   import torch
   from cherimoya import Cherimoya
   from cherimoya import ControlWrapper
   from cherimoya import LogCountWrapper
   from tangermeme.saturation_mutagenesis import saturation_mutagenesis

   model = Cherimoya.load("my_model.torch", device="cuda")
   # ControlWrapper passes control-free models straight through, so it is
   # safe to apply unconditionally.
   model = ControlWrapper(model)
   wrapper = LogCountWrapper(model)

   mid = X.shape[-1] // 2
   X_attr = saturation_mutagenesis(
       wrapper, X,
       batch_size=512,
       device="cuda",
       hypothetical=True,
       start=mid - 200, end=mid + 200,
   )

This is what the ``attribute`` CLI subcommand calls internally.


Variant effect helpers (Python)
-------------------------------

For per-variant scoring (not exhaustive ISM), ``tangermeme.variant_effect``
provides ``substitution_effect``, ``deletion_effect``,
``insertion_effect``, and ``marginalize`` — each of which wraps the
ref/alt forward-pair pattern in a helper that handles batching and
padding. See the tangermeme documentation for signatures.

For ad-hoc substitutions or insertions where you want full control,
``tangermeme.ersatz`` provides ``substitute``, ``insert``, ``delete``,
``multisubstitute``, and ``dinucleotide_shuffle`` operations on one-hot
encoded tensors. The typical pattern is:

.. code-block:: python

   from tangermeme.ersatz import substitute
   from tangermeme.predict import predict

   X_alt = substitute(X_ref, "<alt sequence>", start=position)
   y_alt = predict(model, X_alt, batch_size=64, device="cuda")


Motif marginalization (CLI)
---------------------------

To quantify the average causal effect of *inserting* a known motif
into negative backgrounds, use the ``marginalize`` subcommand. This
inserts each motif from a MEME file at the center of negative loci
and reports the predicted change in profile and counts:

.. code-block:: bash

   cherimoya marginalize -p marginalize_params.json

Output goes into a report directory containing per-motif CSVs and an
HTML summary. The marginalize step is run automatically at the end of
``cherimoya pipeline`` when a motif database is provided to
``pipeline-json``.
