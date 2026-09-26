Attribution and Motif Analysis
==============================

This tutorial covers per-base attribution, seqlet extraction, and
TF-MoDISco motif discovery using a trained Cherimoya model.

For ref/alt and per-variant scoring see :doc:`variant_effect`.


What are attributions?
----------------------

Attributions quantify how much each base in the input sequence
contributes to the model's prediction. ``cherimoya attribute``
computes hypothetical importance scores with **DeepLIFT/SHAP** by
default — gradients propagated relative to dinucleotide-shuffled
reference sequences — and with **saturation mutagenesis** —
evaluating every single-nucleotide substitution and taking the
predicted delta — when ``algorithm`` is ``"saturation_mutagenesis"``.
Either way the output array has shape ``(n_examples, 4, window)``: one
score per base per position.

These scores are commonly used to:

* identify transcription factor binding sites,
* discover *de novo* motifs (via TF-MoDISco), and
* understand regulatory grammar.


Computing attributions (CLI)
----------------------------

.. code-block:: bash

   cherimoya attribute -p attribute_params.json

Example JSON:

.. code-block:: json

   {
       "model": "my_model.torch",
       "sequences": "hg38.fa",
       "loci": "peaks.narrowPeak",
       "chroms": ["chr2", "chr4", "chr5"],
       "algorithm": "deep_lift_shap",
       "output": "counts",
       "group": 0,
       "batch_size": 64,
       "device": "cuda",
       "ohe_filename": "attributions.ohe.npz",
       "attr_filename": "attributions.attr.npz",
       "idx_filename": "attributions.idx.npy"
   }

``output`` controls what is being attributed to:

* ``"counts"`` — attribute to total predicted log-counts (recommended
  for most analyses; uses :class:`cherimoya.LogCountWrapper`).
* ``"profile"`` — attribute to the predicted profile shape (uses
  :class:`cherimoya.ProfileWrapper`).

``group`` selects which signal group of a multi-group model is
attributed (see :doc:`../multi_task`); for a single-group model the
default, ``0``, is the whole output.

``algorithm`` chooses the method:

* ``"deep_lift_shap"`` (default) — ``tangermeme.deep_lift_shap`` with
  ``n_shuffles`` dinucleotide-shuffled references per sequence and
  Cherimoya's DeepLIFT rules registered. ``batch_size`` counts
  sequence-reference pairs, each run forward and backward.
* ``"saturation_mutagenesis"`` — ``tangermeme.saturation_mutagenesis``,
  forward passes only, three per attributed position.

The model is loaded uncompiled by default. ``"compile": true`` compiles
it for saturation mutagenesis only; DeepLIFT/SHAP never compiles, since
its backward hooks break the compiled forward. Neither algorithm ran
faster compiled in our measurements, and compiling lengthened the first
call.

The CLI automatically:

1. Loads sequences from ``loci`` on ``chroms`` and filters out any
   example containing an ``N`` over the input window.
2. Wraps the model with :class:`cherimoya.ControlWrapper` (passing
   zero controls if the model has none) and then with the chosen
   output wrapper.
3. Runs the chosen algorithm and keeps the central ``attr_window``
   (default 400) bp of each input. DeepLIFT/SHAP attributes the whole
   input window and the centre is sliced out; saturation mutagenesis
   only mutates the centre.
4. Writes one-hot encoded inputs to ``ohe_filename``, hypothetical
   importance scores to ``attr_filename``, and a boolean mask
   (``idx_filename``) recording which loci survived the N-filter, so
   that downstream stages can re-align the attribution rows back to
   the original locus list.

The ``.npz`` outputs store the array under key ``arr_0``
(``numpy.savez_compressed`` default).


Computing attributions (Python)
-------------------------------

This is what ``cherimoya attribute`` does with its defaults:
DeepLIFT/SHAP against 20 dinucleotide-shuffled references per
sequence, keeping the central 400 bp.

.. code-block:: python

   from cherimoya import Cherimoya
   from cherimoya import ControlWrapper
   from cherimoya import LogCountWrapper
   from cherimoya import ProfileWrapper
   from cherimoya.deep_lift_shap import attribution_ops
   from tangermeme.deep_lift_shap import deep_lift_shap

   # DeepLIFT's backward hooks cannot be traced by torch.compile.
   model = Cherimoya.load("my_model.torch", device="cuda", compile=False)

   # ControlWrapper wraps the model so that .forward(X) returns just the
   # profile/counts tuple, supplying zero controls if the model has none.
   model = ControlWrapper(model)
   wrapper = LogCountWrapper(model, group=0)   # or ProfileWrapper(model, group=0) for profile shape

   X_attr = deep_lift_shap(
       wrapper, X,
       n_shuffles=20,
       batch_size=64,
       hypothetical=True,
       additional_nonlinear_ops=attribution_ops(),
       device="cuda",
       random_state=0,
   )

   # Keep the central 400 bp of each input sequence.
   mid = X.shape[-1] // 2
   X_attr = X_attr[:, :, mid - 200:mid + 200]

This produces hypothetical importance scores. To get actual
importance, multiply elementwise by the one-hot encoding and sum
across the channel axis:

.. code-block:: python

   importance = (X_attr * X[:, :, mid - 200:mid + 200]).sum(dim=1)

``additional_nonlinear_ops=attribution_ops()`` is required, not an
optimization. ``FusedDilatedConvNorm`` and the profile head's logit
scaling are neither linear nor in tangermeme's table, so without them
the attributions carry no guarantee that they sum to the change in the
prediction — and for the profile head the error exceeds the prediction
itself. See :doc:`../api/deep_lift_shap` for what each rule does and
why.

tangermeme's ``deep_lift_shap`` attributes output ``target=0`` of the
model it is given. ``LogCountWrapper`` without ``group`` returns one
count per signal group, so on a multi-group model it would silently
attribute group 0 only; pass ``group`` to say which one you mean.


Saturation mutagenesis instead of DeepLIFT/SHAP
-----------------------------------------------

Saturation mutagenesis (ISM) makes forward passes only and needs
nothing registered, at the cost of three forward passes per attributed
position: on a 9-layer model over a 2114 bp window, DeepLIFT/SHAP takes
about 5 ms against 73 ms for a central-400 bp ISM, both measured on one
H200 at batch 8. Only the positions between ``start`` and ``end`` are
mutated:

.. code-block:: python

   from tangermeme.saturation_mutagenesis import saturation_mutagenesis

   model = ControlWrapper(Cherimoya.load("my_model.torch", device="cuda",
       compile=False))
   wrapper = LogCountWrapper(model, group=0)

   mid = X.shape[-1] // 2
   X_attr = saturation_mutagenesis(
       wrapper, X,
       batch_size=64,
       device="cuda",
       hypothetical=True,
       start=mid - 200, end=mid + 200,
   )

``cherimoya attribute`` runs this with ``"algorithm":
"saturation_mutagenesis"``.


Identifying seqlets
-------------------

Seqlets are contiguous subsequences with high attribution scores that
likely correspond to functional elements — binding motifs, in
practice. They are extracted via TF-MoDISco-style recursive seqlet
calling on the (attribution × one-hot) signal.

**CLI:**

.. code-block:: bash

   cherimoya seqlets -p seqlet_params.json

**Python:**

.. code-block:: python

   from tangermeme.seqlet import recursive_seqlets

   importance = (X_attr * X_ohe).sum(dim=1)

   seqlets = recursive_seqlets(
       importance,
       threshold=0.01,
       min_seqlet_len=4,
       max_seqlet_len=25,
       additional_flanks=3,
   )

The default seqlet parameters mirror the CLI defaults (see
:doc:`../cli`). After the recursive call, the CLI converts
example-relative coordinates to genome coordinates using
``tangermeme.utils.example_to_fasta_coords`` and writes a BED file.


tomtom-lite annotation
----------------------

When the pipeline JSON provides a ``motifs`` MEME file, the
``pipeline`` subcommand additionally invokes ``ttl`` (tomtom-lite) on
the seqlet BED to annotate each seqlet with its closest match against
the motif database. This is what produces
``{name}.seqlets_annotated.bed`` and ``{name}.motif_seqlet_count.tsv``.

If you want to run this independently, the equivalent shell call is:

.. code-block:: bash

   ttl -f hg38.fa -b seqlets.bed \
       -t JASPAR_2024.meme \
       -s 100 -m 1000 -a 100 -c 250 -j -1 > seqlets_annotated.bed


TF-MoDISco motif discovery
--------------------------

TF-MoDISco clusters seqlets into motif patterns. The pipeline runs
this automatically; run it manually like so:

.. code-block:: bash

   modisco motifs \
       -s attributions.ohe.npz \
       -a attributions.attr.npz \
       -n 100000 \
       -o modisco_results.h5

   modisco report \
       -i modisco_results.h5 \
       -o modisco_report/ \
       -s ./

The pipeline's TF-MoDISco step uses 100,000 seqlets by default
(``modisco_motifs_parameters.n_seqlets``).


Motif marginalization
---------------------

To quantify the *causal* effect of an inserted motif on the predicted
profile and counts:

.. code-block:: bash

   cherimoya marginalize -p marginalize_params.json

The output directory contains per-motif plots and a summary report
showing how predictions change when each motif is inserted at the
center of negative (non-peak) backgrounds.

.. note::

   Marginalization requires a motif database in MEME format. JASPAR
   provides such files for many species; the latest as of writing is
   JASPAR 2024.
