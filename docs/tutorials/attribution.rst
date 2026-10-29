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
       "output": "logcounts",
       "group": 0,
       "batch_size": 64,
       "device": "cuda",
       "ohe_filename": "attributions.ohe.npz",
       "attr_filename": "attributions.attr.npz",
       "idx_filename": "attributions.idx.npy"
   }

``output`` controls what is being attributed to:

* ``"logcounts"`` — attribute to total predicted log-counts (recommended
  for most analyses; uses :class:`cherimoya.LogCountWrapper`).
* ``"profile"`` — attribute to the predicted profile shape (uses
  :class:`cherimoya.ProfileWrapper`).

``group`` selects which signal group of a multi-group model is
attributed (see :doc:`../multi_task`); for a single-group model the
default, ``0``, is the whole output.

``algorithm`` chooses the method:

* ``"deep_lift_shap"`` (default) —
  :func:`cherimoya.deep_lift_shap.deep_lift_shap`, with ``n_shuffles``
  dinucleotide-shuffled references per sequence and Cherimoya's DeepLIFT
  rules registered. It runs ``tangermeme.deep_lift_shap``, or, for the
  log-counts, a fast path (below). ``batch_size`` counts
  sequence-reference pairs, each run forward and backward, on
  tangermeme's path.
* ``"saturation_mutagenesis"`` — ``tangermeme.saturation_mutagenesis``,
  forward passes only, three per attributed position.

The model is loaded uncompiled by default. ``"compile": true`` compiles
it for saturation mutagenesis only; DeepLIFT/SHAP never compiles, since
its backward hooks cause graph breaks and recompiles. Neither algorithm ran
faster compiled in our measurements, and compiling lengthened the first
call.

``fast``, true by default, lets DeepLIFT/SHAP of the log-counts take the
fast path, which runs several times faster on a GPU and is checked against
tangermeme's ``deep_lift_shap`` on every job that uses it (see `The fast
path for the log-counts`_). ``"fast": false`` always uses tangermeme's
``deep_lift_shap``.

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
   from cherimoya.deep_lift_shap import deep_lift_shap

   # DeepLIFT's backward hooks cannot be traced by torch.compile.
   model = Cherimoya.load("my_model.torch", device="cuda", compile=False)

   X_attr = deep_lift_shap(
       model, X,
       output="logcounts",   # or "profile" for profile shape
       group=0,
       n_shuffles=20,
       batch_size=64,
       hypothetical=True,
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

:func:`~cherimoya.deep_lift_shap.deep_lift_shap` wraps the model in
:class:`cherimoya.ControlWrapper`, which supplies zero control tracks, and
in :class:`cherimoya.LogCountWrapper` or :class:`cherimoya.ProfileWrapper`
for ``group``, and registers Cherimoya's DeepLIFT rule. A tangermeme
function called directly, such as ``tangermeme.deep_lift_shap`` or
``tangermeme.pisa``, needs the same: the wrappers, and
``additional_nonlinear_ops=attribution_ops()``. The rule is required, not
an optimization. ``FusedDilatedConvNorm`` is neither linear nor in
tangermeme's table, so without it the attributions carry no guarantee
that they sum to the change in the prediction. See
:doc:`../api/deep_lift_shap` for what the rule does and why.

tangermeme's ``deep_lift_shap`` attributes output ``target=0`` of the
model it is given. ``LogCountWrapper`` without ``group`` returns one
count per signal group, so on a multi-group model it would silently
attribute group 0 only; pass ``group`` to say which one you mean.
:func:`~cherimoya.deep_lift_shap.deep_lift_shap` raises instead.


The fast path for the log-counts
--------------------------------

:func:`~cherimoya.deep_lift_shap.deep_lift_shap` computes the log-counts'
attributions on a fast path where it can, and so does ``cherimoya
attribute`` with ``"fast": true``, its default. The method is
tangermeme's: the same rules (tangermeme's rescale rule at every GELU and
the normalization rule of ``conv_norm_op``), the same dinucleotide-shuffled
references with the same seeds, and the same hypothetical projection and
mean over references. What changes is how they are computed:

* Each sequence is forwarded once per step together with its references,
  rather than once per reference: every operation acts on each row alone,
  so the copies of a sequence that ``deep_lift_shap`` forwards are the
  same row.
* The backward runs over the sequence half of each sequence-reference pair
  only. DeepLIFT's rule for a sequence reads the reference's activations
  but never the reference's gradient, so the reference half is never
  needed.
* The forward and backward passes are written out by hand, applying
  tangermeme's rule functions and hypothetical projection, and, on a GPU,
  compiled with ``torch.compile``. Each step's references are drawn with
  one call to tangermeme's ``dinucleotide_shuffle`` while the GPU runs the
  previous step, and the host waits for the GPU once per step.

**When it is used.** The fast path applies to the ``"logcounts"`` of a
Cherimoya model computed in float32, with dinucleotide-shuffled references
or references passed in. The profile, whose softmax couples every position
to every other, other dtypes, other reference functions and jobs of four
sequences or fewer take tangermeme's ``deep_lift_shap``, and saturation
mutagenesis is unaffected. Where the fast path applies, every sequence is
checked first as tangermeme checks its input, so an unknown character
raises the same error with either path; then the first four sequences are
attributed with tangermeme's ``deep_lift_shap`` and the last two of them
timed. A job that would take tangermeme's path
less than thirty seconds at that rate stays on it, because on a GPU the
fast path warms up the first time a process runs it, compiling its passes,
which took 23 seconds on an H200 with empty compile caches and from 7 to
23 seconds with warm ones in our measurements. Otherwise the fast path
attributes every sequence and is checked against those first four: a
Pearson correlation below 0.9999 or a relative L2 distance above 1e-2 on
any of them, or an error in the fast path, gives a ``RuntimeWarning``, and
tangermeme's ``deep_lift_shap`` attributes the rest.

**How close the two paths are.** On the CPU the tests hold the fast path
to tangermeme's ``deep_lift_shap`` on small models: given the same
references, the attributions of every sequence agree to a relative L2
distance of 1e-9 in float64 and 1e-5 in float32, and each path drawing its
own references, to 1e-5 in float32. On a GPU both paths
compute float32 matrix products in TF32, which ``import cherimoya`` turns
on, and neither is bitwise reproducible, since cuDNN and Triton pick their
kernels by timing in each process. On an H200, over 40 peaks with 20
references each, the per-sequence attributions of the two paths were a
median relative L2 distance of 6.2e-4, 1.2e-4 and 2.5e-4 apart on trained
models with 64, 128 and 256 filters.

**Speed.** On an H200, with 1,000 peaks of 2,114 bp and 20 shuffles each,
the fast path attributed 7,753 sequence-reference pairs per second at
steady state with a 128-filter, 9-layer model, against 986 to 999 for
tangermeme's ``deep_lift_shap`` at a ``batch_size`` of 64 after its first
call. On the CPU with 8 threads, with the same model, it attributed 12.6
pairs per second against 4.6.

**Warnings.** The fast path warns once per call when deltas exceed
``warning_threshold``, and once when the mean delta exceeds 1% of the mean
difference between the outputs of the sequences and their references,
where tangermeme's ``deep_lift_shap`` makes both checks once per batch.


PISA: attributions of every profile position
--------------------------------------------

PISA (pairwise influence by sequence attribution) runs DeepLIFT/SHAP once
for every output position, which gives the attributions of the prediction
at each position to every input nucleotide.
:func:`~cherimoya.pisa.pisa` attributes the profile logits of one
channel, as the model returns them before any softmax, with zero control
tracks and Cherimoya's DeepLIFT rule registered:

.. code-block:: python

   from cherimoya.pisa import pisa

   model = Cherimoya.load("my_model.torch", device="cuda", compile=False)
   X_attr = pisa(model, X, channel=0, random_state=0)

``X_attr`` has shape ``(n_examples, out_length, 4, window)``, where
``out_length`` is ``window - 2 * model.trimming``, the length of the
predicted profile, and ``X_attr[i, j]`` holds the attributions of the
logit at input position ``j + model.trimming``. That is 34 MB per
sequence in float32 for a 2,114 bp window and a 9-layer model. ``channel``
indexes the ``sum(signal_groups)`` profile channels, so the two strands of
a stranded group take two calls.

Like :func:`~cherimoya.deep_lift_shap.deep_lift_shap`, it takes a fast path
where it can, with tangermeme's rules and references and the forward and
rules of the log-counts' fast path. For every reference of a sequence,
tangermeme's ``pisa`` forwards ``batch_size`` copies of the sequence and of
the reference, and runs a backward over both halves of that batch for
every ``batch_size`` output positions. The fast path:

* forwards each sequence once, together with its references;
* computes the terms of each rule that depend only on the activations, the
  rescale rule's multiplier at each GELU and the normalization rule's
  terms, once per sequence-reference pair, and shares them between all
  output positions;
* runs the backward over the sequence half of each pair only, for a chunk
  of output positions at once, sized from the memory free on the GPU.

**When it is used.** The fast path applies to a Cherimoya model with
float32 parameters, with dinucleotide-shuffled references or references
passed in. Other reference functions and jobs of two sequences or fewer
take tangermeme's ``pisa``. Where the fast path applies, every sequence is
checked first as tangermeme checks its input. Then the first two
sequences are attributed with tangermeme's ``pisa``, and the second of them
is timed. A job that would take tangermeme's path less than thirty seconds
at that rate stays on it. Otherwise the fast path attributes every
sequence and is checked against those first two with the bounds of the
log-counts' fast path, and falls back to tangermeme's ``pisa`` with a
``RuntimeWarning`` when they disagree or the fast path raises. On a GPU,
the fast path's first call in a process compiles its passes: in our
measurements on an H200, a first call took 8 to 15 seconds longer than a
second one with warm compile caches, and 41 to 48 seconds longer the
first time a model's width was compiled.

**How close the two paths are.** On the CPU the tests hold the fast path
to tangermeme's ``pisa`` on small models, with the same references: the
attributions of every sequence at every output position agree to a
relative L2 distance of 1e-9 in float64 and 1e-5 in float32. On a GPU both
paths compute float32 matrix products in TF32. On an H200, with a trained
128-filter model, the attributions of two peaks by the two paths were a
relative L2 distance of 6.2e-5 and 6.4e-5 apart. Over 3 peaks with 20
references each, the two paths were as far from float64 as each other
(median relative L2 distance per sequence):

.. list-table::
   :header-rows: 1

   * - model
     - fast path
     - tangermeme's ``pisa``
   * - 64 filters
     - 1.5e-4
     - 1.7e-4
   * - 128 filters
     - 2.2e-4
     - 2.2e-4
   * - 128 filters, stranded, 2 control tracks
     - 6.1e-4
     - 6.1e-4
   * - 256 filters
     - 1.4e-4
     - 1.4e-4

**Speed.** On an H200, with 2,114 bp peaks, 20 shuffles each and 1,000
output positions, in seconds per peak at steady state:

.. list-table::
   :header-rows: 1

   * - model
     - fast path
     - tangermeme's ``pisa``
   * - 64 filters
     - 0.56
     - 9.33
   * - 128 filters
     - 1.06
     - 16.8
   * - 128 filters, stranded, 2 control tracks
     - 1.07
     - 17.1
   * - 256 filters
     - 2.42
     - 32.7
   * - 512 filters, untrained
     - 6.06
     - 81.5

**Warnings.** The fast path warns once per call when deltas exceed
``warning_threshold``, where tangermeme's ``pisa`` warns once per chunk of
output positions. In TF32, on one peak with 20 references, the deltas of
both paths exceeded the default of 1e-3 for some output positions: 19
and 20 of 20,000 with a trained 128-filter model, and 1,885 and 1,938 with
a trained 64-filter model.


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
