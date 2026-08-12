Saving and Loading Models
=========================

Cherimoya checkpoints store the constructor arguments next to the
parameter state dict. This format is robust to source-layout changes
and is safe to load with PyTorch's ``weights_only=True`` flag, which
the loader uses by default.


Saving
------

.. code-block:: python

   from cherimoya import Cherimoya

   model = Cherimoya(n_filters=128, n_layers=9)
   # ... train ...
   model.save("my_model.torch")

The saved payload is a dict with two keys:

* ``config`` — the kwargs needed to reconstruct the model (every
  argument to ``Cherimoya.__init__``).
* ``state_dict`` — the parameter tensors.

The ``CheriBlock._w_cache`` (bf16 cast cache used by the inference
megakernel) is intentionally not part of the state dict and is not
saved. It is rebuilt on the next forward pass.


State dict keys are frozen
~~~~~~~~~~~~~~~~~~~~~~~~~~

One key does not match the module tree, deliberately. Each block's
depthwise convolution weight is the parameter
``blocks.N.conv.conv_weight`` — it lives on the ``conv`` submodule — but
it is written to and read from the state dict as ``blocks.N.conv_weight``,
in that position, which is where it has always been:

.. code-block:: python

   >>> [n for n, _ in model.named_parameters() if 'conv_weight' in n][:1]
   ['blocks.0.conv.conv_weight']
   >>> [k for k in model.state_dict() if 'conv_weight' in k][:1]
   ['blocks.0.conv_weight']

The on-disk format is a compatibility surface shared with every
already-trained checkpoint and with any installed version of the package,
so it is held fixed while the module tree is free to change. A
``state_dict`` post-hook renames on the way out and
``CheriBlock._load_from_state_dict`` accepts either spelling on the way
in, so checkpoints move in both directions between versions.

Write ``block.conv.conv_weight`` when you need the parameter itself.
``block.conv_weight`` still reads it — the same object, not a copy — but
is read-only; assigning through it raises rather than leaving the
submodule holding the old tensor.


Loading
-------

.. code-block:: python

   from cherimoya import Cherimoya

   # CPU by default — fine for inspection or CPU-only inference.
   model = Cherimoya.load("my_model.torch")

   # Or load directly onto a GPU:
   model = Cherimoya.load("my_model.torch", device="cuda")

.. important::

   The weights inside a checkpoint are the **EMA snapshot**, not the
   running training weights. ``model.fit`` applies the EMA shadow
   before every save, both for the best-by-validation checkpoint and
   the final-epoch checkpoint. This is what produces the smoothed
   validation numbers reported in the training log; the model you
   load and use at inference is the same one those numbers describe.

The loader:

1. Reads the payload with ``torch.load(..., weights_only=True)``.
2. Reconstructs the module via ``Cherimoya(**payload['config'])``.
3. Calls ``load_state_dict`` with the saved state dict.
4. Moves the module to ``device`` before returning.

Because ``load_state_dict`` writes through ``data.copy_``, the
``_version`` counter on each parameter bumps and the inference
megakernel's weight cache invalidates automatically on the next
forward pass.


Best-of and final checkpoints
-----------------------------

``model.fit(...)`` writes two checkpoint files:

* ``{model.name}.torch`` — the best checkpoint observed during
  training, selected by validation count Pearson correlation. Saved
  after EMA weights are applied.
* ``{model.name}.final.torch`` — the model at the very end of
  training, also with EMA weights applied.

Both are produced via ``model.save(...)`` and are loadable with
``Cherimoya.load(...)``.

The ``.log`` file emitted alongside the checkpoints is a plain TSV
with one row per epoch. It is not used by the loader; it is for your
own inspection.


Compatibility
-------------

Checkpoints saved with the v0.1.0+ ``model.save`` format are forward-
and backward-compatible across point releases as long as the
constructor signature does not change. The version of Cherimoya you
trained with is recorded implicitly by the keys in the ``config``
dict — if you later add or rename a constructor argument, you may
need to map old keys to new ones before loading.

The pinned-version warning in the README is about this: if a future
release renames a constructor argument, an older checkpoint will
fail to load until you do the mapping.
