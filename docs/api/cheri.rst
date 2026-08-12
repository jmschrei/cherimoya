cherimoya.cheri
===============

.. module:: cherimoya.cheri

The Cheri Block — Cherimoya's adaptation of the ConvNeXt block — plus
the fused dilated-conv-plus-layer-norm dispatcher used inside it. For
the high-level model see :doc:`model`; for the architectural story
behind these primitives see :doc:`../architecture`.


CheriBlock
----------

.. autoclass:: CheriBlock
   :members: forward
   :undoc-members:
   :show-inheritance:

   .. rubric:: Constructor

   .. automethod:: __init__


fused_dilated_conv_norm
-----------------------

.. autofunction:: fused_dilated_conv_norm

Public dispatcher used inside :class:`CheriBlock`. Routes to the
training Triton kernel on CUDA when gradients are enabled, otherwise
to the pure-PyTorch fallback (``_cheri_conv_norm_cpu``). Numerically
equivalent up to floating-point error.


FusedDilatedConvNorm
--------------------

.. autoclass:: FusedDilatedConvNorm
   :members: forward
   :undoc-members:
   :show-inheritance:

   .. rubric:: Constructor

   .. automethod:: __init__

An ``nn.Module`` wrapper around :func:`fused_dilated_conv_norm`, held by
each :class:`CheriBlock` as ``block.conv``. It owns the depthwise weight
(``block.conv.conv_weight``) and adds nothing but module dispatch — the
kernel, its launches, and the numerics are identical to calling the
function directly.

Its purpose is to give attribution methods that discover their targets by
walking ``named_modules`` — DeepLIFT and SHAP in particular — a concrete
node to hook or substitute. Registering a handler for it is the caller's
job; the module is inert on its own, so an attribution run that does not
mention it behaves exactly as it did before this class existed.

.. note::

   The inference megakernel calls the fused op directly rather than
   through this submodule, so hooks registered on ``block.conv`` fire on
   the CPU and Triton training paths but **not** under ``no_grad`` on
   CUDA. Attribution runs are unaffected, since they need gradients and
   therefore take the training path.

.. warning::

   Do not confuse this with :class:`FusedDilatedConvNormFunc` below. The
   module is the hookable node in the tree; the ``Func`` is the
   ``torch.autograd.Function`` implementing the fused kernel that the
   module ultimately calls.


FusedDilatedConvNormFunc
------------------------

.. autoclass:: FusedDilatedConvNormFunc
   :members: forward, backward
   :undoc-members:

A custom ``torch.autograd.Function`` that fuses the 3-tap dilated
depthwise convolution and per-example layer normalization. User code
interacts with it through :func:`fused_dilated_conv_norm`,
:class:`FusedDilatedConvNorm` and :class:`CheriBlock`.


Forward kernels
~~~~~~~~~~~~~~~

The forward pass runs three Triton kernels in sequence:

* ``_fwd_stats_kernel`` — runs over ``(N, num_chunks_of_L)``
  blocks. For each block it loads the three dilated taps, computes
  the convolved output and stores it into the output buffer ``y``,
  and atomically accumulates per-example ``sum`` and ``sq_sum`` of
  the convolved values into ``(N,)`` float32 buffers.
* ``_fwd_finalize_kernel`` — runs over ``(N,)``: turns the per-example
  ``sum``/``sq_sum`` into ``mean``/``rstd`` with the
  ``eps=1e-3`` numerical stability constant.
* ``_fwd_apply_kernel`` — runs over ``(N, num_chunks_of_L)``: loads
  ``y`` in-place, subtracts the mean, multiplies by ``rstd``, writes
  back the normalized value.

Layer-norm statistics are always accumulated and stored in fp32 even
when ``y`` is bf16, which keeps the normalization numerically robust
under autocast.


Backward kernels
~~~~~~~~~~~~~~~~

The backward pass also runs three Triton kernels in sequence and
**recomputes** the convolved output during the backward pass rather
than caching it from the forward — trading a small amount of FLOPS
for the activation memory of an extra ``(N, L, C)`` tensor:

* ``_bwd_stats_kernel`` — recomputes the conv output ``conv = x0*w0
  + x1*w1 + x2*w2``, stores it into a scratch buffer, and atomically
  accumulates two per-example reductions: ``sum_dy`` and
  ``sum_dy_xhat`` (where ``xhat = (conv - mean) * rstd``). These are
  the two scalars the layer-norm backward needs.
* ``_bwd_apply_kernel`` — uses ``sum_dy`` and ``sum_dy_xhat`` to
  compute ``d_conv = (rstd/count) * (count*dy - sum_dy - xhat *
  sum_dy_xhat)`` for every position, and accumulates the depthwise
  weight gradient ``dw`` by computing ``d_conv * x{0,1,2}`` and
  reducing along the length axis. Per-block ``dw`` partials are
  written into a ``(N * num_chunks, 3 * C)`` buffer; the final
  ``dw`` is the sum over all partials (handled outside the kernel).
* ``_bwd_dx_kernel`` — applies the dilated transpose convolution to
  ``d_conv`` to produce ``dx``: at each position ``p``, ``dx[p] =
  d_conv[p] * w1 + d_conv[p + d] * w0 + d_conv[p - d] * w2`` (the
  three terms each masked at the sequence ends).

The first call on a given ``(C, L)`` shape triggers Triton autotune,
and the user-visible *gradient* output from that very first call is
contaminated by atomic-add residue from the benchmarking trials.
Subsequent calls use the locked-in best config and agree with CPU
autograd at fp32 precision. The test suite warms up the kernel
before any gradient is read; training is unaffected in practice
because step 2 onward is clean.


Autotune configuration space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the forward stats kernel and the two backward kernels are
autotuned over the cartesian product of:

* ``num_warps``: 4, 8, 16
* ``num_stages``: 2, 3, 4, 5

with the autotune key set to ``(C, L)``. ``BLOCK_C`` is set to
``triton.next_power_of_2(C)`` per call (so it adapts to the
actual channel width); ``BLOCK_L`` is fixed at 64.

The inference megakernel autotunes its own normalization-plus-MLP
kernel over the same warp/stage grid, with an additional
``BLOCK_HK`` (hidden width tile) constraint applied via the
``prune_configs_by={'early_config_prune': ...}`` callback to keep
the hidden tile a divisor of ``expansion * n_filters``.


Inference megakernel
--------------------

When ``torch.is_grad_enabled()`` is ``False`` and the MLP hidden
width is a multiple of 16, the Cheri Block dispatches to a fused
inference megakernel that performs conv + norm + MLP + residual in
two GPU passes rather than four separate ops. The implementation
lives under the ``_fwd_inf_`` prefix in
``cherimoya/cheri.py``. It is not part of the public API, but the
behavior is observable:

* The bf16 cast of the linear weights is materialized at ``.eval()``
  time as three non-persistent buffers (``_w1_eval_bf16``,
  ``_w2_eval_bf16``, ``_w2_eval_native``) so it lives outside the
  compiled forward and survives ``torch.compile`` cudagraph
  replays. Calling ``.train(True)`` clears the cache;
  ``load_state_dict`` while in eval mode refreshes it via a post-hook.
* The buffers are registered with ``persistent=False`` and so are
  **not** part of the state dict.
* If the inference path is reached under ``no_grad`` *without* a
  prior ``.eval()`` call, the megakernel still fires correctly; it
  just recomputes the bf16 cast inline per call. See
  :doc:`/benchmarks` for the per-call cost.
* The path produces outputs that differ from the training Triton
  path by at most ~1e-5 max-abs at unit-scale fp32 outputs (~1e-2
  for bf16 input, ~1e-3 for fp16 input).
