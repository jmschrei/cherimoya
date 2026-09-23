Benchmarks
==========

This page reports measured timing and numerical agreement for the
forward paths of the Cheri Block across batch sizes, input dtypes, and
eval mode. See :doc:`architecture` for the description of each path.

Setup
-----

All numbers below were measured on a single NVIDIA H200 with PyTorch
2.12 and Triton 3.7. Each measurement uses a warmup phase to lock in
Triton autotune configurations before the timed iterations begin;
reported numbers are the median over 50-100 iterations.

These numbers were produced by ``bench_kernels.py``, a local
development script that lives in the repository root but is not tracked
in git or shipped with the package.

Forward timing
--------------

Per-call latency (ms) for a single :class:`~cherimoya.CheriBlock` at
``L=1024, C=128, dilation=4``. ``training-fwd`` is the grad-enabled
training path (eager PyTorch MLP on top of the fused conv+norm Triton
kernel). The two right-most columns dispatch the inference megakernel;
the right-most one shows what happens when the model is left in
training mode but called under ``torch.no_grad()``.

.. list-table::
   :header-rows: 1
   :widths: 10 10 22 28 30

   * - Batch
     - dtype
     - training-fwd
     - megakernel + ``.eval()``
     - megakernel, no ``.eval()``
   * - N=16
     - fp32
     - 0.101
     - **0.065**
     - 0.080 (+24%)
   * -
     - bf16
     - 0.104
     - **0.062**
     - 0.068 (+9%)
   * -
     - fp16
     - 0.102
     - **0.062**
     - 0.068 (+10%)
   * - N=64
     - fp32
     - 0.199
     - **0.074**
     - 0.081 (+10%)
   * -
     - bf16
     - 0.108
     - **0.062**
     - 0.069 (+10%)
   * -
     - fp16
     - 0.107
     - **0.062**
     - 0.068 (+10%)
   * - N=512
     - fp32
     - 1.337
     - **0.498**
     - 0.505 (+2%)
   * -
     - bf16
     - 0.707
     - **0.347**
     - 0.349 (+1%)
   * -
     - fp16
     - 0.706
     - **0.347**
     - 0.353 (+2%)

The megakernel is dispatched automatically whenever
``torch.is_grad_enabled()`` is ``False`` and the MLP hidden width
(``expansion * n_filters``) is a multiple of 16. To hit the *fast*
megakernel path, also call ``.eval()`` on the model: this
materializes the bf16 weight cast as a non-persistent buffer once,
outside the compiled forward, so the cast is reused across calls
instead of being recomputed every call. At small batch (N≤64) the
no-eval inline-cast path adds ~9-24%; at production batch (N=512)
the overhead is ~1-2% because the per-call cast cost is a fixed
~15 μs while the megakernel runtime scales with the batch. Training
is unaffected by the eval cache — it only fires under no_grad.

All paths agree on the fp32 model output to ~1e-5 max-abs at
unit-scale outputs, so existing trained checkpoints produce
numerically equivalent predictions through training-fwd and the
megakernel paths. Running the megakernel with bf16 or fp16 inputs
(after ``model.to(dtype)``) drifts to ~3e-2 / ~4e-3 max-abs
respectively, dominated by the reduced-precision MLP dot. The
detailed breakdown of which path pairs agree to fp32 precision vs.
the bf16 weight cast is in :doc:`architecture`.


Caveats
-------

* The numbers above are for *forward* passes only. Training step time
  is dominated by the backward pass and the optimizer step, neither of
  which the megakernel touches.
* H200-specific. On older GPUs (A100, V100) the absolute numbers
  change, but the relative ordering CPU ≫ train-fwd > inference-fwd
  holds.
* CPU inference is comfortable for development and one-off evaluation
  on a laptop; only training and high-throughput inference benefit
  from a GPU.
* The autotune step on the first call to any Triton kernel takes
  significant wall time. The warmup pass in ``bench_kernels.py``
  exists specifically to amortize this. If you call the model once
  and observe a slow first call, that is autotune — subsequent calls
  use the cached configuration. The first call is slower but not less
  accurate: the backward kernel declares ``restore_value`` for the
  buffer it overwrites, so autotune's benchmark trials cannot corrupt
  the gradient it returns.
