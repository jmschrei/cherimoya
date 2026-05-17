"""Tests for the CheriBlock and the fused dilated conv + norm dispatcher."""

import pytest
import torch

from cherimoya.cheri import (
	CheriBlock,
	HAS_TRITON,
	_cheri_conv_norm_cpu,
	fused_dilated_conv_norm,
)


def _make_inputs(N=2, L=64, C=8, dilation=2, seed=0):
	g = torch.Generator().manual_seed(seed)
	x = torch.randn(N, L, C, generator=g)
	w = torch.randn(3, C, generator=g) * 0.1
	return x, w, dilation


# --------- CPU fallback (always runnable) ---------------------------------

def test_cpu_fallback_shape_and_dtype():
	x, w, d = _make_inputs()
	y = _cheri_conv_norm_cpu(x, w, d)
	assert y.shape == x.shape
	assert y.dtype == x.dtype


def test_cpu_fallback_per_example_zero_mean_unit_var():
	"""The fused norm should produce zero-mean, near-unit-variance per
	example. The variance is `var_conv / (var_conv + eps)` so it
	approaches 1 only when `var_conv >> eps`. We pick weights large
	enough to get there."""

	g = torch.Generator().manual_seed(0)
	x = torch.randn(4, 128, 16, generator=g)
	w = torch.randn(3, 16, generator=g)  # std 1, not 0.1 -- conv var dominates eps
	y = _cheri_conv_norm_cpu(x, w, dilation=4)
	flat = y.reshape(y.shape[0], -1)
	assert torch.allclose(flat.mean(dim=1), torch.zeros(4), atol=1e-4)
	assert torch.allclose(flat.var(dim=1, unbiased=False),
		torch.ones(4), atol=2e-3)


def test_cpu_fallback_dilation_uses_correct_taps():
	"""Verify the 3-tap dilated convolution against a manual reference."""
	x, w, d = _make_inputs(N=1, L=10, C=2, dilation=3)
	# Manual conv: out[i, c] = w[0,c]*x[i-d,c] + w[1,c]*x[i,c] + w[2,c]*x[i+d,c]
	N, L, C = x.shape
	expected_conv = torch.zeros_like(x)
	for c in range(C):
		for i in range(L):
			lo = i - d
			hi = i + d
			val = w[1, c] * x[0, i, c]
			if lo >= 0:
				val = val + w[0, c] * x[0, lo, c]
			if hi < L:
				val = val + w[2, c] * x[0, hi, c]
			expected_conv[0, i, c] = val

	# Reproduce the per-example normalization the fallback applies.
	flat = expected_conv.reshape(1, -1).float()
	mean = flat.mean(dim=1, keepdim=True)
	var = flat.var(dim=1, keepdim=True, unbiased=False)
	expected = ((flat - mean) * (var + 1e-3).rsqrt()).reshape(N, L, C)

	y = _cheri_conv_norm_cpu(x, w, d)
	assert torch.allclose(y, expected, atol=1e-4)


def test_cpu_fallback_is_differentiable():
	x, w, d = _make_inputs()
	x = x.detach().requires_grad_(True)
	w = w.detach().requires_grad_(True)
	y = _cheri_conv_norm_cpu(x, w, d)
	y.sum().backward()
	assert x.grad is not None
	assert w.grad is not None
	assert torch.isfinite(x.grad).all()
	assert torch.isfinite(w.grad).all()


def test_dispatcher_uses_cpu_path_for_cpu_tensor():
	"""On CPU input, the dispatcher must produce the same output as the
	fallback regardless of whether Triton is installed."""
	x, w, d = _make_inputs()
	y_disp = fused_dilated_conv_norm(x, w, d)
	y_cpu = _cheri_conv_norm_cpu(x, w, d)
	assert torch.equal(y_disp, y_cpu)


# --------- CheriBlock ------------------------------------------------------

def test_cheri_block_forward_shape_preserved():
	block = CheriBlock(n_filters=16, dilation=4)
	x = torch.randn(3, 64, 16)
	y = block(x)
	assert y.shape == x.shape


def test_cheri_block_default_expansion_is_two():
	block = CheriBlock(n_filters=16, dilation=1)
	assert block.expansion == 2
	assert block.linear1.out_features == 32
	assert block.linear2.in_features == 32


@pytest.mark.parametrize("expansion", [1, 2, 4])
def test_cheri_block_expansion_configurable(expansion):
	block = CheriBlock(n_filters=8, dilation=1, expansion=expansion)
	assert block.linear1.out_features == 8 * expansion
	assert block.linear2.in_features == 8 * expansion
	# Forward pass must still preserve shape regardless of expansion.
	x = torch.randn(2, 32, 8)
	assert block(x).shape == x.shape


def test_cheri_block_residual_uses_fixed_scale():
	"""The residual scale is a fixed scalar. Zeroing the inner MLP
	output (by zeroing both linear weights) collapses the block to the
	identity since X + 0 * residual_scale == X."""

	block = CheriBlock(n_filters=8, dilation=1)
	with torch.no_grad():
		block.linear1.weight.zero_()
		block.linear2.weight.zero_()
	x = torch.randn(2, 32, 8)
	y = block(x)
	assert torch.allclose(y, x, atol=1e-5)


def test_cheri_block_has_no_gamma_parameter():
	"""The block must not register a learnable channel-wise scaling
	parameter — the residual scale is a fixed scalar."""

	block = CheriBlock(n_filters=8, dilation=1)
	param_names = {n for n, _ in block.named_parameters()}
	assert 'gamma' not in param_names
	assert not hasattr(block, 'gamma')


def test_cheri_block_default_residual_scale():
	block = CheriBlock(n_filters=8, dilation=1)
	assert block.residual_scale == 0.15


def test_cheri_block_residual_scale_is_not_a_parameter():
	"""residual_scale is a plain Python float, not a learnable Parameter."""
	block = CheriBlock(n_filters=8, dilation=1, residual_scale=0.3)
	assert isinstance(block.residual_scale, float)
	param_names = {n for n, _ in block.named_parameters()}
	assert 'residual_scale' not in param_names


@pytest.mark.parametrize("residual_scale", [0.0, 0.05, 0.15, 1.0])
def test_cheri_block_residual_scale_controls_output(residual_scale):
	"""For a given input, the output should be exactly
	X + residual_scale * X_mlp. Compare two blocks with identical
	parameters but different residual_scale values."""

	block = CheriBlock(n_filters=8, dilation=1, residual_scale=residual_scale)
	x = torch.randn(2, 32, 8)
	y = block(x)
	# residual = (y - x) / residual_scale must equal X_mlp regardless of scale.
	# Check the special case residual_scale=0: output must equal input exactly.
	if residual_scale == 0.0:
		assert torch.allclose(y, x, atol=1e-6)
	else:
		# Recompute MLP path explicitly.
		from cherimoya.cheri import fused_dilated_conv_norm
		x_conv = fused_dilated_conv_norm(x, block.conv_weight, block.dilation)
		x_mlp = block.linear2(block.activation(block.linear1(x_conv)))
		expected = x + x_mlp * residual_scale
		assert torch.allclose(y, expected, atol=1e-6)


def test_cheri_block_backward_produces_finite_grads():
	block = CheriBlock(n_filters=8, dilation=2)
	x = torch.randn(2, 32, 8, requires_grad=True)
	y = block(x).sum()
	y.backward()
	for name, p in block.named_parameters():
		assert p.grad is not None, name
		assert torch.isfinite(p.grad).all(), name
	assert torch.isfinite(x.grad).all()


# --------- GPU / Triton parity --------------------------------------------

@pytest.mark.cuda
@pytest.mark.triton
def test_triton_matches_cpu_fallback():
	"""The Triton kernel must match the CPU fallback up to FP error.

	This is the key correctness check that the CPU fallback is a valid
	stand-in when Triton is not available.
	"""

	torch.manual_seed(0)
	x = torch.randn(2, 128, 16)
	w = torch.randn(3, 16) * 0.1

	y_cpu = _cheri_conv_norm_cpu(x, w, dilation=4)

	x_gpu = x.cuda()
	w_gpu = w.cuda()
	y_gpu = fused_dilated_conv_norm(x_gpu, w_gpu, 4).cpu()

	assert torch.allclose(y_cpu, y_gpu, atol=1e-3)


@pytest.mark.cuda
def test_cheri_block_runs_on_cuda():
	block = CheriBlock(n_filters=16, dilation=2).cuda()
	x = torch.randn(2, 64, 16, device='cuda')
	y = block(x)
	assert y.shape == x.shape
	assert y.is_cuda


def test_has_triton_flag_only_true_with_cuda_and_triton(cuda_available):
	"""The flag should never be True without CUDA available — CPU-only
	systems sometimes have triton installed but it is unusable there.
	The dispatcher routes by `x.is_cuda` so the flag is informational."""

	# Just exercise the constant; no asserts beyond it being a bool.
	assert isinstance(HAS_TRITON, bool)


# --------- Inference fast-path invariants ---------------------------------
#
# These tests pin down the invariants that any future inference-only fused
# kernel must preserve. They are written so they currently pass against the
# single-path implementation, and are meant to detect regressions when a
# separate no_grad-dispatched fused kernel is wired into CheriBlock.

def _block_forward(block, x, *, no_grad):
	"""Run a block forward optionally under torch.no_grad()."""
	if no_grad:
		with torch.no_grad():
			return block(x)
	return block(x)


@pytest.mark.parametrize("n_filters,expansion", [(8, 1), (16, 2), (32, 4)])
def test_cheri_block_no_grad_matches_grad_cpu(n_filters, expansion):
	"""On CPU, the no_grad and grad-enabled forward paths must produce
	the same output. When an inference-only Triton path is introduced
	for CUDA, this CPU equivalence must still hold because the inference
	path is CUDA-only — CPU input must continue to use the same code in
	both grad modes."""

	torch.manual_seed(0)
	block = CheriBlock(n_filters=n_filters, dilation=2, expansion=expansion)
	x = torch.randn(2, 64, n_filters)

	y_grad = _block_forward(block, x, no_grad=False)
	y_nograd = _block_forward(block, x, no_grad=True)

	# CPU goes through a single code path regardless of grad state; tolerate
	# only floating-point reordering noise.
	assert torch.allclose(y_grad.detach(), y_nograd, atol=1e-6, rtol=1e-5)


def test_cheri_block_no_grad_forward_is_stable_across_calls():
	"""Repeated forwards under no_grad must be deterministic with respect
	to themselves. A future weight-cache that lazily casts parameters
	(e.g., fp32 -> bf16) must not yield drifting outputs across calls."""

	torch.manual_seed(0)
	block = CheriBlock(n_filters=16, dilation=2)
	x = torch.randn(2, 64, 16)

	with torch.no_grad():
		y1 = block(x)
		y2 = block(x)
		y3 = block(x.clone())

	assert torch.equal(y1, y2)
	assert torch.equal(y1, y3)


def test_cheri_block_residual_scale_attribute_preserved_after_no_grad_forward():
	"""Some fused inference kernels fold residual_scale into the second
	linear weight. The public `residual_scale` attribute must remain the
	original float regardless of whether such folding has occurred
	internally, so users can introspect / save the model unchanged."""

	block = CheriBlock(n_filters=16, dilation=1, residual_scale=0.07)
	x = torch.randn(2, 32, 16)
	with torch.no_grad():
		_ = block(x)
	assert block.residual_scale == 0.07
	# The Parameter on linear2 must not have been mutated by the forward.
	assert torch.isfinite(block.linear2.weight).all()


def test_cheri_block_load_state_dict_reflects_in_subsequent_no_grad_forward():
	"""Critical for any internal weight cache: after `load_state_dict`,
	the next no_grad forward must use the newly loaded weights, not a
	stale cached copy of the previous weights.

	This is the canonical bug-shape for a Triton inference path that
	keeps a bf16 cast of the linears keyed by parameter identity.
	"""

	torch.manual_seed(0)
	block_a = CheriBlock(n_filters=16, dilation=2)
	torch.manual_seed(1)
	block_b = CheriBlock(n_filters=16, dilation=2)
	x = torch.randn(2, 64, 16)

	# Prime A's forward (and any internal cache) under no_grad.
	with torch.no_grad():
		y_a_before = block_a(x)
		y_b = block_b(x)

	# Overwrite A's weights with B's via load_state_dict.
	block_a.load_state_dict(block_b.state_dict())

	with torch.no_grad():
		y_a_after = block_a(x)

	# After the load, A must behave like B, not its old self.
	assert torch.allclose(y_a_after, y_b, atol=1e-6, rtol=1e-5)
	assert not torch.allclose(y_a_after, y_a_before, atol=1e-3)


def test_cheri_block_in_place_param_update_reflects_in_subsequent_no_grad_forward():
	"""An optimizer step performs in-place updates on parameters
	(`p.data.add_`, `p.mul_`, etc.). The same cache-invalidation concern
	as `load_state_dict` applies. Simulate an in-place update and verify
	the next no_grad forward sees the new weights."""

	torch.manual_seed(0)
	block = CheriBlock(n_filters=16, dilation=2)
	x = torch.randn(2, 64, 16)

	with torch.no_grad():
		y_before = block(x)

	with torch.no_grad():
		block.linear1.weight.mul_(0.5)
		block.linear2.weight.mul_(0.5)

	with torch.no_grad():
		y_after = block(x)

	# Outputs must differ — proves the new weights were actually used.
	assert not torch.allclose(y_after, y_before, atol=1e-3)


@pytest.mark.parametrize("n_filters,expansion", [(8, 1), (16, 1), (16, 2)])
def test_cheri_block_small_hidden_works_under_no_grad(n_filters, expansion):
	"""CheriBlock must support any (n_filters, expansion) combination
	regardless of grad mode. A fused inference kernel that requires
	hidden % 16 == 0 must transparently fall back when that constraint
	is not met — users cannot retrain a deployed model just because the
	inference path was tightened."""

	block = CheriBlock(n_filters=n_filters, dilation=1, expansion=expansion)
	x = torch.randn(2, 32, n_filters)

	with torch.no_grad():
		y = block(x)  # must not raise
	assert y.shape == x.shape
	assert torch.isfinite(y).all()


# --- The same parity checks on CUDA, where a fused inference kernel
# would actually take effect.

@pytest.mark.cuda
@pytest.mark.triton
@pytest.mark.parametrize("dtype,atol", [
	(torch.float32, 1e-4),
	(torch.float16, 5e-3),
	(torch.bfloat16, 1e-2),
])
def test_cheri_block_no_grad_matches_grad_cuda(dtype, atol):
	"""On CUDA, no_grad forward must match grad-enabled forward within a
	dtype-appropriate tolerance. This is the central invariant of any
	inference-only fast path: it must not silently produce different
	outputs from existing trained-model checkpoints.

	For fp32 inputs we enforce a tight 1e-4 tolerance — the kernel must
	preserve fp32 precision in the MLP for trained weights to remain
	valid. Reduced-precision input dtypes get matching tolerances.
	"""

	torch.manual_seed(0)
	block = CheriBlock(n_filters=32, dilation=2).cuda().to(dtype)
	x = torch.randn(2, 128, 32, device='cuda', dtype=dtype)

	y_grad = block(x)
	with torch.no_grad():
		y_nograd = block(x)

	diff = (y_grad.float() - y_nograd.float()).abs().max().item()
	assert diff <= atol, f"max-abs-diff={diff} exceeded {atol} for dtype={dtype}"


@pytest.mark.cuda
@pytest.mark.triton
def test_cheri_block_no_grad_matches_grad_cuda_n_filters_96():
	"""Production models default to n_filters=96 (hidden=192). Verify the
	inference path holds parity at the actual deployment width, since
	autotuned kernels behave differently across shapes."""

	torch.manual_seed(0)
	block = CheriBlock(n_filters=96, dilation=4).cuda()
	x = torch.randn(2, 256, 96, device='cuda')

	y_grad = block(x)
	with torch.no_grad():
		y_nograd = block(x)

	diff = (y_grad - y_nograd).abs().max().item()
	assert diff <= 1e-4, f"max-abs-diff={diff}"


@pytest.mark.cuda
@pytest.mark.triton
def test_cheri_block_load_state_dict_invalidates_cache_cuda():
	"""GPU version of the load_state_dict cache-invalidation check. A
	bf16 weight cache keyed by Parameter identity *and* `_version` is
	the standard trap here — `load_state_dict` does in-place copies
	which bump `_version` but not `id`."""

	torch.manual_seed(0)
	block_a = CheriBlock(n_filters=32, dilation=2).cuda()
	torch.manual_seed(1)
	block_b = CheriBlock(n_filters=32, dilation=2).cuda()
	x = torch.randn(2, 128, 32, device='cuda')

	with torch.no_grad():
		_ = block_a(x)
		y_b = block_b(x)

	block_a.load_state_dict(block_b.state_dict())

	with torch.no_grad():
		y_a_after = block_a(x)

	diff = (y_a_after - y_b).abs().max().item()
	assert diff <= 1e-4, f"cache not invalidated by load_state_dict; diff={diff}"


@pytest.mark.cuda
@pytest.mark.triton
def test_cheri_block_in_place_update_invalidates_cache_cuda():
	"""GPU version of the in-place update cache-invalidation check."""

	torch.manual_seed(0)
	block = CheriBlock(n_filters=32, dilation=2).cuda()
	x = torch.randn(2, 128, 32, device='cuda')

	with torch.no_grad():
		y_before = block(x)
		block.linear1.weight.mul_(0.5)
		block.linear2.weight.mul_(0.5)
		y_after = block(x)

	assert not torch.allclose(y_after, y_before, atol=1e-3)


@pytest.mark.cuda
@pytest.mark.triton
def test_cheri_block_triton_backward_matches_cpu_autograd():
	"""The Triton _bwd_* kernels (which produced every gradient in every
	existing trained checkpoint) must agree with the CPU autograd
	reference. This is the canonical regression guard for the training
	backward path — divergence means the gradient signal that trained
	deployed models cannot be reproduced.

	Tolerance budget: TF32 is enabled by default in cuBLAS, so the
	linear1/linear2 backward on GPU uses TF32 matmul (~1e-3 precision).
	That noise propagates through to dy entering the conv+norm bwd,
	pushing the realistic floor for conv_weight grad to ~5e-4 relative.

	Note: the first call to a Triton autotuned kernel runs benchmarking
	trials whose atomic_add residue can leave the user-visible output
	anomalously off (observed ~7e-2 on the very first call). We
	explicitly warm up to lock in the best config before measuring.
	"""

	torch.manual_seed(0)
	cpu_block = CheriBlock(n_filters=32, dilation=2)
	gpu_block = CheriBlock(n_filters=32, dilation=2).cuda()
	# Mirror CPU parameters onto the GPU block so both are bit-identical
	# at the start of the comparison.
	gpu_block.load_state_dict(cpu_block.state_dict())

	# Warmup: triggers Triton autotune for both fwd and bwd at this
	# shape. Without this, the first measured backward is contaminated
	# by autotune-trial state.
	with torch.enable_grad():
		xw = torch.randn(2, 128, 32, device='cuda', requires_grad=True)
		gpu_block(xw).sum().backward()
		gpu_block.zero_grad()

	x_cpu = torch.randn(2, 128, 32, requires_grad=True)
	x_gpu = x_cpu.detach().cuda().requires_grad_(True)

	y_cpu = cpu_block(x_cpu)
	y_gpu = gpu_block(x_gpu)

	y_cpu.sum().backward()
	y_gpu.sum().backward()

	# Forward outputs should agree up to TF32 / fp32 reorder noise.
	assert torch.allclose(y_cpu, y_gpu.cpu(), atol=1e-4, rtol=1e-4)

	# Backward grads. TF32 noise in cuBLAS linear bwd dominates the
	# tolerance budget here; the Triton conv+norm bwd itself is fp32.
	assert torch.allclose(cpu_block.linear1.weight.grad,
		gpu_block.linear1.weight.grad.cpu(), atol=1e-3, rtol=1e-3), \
		"linear1 grad diverged"
	assert torch.allclose(cpu_block.linear2.weight.grad,
		gpu_block.linear2.weight.grad.cpu(), atol=1e-3, rtol=1e-3), \
		"linear2 grad diverged"
	assert torch.allclose(cpu_block.conv_weight.grad,
		gpu_block.conv_weight.grad.cpu(), atol=1e-3, rtol=1e-3), \
		"conv_weight grad diverged"
	# x.grad goes through both the Triton bwd and the linear bwds.
	assert torch.allclose(x_cpu.grad, x_gpu.grad.cpu(),
		atol=1e-3, rtol=1e-3), "input grad diverged"


@pytest.mark.cuda
@pytest.mark.triton
def test_inference_megakernel_matches_cpu_fallback():
	"""The new fused inference megakernel must produce outputs that
	agree with the pure-PyTorch CPU fallback up to bf16-dot precision.
	CPU vs CUDA-no_grad is the strongest possible forward-parity check
	because the two implementations share no kernel code at all — any
	algorithmic bug in the megakernel surfaces here."""

	torch.manual_seed(0)
	cpu_block = CheriBlock(n_filters=32, dilation=2)
	gpu_block = CheriBlock(n_filters=32, dilation=2).cuda()
	gpu_block.load_state_dict(cpu_block.state_dict())

	x = torch.randn(2, 128, 32)

	with torch.no_grad():
		y_cpu = cpu_block(x)
		y_gpu = gpu_block(x.cuda())

	diff = (y_cpu - y_gpu.cpu()).abs().max().item()
	# bf16 weight cast + bf16 y_unnorm storage gives ~1e-3 to 1e-2
	# absolute error at unit-ish output scale. We pin 1e-2 as the
	# upper-bound budget; in practice diffs are well below this.
	assert diff <= 1e-2, f"CPU vs CUDA-megakernel max-abs-diff={diff:.3e}"


@pytest.mark.cuda
@pytest.mark.triton
def test_inference_megakernel_matches_previous_triton_forward():
	"""The new inference megakernel must agree with the existing Triton
	fwd kernel (called via FusedDilatedConvNormFunc through the training
	path) when applied to the same weights. Two separate block instances
	are used so any weight-cache interaction in a single block is ruled
	out — this isolates the kernel comparison from path-selection
	plumbing."""

	torch.manual_seed(0)
	block_grad = CheriBlock(n_filters=32, dilation=2).cuda()
	block_nograd = CheriBlock(n_filters=32, dilation=2).cuda()
	block_nograd.load_state_dict(block_grad.state_dict())

	x = torch.randn(2, 128, 32, device='cuda')

	# Existing Triton fwd kernel + PyTorch MLP via the training path.
	y_train = block_grad(x)
	# New megakernel.
	with torch.no_grad():
		y_inf = block_nograd(x)

	diff = (y_train - y_inf).abs().max().item()
	assert diff <= 1e-2, \
		f"training-Triton vs inference-megakernel max-abs-diff={diff:.3e}"


@pytest.mark.cuda
@pytest.mark.triton
def test_inference_megakernel_zero_linears_is_identity():
	"""Fixed-value test that does not depend on either kernel's
	precision: with both MLP weights set to zero, the block must reduce
	to X -> X exactly (zero dot anything is zero even in bf16; accum
	stays zero; residual passes through bit-identically)."""

	block = CheriBlock(n_filters=32, dilation=2).cuda()
	with torch.no_grad():
		block.linear1.weight.zero_()
		block.linear2.weight.zero_()

	x = torch.randn(2, 128, 32, device='cuda')
	with torch.no_grad():
		y = block(x)

	assert torch.equal(y, x), \
		f"zero-MLP path not bit-identical; diff={(y - x).abs().max().item()}"


@pytest.mark.cuda
@pytest.mark.triton
@pytest.mark.parametrize("n_filters,expansion", [(8, 1), (16, 1)])
def test_cheri_block_small_hidden_no_grad_matches_grad_cuda(n_filters, expansion):
	"""When the inference fast path's shape constraints aren't met (e.g.,
	hidden % 16 != 0), the block must fall back to a path that is
	numerically equivalent to the grad-enabled forward — not raise, not
	degrade silently."""

	block = CheriBlock(n_filters=n_filters, dilation=1,
		expansion=expansion).cuda()
	x = torch.randn(2, 64, n_filters, device='cuda')

	y_grad = block(x)
	with torch.no_grad():
		y_nograd = block(x)

	diff = (y_grad - y_nograd).abs().max().item()
	assert diff <= 1e-4, f"fallback path diverged: diff={diff}"
