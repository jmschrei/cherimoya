"""The first backward at a new shape must be correct.

`_bwd_apply_kernel` reads the convolution out of a scratch buffer and
writes the normalization gradient back over it, so it is not
idempotent. `triton.autotune` benchmarks a config by running the kernel
repeatedly, so every trial after the first read the previous trial's
output as though it were the convolution -- leaving the buffer garbage
by the time the real launch happens.

That made the user-visible gradient from the *first* call at any new
``(C, L)`` wrong. It matters more than "first call" suggests: attribution
is gradient-based and a fresh process attributing one shape hits exactly
this path, once, with nothing after it to notice.

This file lives apart from ``test_cheri.py`` on purpose. The autotune
cache is keyed on ``(C, L)``, so a shape used by any earlier test in the
same process is already tuned and the bug cannot reproduce; the shapes
here are chosen not to collide with any other test file.
"""

import pytest
import torch

from cherimoya.cheri import CheriBlock


# Shapes no other test uses, so the first call in this process is the
# one that triggers autotune.
FRESH_SHAPES = [(48, 176), (80, 208)]


def _grads(block, x):
	"""Forward, backward, and return every gradient as CPU tensors."""

	y = block(x)
	y.sum().backward()
	return {
		"x": x.grad.detach().float().cpu(),
		"conv": block.conv.conv_weight.grad.detach().float().cpu(),
		"linear1": block.linear1.weight.grad.detach().float().cpu(),
		"linear2": block.linear2.weight.grad.detach().float().cpu(),
	}


##


@pytest.mark.cuda
@pytest.mark.triton
@pytest.mark.parametrize("n_filters,length", FRESH_SHAPES)
def test_first_backward_at_a_new_shape_matches_cpu(n_filters, length):
	"""No warmup: the very first CUDA backward at this shape has to
	agree with CPU autograd.

	The tolerance is the ordinary fp32-with-TF32 budget for a single
	block, not a loosened one -- on the unfixed kernel the difference is
	order 1e-2, two orders above this.
	"""

	torch.manual_seed(0)
	cpu_block = CheriBlock(n_filters=n_filters, dilation=2)
	gpu_block = CheriBlock(n_filters=n_filters, dilation=2).cuda()
	gpu_block.load_state_dict(cpu_block.state_dict())

	x = torch.randn(2, length, n_filters)
	x_cpu = x.clone().requires_grad_()
	x_gpu = x.clone().cuda().requires_grad_()

	expected = _grads(cpu_block, x_cpu)
	got = _grads(gpu_block, x_gpu)

	for name in expected:
		diff = (expected[name] - got[name]).abs().max().item()
		assert diff < 5e-3, \
			"{} diverged on the first backward: max-abs-diff={:.3e}".format(
				name, diff)


@pytest.mark.cuda
@pytest.mark.triton
@pytest.mark.parametrize("n_filters,length", FRESH_SHAPES)
def test_second_backward_matches_the_first(n_filters, length):
	"""The control. The second call reuses the locked-in config and was
	always correct, so this passing while the test above fails is what
	localizes the bug to autotune rather than to the kernel's maths."""

	torch.manual_seed(0)
	block = CheriBlock(n_filters=n_filters, dilation=2).cuda()

	x = torch.randn(2, length, n_filters, device="cuda")

	first_x = x.clone().requires_grad_()
	first = _grads(block, first_x)
	block.zero_grad(set_to_none=True)

	second_x = x.clone().requires_grad_()
	second = _grads(block, second_x)

	for name in first:
		diff = (first[name] - second[name]).abs().max().item()
		assert diff < 1e-5, \
			"{} differs between calls: max-abs-diff={:.3e}".format(
				name, diff)
