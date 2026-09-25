"""Tests for the EMA helper."""

import pytest
import torch
import torch.nn as nn

from cherimoya import EMA


class _Tiny(nn.Module):
	def __init__(self):
		super().__init__()
		self.lin = nn.Linear(4, 4)
		self.scalar = nn.Parameter(torch.tensor(0.0))


def _flatten(model):
	return torch.cat([p.detach().flatten() for _, p in model.named_parameters()])


def test_shadow_initialized_from_model_weights():
	model = _Tiny()
	ema = EMA(model, decay=0.9)
	for name, p in model.named_parameters():
		assert name in ema.shadow
		assert torch.equal(ema.shadow[name], p.detach())


def test_update_blends_toward_current_weights():
	"""After many updates with the model held fixed, shadow should converge
	to the current model weights regardless of starting state."""

	model = _Tiny()
	ema = EMA(model, decay=0.5)

	# Start with shadow at zero so the difference is obvious.
	for k in ema.shadow:
		ema.shadow[k].zero_()

	for _ in range(50):
		ema.update(model)

	for name, p in model.named_parameters():
		assert torch.allclose(ema.shadow[name], p.detach(), atol=1e-6)


def test_update_obeys_decay_factor():
	"""One update with decay=d should produce shadow = d*shadow + (1-d)*p."""
	model = _Tiny()
	ema = EMA(model, decay=0.7)

	# Capture initial shadow (= initial weights).
	old_shadow = {k: v.clone() for k, v in ema.shadow.items()}

	# Mutate the model so update has something to blend toward.
	with torch.no_grad():
		for p in model.parameters():
			p.add_(torch.ones_like(p))

	ema.update(model)

	for name, p in model.named_parameters():
		expected = 0.7 * old_shadow[name] + 0.3 * p.detach()
		assert torch.allclose(ema.shadow[name], expected, atol=1e-6)


def test_apply_shadow_then_restore_is_identity():
	model = _Tiny()
	ema = EMA(model)

	# Mutate shadow so apply_shadow is observable.
	for k in ema.shadow:
		ema.shadow[k].fill_(7.0)

	original = _flatten(model)
	ema.apply_shadow(model)
	assert (_flatten(model) == 7.0).all()

	ema.restore(model)
	assert torch.equal(_flatten(model), original)
	# After restore the backup is cleared so apply_shadow can be called again.
	assert ema._backup == {}


def test_apply_shadow_twice_without_restore_raises():
	model = _Tiny()
	ema = EMA(model)
	ema.apply_shadow(model)
	with pytest.raises(AssertionError):
		ema.apply_shadow(model)


def test_non_floating_buffers_are_not_tracked():
	class WithBuffer(nn.Module):
		def __init__(self):
			super().__init__()
			self.lin = nn.Linear(2, 2)
			self.register_buffer("counter", torch.zeros(1, dtype=torch.long))

	model = WithBuffer()
	ema = EMA(model)
	assert all(not k.endswith("counter") for k in ema.shadow)


##
# Interaction with CheriBlock's eval-time weight cache.
#
# `CheriBlock.train(False)` materializes bf16 casts of the MLP weights
# into non-persistent buffers, which the inference megakernel reads.
# EMA swaps parameters in place, so a swap performed while the model is
# already in eval mode has to be visible to that cache -- otherwise the
# depthwise convolution runs on EMA weights and the MLP runs on the
# pre-EMA ones.
##


def test_apply_shadow_bumps_the_parameter_version():
	"""The cache detects staleness through the parameter version
	counter, and `Tensor.data` is specifically the attribute that
	bypasses it. Pin that EMA does not write through `.data`."""

	model = _Tiny()
	ema = EMA(model, decay=0.999)

	with torch.no_grad():
		model.lin.weight.add_(torch.randn_like(model.lin.weight))

	# Captured after the mutation above, so only `apply_shadow` can be
	# what moves it.
	before = model.lin.weight._version
	ema.apply_shadow(model)

	assert model.lin.weight._version > before


def test_restore_bumps_the_parameter_version():
	model = _Tiny()
	ema = EMA(model, decay=0.999)
	ema.apply_shadow(model)
	before = model.lin.weight._version
	ema.restore(model)

	assert model.lin.weight._version > before


def test_apply_shadow_is_visible_to_the_cheri_eval_cache():
	"""The end-to-end invariant: after `eval()` then `apply_shadow`, the
	weights the block hands the kernel are the EMA weights.

	This is the documented usage pattern, and on `main` the cached bf16
	casts still held the pre-EMA weights, so the MLP and the depthwise
	convolution disagreed about which snapshot they were running.
	"""

	from cherimoya import CheriBlock

	torch.manual_seed(0)
	block = CheriBlock(n_filters=16, dilation=1)
	ema = EMA(block, decay=0.999)

	# "Train" the live weights away from the shadow.
	with torch.no_grad():
		for p in block.parameters():
			p.add_(torch.randn_like(p))

	block.eval()               # cache built from the live weights
	ema.apply_shadow(block)    # ... which are now the wrong ones

	w1, w2 = block._cast_weights(torch.zeros(1, 1, 16))

	assert torch.allclose(w1.float(), block.linear1.weight.float(),
		atol=1e-2), "MLP weight handed to the kernel is not the EMA weight"
	assert torch.allclose(
		w2.float(),
		(block.linear2.weight * block.residual_scale).float(),
		atol=1e-2)


def test_restore_is_visible_to_the_cheri_eval_cache():
	"""The same for the swap back, which is what `fit` does before
	resuming training.

	Ordered so that `restore` is the only thing that can invalidate the
	cache: the shadow is applied *before* `eval()`, so the cache is
	built from the shadow weights and `restore` is what makes it stale.
	"""

	from cherimoya import CheriBlock

	torch.manual_seed(0)
	block = CheriBlock(n_filters=16, dilation=1)
	ema = EMA(block, decay=0.999)

	with torch.no_grad():
		for p in block.parameters():
			p.add_(torch.randn_like(p))

	ema.apply_shadow(block)
	block.eval()               # cache built from the shadow weights
	ema.restore(block)         # ... which are no longer what is loaded

	w1, _ = block._cast_weights(torch.zeros(1, 1, 16))

	assert torch.allclose(w1.float(), block.linear1.weight.float(),
		atol=1e-2)


def test_eval_cache_survives_when_nothing_mutates():
	"""The fast path must still be taken in the ordinary case, or the
	fix would silently cost the cast on every call."""

	from cherimoya import CheriBlock

	torch.manual_seed(0)
	block = CheriBlock(n_filters=16, dilation=1).eval()

	w1, _ = block._cast_weights(torch.zeros(1, 1, 16))

	assert w1 is block._w1_eval_bf16
