"""Tests for the DeepLIFT rules in ``cherimoya.deep_lift_shap``."""

import warnings

import pytest
import torch
import torch.nn.functional as F

from numpy.testing import assert_array_almost_equal

from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.deep_lift_shap import integrated_gradients_op
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.utils import random_one_hot

from cherimoya import Cherimoya
from cherimoya import LogCountWrapper
from cherimoya import ProfileWrapper
from cherimoya.deep_lift_shap import attribution_ops
from cherimoya.deep_lift_shap import conv_norm_op
from cherimoya.cheri import CONV_NORM_EPS
from cherimoya.cheri import FusedDilatedConvNorm


torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


CHANNELS, LENGTH, DILATION = 8, 64, 2


class _Isolated(torch.nn.Module):
	"""proj -> one FusedDilatedConvNorm -> dense.

	The rule only matters when the normalization's statistics move between
	a sequence and its reference. A production model reduces over hundreds
	of thousands of elements per statistic and barely moves them, so one
	block over a short window is what makes the difference measurable --
	and is what these tests need in order to fail when the rule is wrong.
	"""

	def __init__(self):
		super().__init__()
		self.proj = torch.nn.Conv1d(4, CHANNELS, 1)
		self.conv = FusedDilatedConvNorm(CHANNELS, DILATION)
		self.dense = torch.nn.Linear(LENGTH * CHANNELS, 1)

	def forward(self, X):
		# `.contiguous()` is not optional. The Triton kernel is handed only
		# `x.stride(0)` and indexes the rest as `offs * C + offs_c`, so it
		# assumes an (N, L, C) tensor laid out contiguously; a bare
		# `transpose(1, 2)` view has strides (L*C, 1, L) and the kernel
		# reads the wrong elements, silently and only on CUDA. The model
		# itself calls `.contiguous()` at both of its transposes for the
		# same reason.
		h = self.conv(self.proj(X).transpose(1, 2).contiguous())
		return self.dense(h.reshape(h.shape[0], -1))


class _Decomposed(_Isolated):
	"""`_Isolated` with the fused op written as conv + `nn.LayerNorm`.

	The independent oracle. It reaches the closed-form rule through
	tangermeme's ordinary module dispatch rather than through
	`conv_norm_op`, so the two agreeing is evidence about the rule and not
	about the private helper both would otherwise share.
	"""

	def __init__(self):
		super().__init__()
		fused = self.conv
		self.conv = _PureConvNorm(fused.conv_weight, fused.dilation)


class _PureConvNorm(torch.nn.Module):
	def __init__(self, conv_weight, dilation):
		super().__init__()
		self.conv_weight = conv_weight
		self.dilation = dilation
		self.norm = torch.nn.LayerNorm([LENGTH, CHANNELS], eps=CONV_NORM_EPS,
			elementwise_affine=False)

	def forward(self, X):
		weight = self.conv_weight.t().unsqueeze(1)
		y = F.conv1d(X.transpose(1, 2), weight, padding=self.dilation,
			dilation=self.dilation, groups=X.shape[-1]).transpose(1, 2)
		return self.norm(y)


def _build(cls, dtype=torch.float32):
	torch.manual_seed(0)
	return cls().to(dtype).eval()


@pytest.fixture
def X():
	return random_one_hot((2, 4, LENGTH), random_state=0).type(torch.float32)


@pytest.fixture
def references(X):
	return dinucleotide_shuffle(X, n=2, random_state=0)


def _worst_delta(model, X, references, ops, batch_size=8, device='cpu'):
	"""The largest convergence delta over every example-reference pair."""

	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		deep_lift_shap(model, X, references=references, device=device,
			batch_size=batch_size, warning_threshold=1e-30,
			additional_nonlinear_ops=ops)

	message = str([w for w in caught
		if issubclass(w.category, RuntimeWarning)][0].message)
	body = message.split('[')[1].split(']')[0].replace('\n', '')
	return max(float(value) for value in body.split(','))


# --------- conv_norm_op vs the decomposed reference ------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conv_norm_op_matches_the_decomposed_model(X, references, dtype):
	# The rule exists to reach the same answer as rewriting the model into
	# `F.conv1d` + `nn.LayerNorm`, without giving up the fused kernel.
	fused, decomposed = _build(_Isolated, dtype), _build(_Decomposed, dtype)
	X, references = X.to(dtype), references.to(dtype)

	with torch.no_grad():
		assert_array_almost_equal(fused(X), decomposed(X), 6)

	X_attr = deep_lift_shap(fused, X, references=references, device='cpu',
		additional_nonlinear_ops={FusedDilatedConvNorm: conv_norm_op})
	X_attr_decomposed = deep_lift_shap(decomposed, X, references=references,
		device='cpu')

	assert_array_almost_equal(X_attr, X_attr_decomposed, 6)


# --------- convergence ------------------------------------------------------


def test_conv_norm_op_converges(X, references):
	delta = _worst_delta(_build(_Isolated), X, references,
		{FusedDilatedConvNorm: conv_norm_op})

	assert delta < 1e-5


def test_conv_norm_op_improves_on_registering_nothing(X, references):
	# Treating the fused op as linear is what happens with no rule, and it
	# is wrong by more than the prediction itself on a model this size.
	model = _build(_Isolated)

	unregistered = _worst_delta(model, X, references, None)
	registered = _worst_delta(model, X, references,
		{FusedDilatedConvNorm: conv_norm_op})

	with torch.no_grad():
		scale = float(model(X).abs().mean())

	assert unregistered > scale / 10
	assert registered < unregistered / 1e4


@pytest.mark.parametrize("K", [4, 8])
def test_integrated_gradients_also_converges(X, references, K):
	# The path integral is the other way to attribute this op without a
	# rewrite. It converges too, so the delta alone does not choose
	# between them -- the reasons to prefer the closed form are that it is
	# exact rather than quadrature, and that it costs one extra
	# convolution rather than K evaluations of the whole module.
	delta = _worst_delta(_build(_Isolated), X, references,
		{FusedDilatedConvNorm: integrated_gradients_op(K=K)})

	assert delta < 1e-5


def test_closed_form_and_integrated_gradients_differ_by_method(X, references):
	"""The gap between the two rules does not shrink with more nodes.

	A path integral and the closed-form multiplier are different
	quantities for a layer that couples positions, not one approximating
	the other. Quadrature error would fall as K rises; a difference in
	method does not, and that is what distinguishes them. Doubling K and
	getting the same gap is the assertion.
	"""

	model = _build(_Isolated)

	X_attr = deep_lift_shap(model, X, references=references, device='cpu',
		additional_nonlinear_ops={FusedDilatedConvNorm: conv_norm_op})
	gaps = []
	for K in (4, 8):
		X_attr_ig = deep_lift_shap(model, X, references=references,
			device='cpu', additional_nonlinear_ops={
				FusedDilatedConvNorm: integrated_gradients_op(K=K)})
		gaps.append(float((X_attr - X_attr_ig).abs().max()))

	scale = float(X_attr.abs().max())

	# Different enough to matter, and unchanged by doubling K.
	assert gaps[0] > scale / 100
	assert_array_almost_equal(gaps[0], gaps[1], 6)


# --------- the rest of the surface ------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 3, 8, 1000])
def test_conv_norm_op_is_batch_size_invariant(X, references, batch_size):
	model = _build(_Isolated)
	ops = {FusedDilatedConvNorm: conv_norm_op}

	X_attr = deep_lift_shap(model, X, references=references, device='cpu',
		batch_size=batch_size, additional_nonlinear_ops=ops)
	X_attr_ = deep_lift_shap(model, X, references=references, device='cpu',
		batch_size=8, additional_nonlinear_ops=ops)

	assert_array_almost_equal(X_attr, X_attr_, 6)


def test_conv_norm_op_regression(X, references):
	X_attr = deep_lift_shap(_build(_Isolated), X, references=references,
		device='cpu', additional_nonlinear_ops={
			FusedDilatedConvNorm: conv_norm_op})

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:, :, :4], [
		[[ 0.0000, -0.0000,  0.0000,  0.0106],
		 [-0.0000, -0.0000,  0.0173,  0.0000],
		 [-0.0000,  0.0000,  0.0000,  0.0000],
		 [-0.0000,  0.0132, -0.0000, -0.0000]],

		[[ 0.0000, -0.0136, -0.0000,  0.0067],
		 [ 0.0000, -0.0000,  0.0000, -0.0000],
		 [ 0.0000,  0.0000, -0.0096,  0.0000],
		 [ 0.0000,  0.0000, -0.0000, -0.0000]]], 4)


# --------- a real model ----------------------------------------------------


@pytest.fixture
def small_model():
	torch.manual_seed(0)
	return Cherimoya(n_filters=8, n_layers=2, verbose=False,
		compile=False).eval()


@pytest.mark.parametrize("wrapper", [LogCountWrapper, ProfileWrapper])
def test_attribution_ops_converges_on_a_real_model(small_model, wrapper):
	length = 2 * small_model.trimming + 64
	X = random_one_hot((2, 4, length), random_state=0).type(torch.float32)
	references = dinucleotide_shuffle(X, n=2, random_state=0)
	model = wrapper(small_model)

	with torch.no_grad():
		scale = float(model(X).abs().mean())

	delta = _worst_delta(model, X, references, attribution_ops())

	assert delta < scale / 1e4


@pytest.mark.parametrize("wrapper", [LogCountWrapper, ProfileWrapper])
def test_attribution_ops_converges_on_one_signal_group(wrapper):
	"""Selecting one group of a multi-group model with `group=` keeps the
	attributions summing to the change in that group's prediction."""

	torch.manual_seed(0)
	grouped_model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		verbose=False, compile=False).eval()
	length = 2 * grouped_model.trimming + 64
	X = random_one_hot((2, 4, length), random_state=0).type(torch.float32)
	references = dinucleotide_shuffle(X, n=2, random_state=0)
	model = wrapper(grouped_model, group=1)

	with torch.no_grad():
		scale = float(model(X).abs().mean())

	delta = _worst_delta(model, X, references, attribution_ops())

	assert delta < scale / 1e4


def test_attribution_ops_fixes_the_profile_head(small_model):
	"""The profile head is wrong by more than its own output untreated.

	`_ProfileLogitScaling` multiplies the logits by their own softmax,
	which is elementwise and shape-preserving but not linear. Left
	unregistered it is treated as linear, and on this model that leaves a
	convergence delta larger than the prediction being explained -- so the
	attributions carry no information about their own scale. The conv-norm
	rule alone does not reach it; both entries of `attribution_ops` are
	needed.
	"""

	length = 2 * small_model.trimming + 64
	X = random_one_hot((2, 4, length), random_state=0).type(torch.float32)
	references = dinucleotide_shuffle(X, n=2, random_state=0)
	model = ProfileWrapper(small_model)

	with torch.no_grad():
		scale = float(model(X).abs().mean())

	unregistered = _worst_delta(model, X, references, None)
	conv_only = _worst_delta(model, X, references,
		{FusedDilatedConvNorm: conv_norm_op})
	both = _worst_delta(model, X, references, attribution_ops())

	assert unregistered > scale
	assert conv_only > scale
	assert both < unregistered / 1e4


@pytest.mark.cuda
def test_conv_norm_op_matches_the_decomposed_model_on_cuda(X, references):
	"""The rule has to reach the same answer against the Triton kernel.

	`FusedDilatedConvNorm` dispatches to Triton on CUDA whenever gradients
	are enabled, which attribution always needs, so the forward and
	backward the rule is correcting there are a different implementation
	from the one on CPU. Agreement with the decomposed model is the
	assertion rather than the convergence delta, because it is the
	stronger of the two: a rule can converge and still be a different
	attribution.

	The convergence delta is deliberately not asserted here. On this
	fixture it lands at 1.0e-07 in most processes and at 3.3e-04 in a
	few, and `_Decomposed`, which launches no Triton kernel, does the
	same, so the variation comes from PyTorch's CUDA path rather than
	from the rule or the fused kernel. The attributions match the
	decomposed model in every process. So the delta is not a usable
	assertion on CUDA, while agreement is, and agreement is the stronger
	claim anyway.

	Takes several seconds, almost entirely CUDA context creation and the
	one-time Triton autotune; the work itself is the same size as its CPU
	counterpart above.
	"""

	X_attr = deep_lift_shap(_build(_Isolated), X, references=references,
		device='cuda', additional_nonlinear_ops={
			FusedDilatedConvNorm: conv_norm_op})
	X_attr_decomposed = deep_lift_shap(_build(_Decomposed), X,
		references=references, device='cuda')

	assert X_attr.shape == X.shape
	assert_array_almost_equal(X_attr.cpu(), X_attr_decomposed.cpu(), 6)


def test_conv_norm_op_declares_it_reads_only_the_input():
	# tangermeme clones only what a rule declares it reads; the rule
	# recomputes the convolution rather than reading the cached output.
	assert conv_norm_op._reads == ("input",)


def test_attribution_ops_returns_a_fresh_dict():
	first, second = attribution_ops(), attribution_ops()

	assert first == second
	assert first is not second

	first[torch.nn.ReLU] = None
	assert torch.nn.ReLU not in attribution_ops()
