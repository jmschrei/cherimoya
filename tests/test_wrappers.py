"""Tests for the model wrappers in ``cherimoya.wrappers``."""

import pytest
import torch

from numpy.testing import assert_array_almost_equal

from cherimoya import Cherimoya
from cherimoya.wrappers import ControlWrapper
from cherimoya.wrappers import ProfileWrapper
from cherimoya.wrappers import LogCountWrapper
from cherimoya.wrappers import ExpectedCountsWrapper


def _input_window_for(model):
	"""Compute a valid input window length for `model`."""
	# Output window must be > 0; trimming bytes are removed from each side.
	return 2 * model.trimming + 64


@pytest.fixture
def grouped_model():
	# One unstranded track plus one stranded (+, -) pair: the profile head
	# emits sum([1, 2]) = 3 channels while the count head emits len([1, 2]) = 2
	# predictions. This is the config that exercises the channel-vs-group
	# bookkeeping in ExpectedCountsWrapper. Seed first so the random weight
	# init is fixed and the regression values below are reproducible.
	torch.manual_seed(0)
	return Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		verbose=False, compile=False).eval()


@pytest.fixture
def control_model():
	torch.manual_seed(0)
	return Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		n_control_tracks=1, verbose=False, compile=False).eval()


# --------- ControlWrapper --------------------------------------------------

def test_control_wrapper_passthrough_no_controls(grouped_model):
	"""A model with no control tracks is forwarded through unchanged."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(0)
	X = torch.randn(4, 4, L)

	with torch.no_grad():
		ref_profile, ref_counts = grouped_model(X)
		profile, counts = ControlWrapper(grouped_model)(X)

	assert torch.equal(profile, ref_profile)
	assert torch.equal(counts, ref_counts)


def test_control_wrapper_synthesizes_zero_control(control_model):
	"""When the model expects a control track but none is passed, an all-zero
	track of the right shape is supplied — equivalent to passing zeros."""
	L = _input_window_for(control_model)
	torch.manual_seed(0)
	X = torch.randn(2, 4, L)
	X_ctl = torch.zeros(2, control_model.n_control_tracks, L)

	with torch.no_grad():
		ref_profile, ref_counts = control_model(X, X_ctl=X_ctl)
		profile, counts = ControlWrapper(control_model)(X)

	assert torch.equal(profile, ref_profile)
	assert torch.equal(counts, ref_counts)


def test_control_wrapper_forwards_given_control(control_model):
	"""An explicit control track is passed straight through to the model."""
	L = _input_window_for(control_model)
	torch.manual_seed(0)
	X = torch.randn(2, 4, L)
	X_ctl = torch.rand(2, 1, L)

	with torch.no_grad():
		ref_profile, ref_counts = control_model(X, X_ctl=X_ctl)
		profile, counts = ControlWrapper(control_model)(X, X_ctl=X_ctl)

	assert torch.equal(profile, ref_profile)
	assert torch.equal(counts, ref_counts)


def test_control_wrapper_matches_bpnetlite(control_model):
	"""The port is numerically identical to bpnet-lite's ControlWrapper."""
	bpnet = pytest.importorskip("bpnetlite.bpnet")

	L = _input_window_for(control_model)
	torch.manual_seed(0)
	X = torch.randn(2, 4, L)

	with torch.no_grad():
		ours_profile, ours_counts = ControlWrapper(control_model)(X)
		theirs_profile, theirs_counts = bpnet.ControlWrapper(control_model)(X)

	assert_array_almost_equal(ours_profile.numpy(), theirs_profile.numpy(), 6)
	assert_array_almost_equal(ours_counts.numpy(), theirs_counts.numpy(), 6)


def test_control_wrapper_composes_with_output_wrappers(control_model):
	"""The CLI pattern: an output wrapper layered on ControlWrapper runs a
	control-track model from the sequence alone."""
	L = _input_window_for(control_model)
	torch.manual_seed(0)
	X = torch.randn(2, 4, L)

	wrapped = ControlWrapper(control_model)
	with torch.no_grad():
		counts = LogCountWrapper(wrapped)(X)
		profile = ProfileWrapper(wrapped)(X)

	assert counts.shape == (2, 2)
	assert profile.shape == (2, 1)


# --------- ProfileWrapper --------------------------------------------------

def test_profile_wrapper_shape(grouped_model):
	"""The wrapper collapses the profile to one number per example."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(0)
	X = torch.randn(4, 4, L)

	with torch.no_grad():
		y_hat = ProfileWrapper(grouped_model)(X)

	assert y_hat.shape == (4, 1)


def test_profile_wrapper_matches_bpnetlite(grouped_model):
	"""The port is numerically identical to bpnet-lite's ProfileWrapper."""
	bpnet = pytest.importorskip("bpnetlite.bpnet")

	L = _input_window_for(grouped_model)
	torch.manual_seed(0)
	X = torch.randn(2, 4, L)

	with torch.no_grad():
		ours = ProfileWrapper(grouped_model)(X)
		theirs = bpnet.ProfileWrapper(grouped_model)(X)

	assert_array_almost_equal(ours.numpy(), theirs.numpy(), 6)


def test_profile_wrapper_regression(grouped_model):
	"""Regression on the profile attribution target for a fixed seed."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(1)
	X = torch.randn(2, 4, L)

	with torch.no_grad():
		y_hat = ProfileWrapper(grouped_model)(X)

	assert_array_almost_equal(y_hat.numpy(), [
		[0.00151387],
		[0.00151552]], 6)


def test_profile_wrapper_passes_control(control_model):
	"""The control track is forwarded through to the model."""
	L = _input_window_for(control_model)
	torch.manual_seed(0)
	X = torch.randn(3, 4, L)
	X_ctl = torch.rand(3, 1, L)

	with torch.no_grad():
		expected = ProfileWrapper(control_model)(X, X_ctl=X_ctl)

	assert expected.shape == (3, 1)
	assert torch.isfinite(expected).all()


def test_profile_wrapper_is_differentiable(grouped_model):
	"""Gradients flow to the input — the attribution use case."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(0)
	X = torch.randn(2, 4, L, requires_grad=True)

	ProfileWrapper(grouped_model)(X).sum().backward()

	assert X.grad is not None
	assert torch.isfinite(X.grad).all()


# --------- LogCountWrapper -------------------------------------------------

def test_logcount_wrapper_matches_count_head(grouped_model):
	"""The wrapper output is exactly the model's second return value."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(0)
	X = torch.randn(4, 4, L)

	with torch.no_grad():
		_, y_logcounts = grouped_model(X)
		y_hat = LogCountWrapper(grouped_model)(X)

	assert y_hat.shape == (4, 2)
	assert torch.equal(y_hat, y_logcounts)


def test_logcount_wrapper_regression(grouped_model):
	"""Regression on the log-count values for a fixed seed."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(1)
	X = torch.randn(2, 4, L)

	with torch.no_grad():
		y_hat = LogCountWrapper(grouped_model)(X)

	assert_array_almost_equal(y_hat.numpy(), [
		[-0.00048958, 0.00141992],
		[0.00084040, 0.00089380]], 6)


def test_logcount_wrapper_passes_control(control_model):
	"""The control track is forwarded through to the model."""
	L = _input_window_for(control_model)
	torch.manual_seed(0)
	X = torch.randn(3, 4, L)
	# Control tracks are read counts; the count head takes log(sum + 1), so
	# they must be non-negative.
	X_ctl = torch.rand(3, 1, L)

	with torch.no_grad():
		expected = control_model(X, X_ctl=X_ctl)[1]
		y_hat = LogCountWrapper(control_model)(X, X_ctl=X_ctl)

	assert torch.equal(y_hat, expected)


def test_logcount_wrapper_is_differentiable(grouped_model):
	"""Gradients flow to the input — the attribution use case."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(0)
	X = torch.randn(2, 4, L, requires_grad=True)

	LogCountWrapper(grouped_model)(X).sum().backward()

	assert X.grad is not None
	assert torch.isfinite(X.grad).all()


# --------- ExpectedCountsWrapper -------------------------------------------

def test_expected_counts_shape_matches_profile(grouped_model):
	"""Output has the profile head's shape: (batch, sum(groups), out_len)."""
	L = _input_window_for(grouped_model)
	out_L = L - 2 * grouped_model.trimming
	torch.manual_seed(0)
	X = torch.randn(4, 4, L)

	with torch.no_grad():
		y_hat = ExpectedCountsWrapper(grouped_model)(X)

	assert y_hat.shape == (4, 3, out_L)


def test_expected_counts_group_sum_equals_counts(grouped_model):
	"""Summing expected counts over a group's channels and positions
	recovers expm1(log_count) for that group — including both strands of
	the stranded pair."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(0)
	X = torch.randn(4, 4, L)

	with torch.no_grad():
		y_logits, y_logcounts = grouped_model(X)
		y_hat = ExpectedCountsWrapper(grouped_model)(X)

	expected_total = torch.expm1(y_logcounts)
	# Group 0 is the unstranded channel; group 1 is the stranded pair.
	group0 = y_hat[:, 0:1].sum(dim=(1, 2))
	group1 = y_hat[:, 1:3].sum(dim=(1, 2))

	assert_array_almost_equal(group0.numpy(), expected_total[:, 0].numpy(), 4)
	assert_array_almost_equal(group1.numpy(), expected_total[:, 1].numpy(), 4)


def test_expected_counts_softmax_is_joint_over_group(grouped_model):
	"""Within a group the distribution is joint: dividing the expected
	counts by the group's counts yields probabilities that sum to one across
	the group's channels and positions, not per-channel."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(2)
	X = torch.randn(3, 4, L)

	with torch.no_grad():
		_, y_logcounts = grouped_model(X)
		y_hat = ExpectedCountsWrapper(grouped_model)(X)

	counts = torch.expm1(y_logcounts)
	probs1 = y_hat[:, 1:3] / counts[:, 1, None, None]
	# Joint over the pair: total mass is one. Per-channel it would be two.
	assert_array_almost_equal(probs1.sum(dim=(1, 2)).numpy(),
		torch.ones(3).numpy(), 4)


def test_expected_counts_single_unstranded_group():
	"""The default single-group config reduces to a plain softmax-times-count
	distribution over positions."""
	torch.manual_seed(0)
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[1],
		verbose=False, compile=False).eval()
	L = _input_window_for(model)
	torch.manual_seed(0)
	X = torch.randn(2, 4, L)

	with torch.no_grad():
		y_logits, y_logcounts = model(X)
		y_hat = ExpectedCountsWrapper(model)(X)

	reference = torch.softmax(y_logits, dim=-1) * torch.expm1(y_logcounts)[:, :, None]
	assert_array_almost_equal(y_hat.numpy(), reference.numpy(), 4)


def test_expected_counts_regression(grouped_model):
	"""Regression on the expected counts at a few positions for a fixed
	seed."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(1)
	X = torch.randn(2, 4, L)

	with torch.no_grad():
		y_hat = ExpectedCountsWrapper(grouped_model)(X)

	# First three positions of each channel for the first example.
	assert_array_almost_equal(y_hat[0, :, :3].numpy(), [
		[-0.00000807, -0.00000754, -0.00000711],
		[0.00001016, 0.00001177, 0.00001103],
		[0.00001058, 0.00001080, 0.00001072]], 6)


def test_expected_counts_passes_control(control_model):
	"""The control track is forwarded through to the model."""
	L = _input_window_for(control_model)
	torch.manual_seed(0)
	X = torch.randn(3, 4, L)
	X_ctl = torch.rand(3, 1, L)

	with torch.no_grad():
		y_logits, y_logcounts = control_model(X, X_ctl=X_ctl)
		y_hat = ExpectedCountsWrapper(control_model)(X, X_ctl=X_ctl)

	expected_total = torch.expm1(y_logcounts)
	group1 = y_hat[:, 1:3].sum(dim=(1, 2))
	assert_array_almost_equal(group1.numpy(), expected_total[:, 1].numpy(), 4)


def test_expected_counts_is_differentiable(grouped_model):
	"""Gradients flow to the input."""
	L = _input_window_for(grouped_model)
	torch.manual_seed(0)
	X = torch.randn(2, 4, L, requires_grad=True)

	ExpectedCountsWrapper(grouped_model)(X).sum().backward()

	assert X.grad is not None
	assert torch.isfinite(X.grad).all()


# --------- Error handling --------------------------------------------------

def test_profile_wrapper_rejects_wrong_channel_count(grouped_model):
	"""A sequence that is not 4-channel one-hot fails in the input conv."""
	L = _input_window_for(grouped_model)
	X = torch.randn(2, 5, L)

	with pytest.raises(RuntimeError):
		ProfileWrapper(grouped_model)(X)


def test_logcount_wrapper_rejects_wrong_channel_count(grouped_model):
	"""A sequence that is not 4-channel one-hot fails in the input conv."""
	L = _input_window_for(grouped_model)
	X = torch.randn(2, 5, L)

	with pytest.raises(RuntimeError):
		LogCountWrapper(grouped_model)(X)


def test_expected_counts_rejects_wrong_channel_count(grouped_model):
	"""A sequence that is not 4-channel one-hot fails in the input conv."""
	L = _input_window_for(grouped_model)
	X = torch.randn(2, 5, L)

	with pytest.raises(RuntimeError):
		ExpectedCountsWrapper(grouped_model)(X)


def test_expected_counts_requires_signal_groups_attribute(grouped_model):
	"""ExpectedCountsWrapper reads ``model.signal_groups`` to split the
	profile head; wrapping a model without it raises AttributeError. This
	pins the contract that it wraps a Cherimoya directly, not another
	wrapper."""
	L = _input_window_for(grouped_model)
	X = torch.randn(2, 4, L)

	# Wrapping the LogCountWrapper (which has no signal_groups and returns
	# only counts) is a misuse and should surface clearly.
	wrapped = ExpectedCountsWrapper(LogCountWrapper(grouped_model))

	with pytest.raises(AttributeError):
		wrapped(X)


def test_expected_counts_missing_control_raises(control_model):
	"""A model built with control tracks needs the control tensor; omitting
	it fails in the profile conv where the channels no longer line up."""
	L = _input_window_for(control_model)
	X = torch.randn(2, 4, L)

	with pytest.raises(RuntimeError):
		ExpectedCountsWrapper(control_model)(X)


##
# Composition with ControlWrapper.
#
# `ControlWrapper`'s docstring calls it "the inner wrapper that
# ProfileWrapper, LogCountWrapper, or ExpectedCountsWrapper are layered
# on top of", and the attribute CLI builds exactly that stack. A wrapper
# that reads configuration off the model therefore has to look through
# it.
##


@pytest.fixture
def controlled_model():
	"""A grouped model that takes control tracks, so `ControlWrapper`
	actually has something to synthesize."""

	torch.manual_seed(0)
	return Cherimoya(n_filters=8, n_layers=2, signal_groups=[1, 2],
		n_control_tracks=2, verbose=False, compile=False).eval()


@pytest.mark.parametrize("wrapper", [ProfileWrapper, LogCountWrapper,
	ExpectedCountsWrapper])
def test_wrappers_compose_over_control_wrapper(wrapper, controlled_model):
	"""Every output wrapper must run when layered over ControlWrapper,
	which is the documented arrangement and the one the CLI builds, and
	must give the same answer as passing the zero control tracks it
	synthesizes explicitly.

	Running without raising is what the composition bug broke, but it is
	not enough on its own: a wrapper that looked through ControlWrapper
	to the wrong attribute could still return a well-shaped wrong
	number.
	"""

	X = torch.randn(2, 4, _input_window_for(controlled_model))
	X_ctl = torch.zeros(2, 2, X.shape[-1])

	with torch.no_grad():
		wrapped = wrapper(ControlWrapper(controlled_model))(X)
		direct = wrapper(controlled_model)(X, X_ctl=X_ctl)

	assert wrapped.shape == direct.shape
	assert_array_almost_equal(wrapped.numpy(), direct.numpy(), 5)


def test_expected_counts_group_structure_through_control_wrapper(
		controlled_model):
	"""The grouping the wrapper needs is the model's, not the
	ControlWrapper's, so each group's expected counts must still sum to
	that group's predicted count."""

	X = torch.randn(2, 4, _input_window_for(controlled_model))
	model = ExpectedCountsWrapper(ControlWrapper(controlled_model))

	with torch.no_grad():
		y = model(X)
		_, logcounts = ControlWrapper(controlled_model)(X)

	counts = torch.expm1(logcounts)
	offset = 0
	for i, g in enumerate(controlled_model.signal_groups):
		total = y[:, offset:offset + g].sum(dim=(1, 2))
		assert_array_almost_equal(total.numpy(), counts[:, i].numpy(), 4)
		offset += g


##
# Selecting one signal group.
#
# For a multi-group model `ProfileWrapper` otherwise softmaxes every group
# together and `LogCountWrapper` returns every group, so `group=` is the
# only way to attribute one modality.
##


def _reference_group_profile(logits):
	"""The ProfileWrapper calculation, written out on one group's logits."""

	logits = logits.reshape(logits.shape[0], -1)
	logits = logits - logits.mean(dim=-1, keepdim=True)
	return (logits * torch.softmax(logits, dim=-1)).sum(dim=-1, keepdim=True)


@pytest.mark.parametrize("group, channels", [(0, slice(0, 1)),
	(1, slice(1, 3))])
def test_profile_wrapper_group_uses_only_that_group(grouped_model, group,
		channels):
	"""The mean-centering and softmax run over the selected group's
	channels and positions only, including both strands of the stranded
	pair."""

	X = torch.randn(3, 4, _input_window_for(grouped_model))

	with torch.no_grad():
		y_logits, _ = grouped_model(X)
		y_hat = ProfileWrapper(grouped_model, group=group)(X)

	expected = _reference_group_profile(y_logits[:, channels])
	assert y_hat.shape == (3, 1)
	assert_array_almost_equal(y_hat.numpy(), expected.numpy(), 6)


def test_profile_wrapper_group_regression(grouped_model):
	"""Regression on the per-group profile target for a fixed seed."""

	torch.manual_seed(1)
	X = torch.randn(2, 4, _input_window_for(grouped_model))

	with torch.no_grad():
		y_hat = torch.cat([ProfileWrapper(grouped_model, group=g)(X)
			for g in range(2)], dim=-1)

	assert_array_almost_equal(y_hat.numpy(), [
		[0.00125393, 0.00164274],
		[0.00115233, 0.00169656]], 6)


def test_profile_wrapper_group_single_group_matches_default():
	"""With one group, selecting it is the same as selecting everything."""

	torch.manual_seed(0)
	model = Cherimoya(n_filters=8, n_layers=2, signal_groups=[2],
		verbose=False, compile=False).eval()
	X = torch.randn(2, 4, _input_window_for(model))

	with torch.no_grad():
		y_all = ProfileWrapper(model)(X)
		y_group = ProfileWrapper(model, group=0)(X)

	assert torch.equal(y_all, y_group)


@pytest.mark.parametrize("group", [0, 1])
def test_logcount_wrapper_group_is_that_column(grouped_model, group):
	"""The output is the selected group's log-count, kept two-dimensional."""

	X = torch.randn(3, 4, _input_window_for(grouped_model))

	with torch.no_grad():
		_, y_logcounts = grouped_model(X)
		y_hat = LogCountWrapper(grouped_model, group=group)(X)

	assert y_hat.shape == (3, 1)
	assert torch.equal(y_hat, y_logcounts[:, group:group+1])


@pytest.mark.parametrize("wrapper", [ProfileWrapper, LogCountWrapper])
def test_group_wrappers_compose_over_control_wrapper(wrapper,
		controlled_model):
	"""The group offsets are read through ControlWrapper, which is the
	stack the attribute CLI builds."""

	X = torch.randn(2, 4, _input_window_for(controlled_model))
	X_ctl = torch.zeros(2, 2, X.shape[-1])

	with torch.no_grad():
		wrapped = wrapper(ControlWrapper(controlled_model), group=1)(X)
		direct = wrapper(controlled_model, group=1)(X, X_ctl=X_ctl)

	assert_array_almost_equal(wrapped.numpy(), direct.numpy(), 5)


@pytest.mark.parametrize("wrapper", [ProfileWrapper, LogCountWrapper])
def test_group_wrappers_are_differentiable(wrapper, grouped_model):
	"""Gradients flow to the input through the group slice."""

	X = torch.randn(2, 4, _input_window_for(grouped_model),
		requires_grad=True)

	wrapper(grouped_model, group=1)(X).sum().backward()

	assert X.grad is not None
	assert torch.isfinite(X.grad).all()
	assert X.grad.abs().sum() > 0


def test_logcount_wrapper_group_matches_ism_target(grouped_model):
	"""Attributing ``LogCountWrapper(group=g)`` gives what tangermeme
	gives for the unselected wrapper with ``target=g``."""

	from tangermeme.saturation_mutagenesis import saturation_mutagenesis

	torch.manual_seed(0)
	L = _input_window_for(grouped_model)
	X = torch.zeros(2, 4, L)
	X[:, 0] = 1
	kwargs = dict(start=L // 2 - 5, end=L // 2 + 5, device='cpu',
		verbose=False)

	X_attr = saturation_mutagenesis(LogCountWrapper(grouped_model, group=1),
		X, **kwargs)
	X_attr_target = saturation_mutagenesis(LogCountWrapper(grouped_model),
		X, target=1, **kwargs)

	assert_array_almost_equal(X_attr.numpy(), X_attr_target.numpy(), 6)


@pytest.mark.parametrize("wrapper", [ProfileWrapper, LogCountWrapper])
@pytest.mark.parametrize("group", [2, -1, 1.0, True, "0"])
def test_group_wrappers_reject_bad_group(wrapper, group, grouped_model):
	"""An index outside ``range(len(signal_groups))``, or anything that is
	not an int, is rejected when the wrapper is built rather than at the
	first forward pass."""

	with pytest.raises(ValueError, match="group"):
		wrapper(grouped_model, group=group)
