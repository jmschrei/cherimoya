"""Tests for the mixture loss."""

import pytest
import torch

from bpnetlite.losses import MNLLLoss

from cherimoya.losses import _mixture_loss


def _toy_inputs(n=2, n_outputs=1, length=8, n_count_outputs=1, seed=0):
	g = torch.Generator().manual_seed(seed)
	y = torch.randint(0, 5, (n, n_outputs, length), generator=g).float()
	y_hat_logits = torch.randn(n, n_outputs, length, generator=g)
	y_hat_logcounts = torch.randn(n, n_count_outputs, generator=g)
	return y, y_hat_logits, y_hat_logcounts


def _legacy_single_output_profile_loss(y, logits, labels=None):
	"""The per-channel profile formula from before signal groups
	(per-channel ``log_softmax`` over length, then per-example MNLL, then
	mean over examples), kept as a bitwise reference for the guarantee that
	models with single-channel groups train exactly as they did."""

	log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
	if labels is not None:
		mnll = MNLLLoss(log_probs[labels == 1], y[labels == 1])
	else:
		mnll = MNLLLoss(log_probs, y)
	return mnll.mean(dim=0)


def test_mixture_loss_returns_per_track_vectors():
	y, logits, logcounts = _toy_inputs()
	profile_loss, count_loss = _mixture_loss(y, logits, logcounts)
	assert profile_loss.shape == (1,)
	assert count_loss.shape == (1,)
	assert torch.isfinite(profile_loss).all()
	assert torch.isfinite(count_loss).all()


def test_mixture_loss_multi_track_shapes():
	y, logits, logcounts = _toy_inputs(n_outputs=3, n_count_outputs=3)
	profile_loss, count_loss = _mixture_loss(y, logits, logcounts)
	assert profile_loss.shape == (3,)
	assert count_loss.shape == (3,)


def test_mixture_loss_count_loss_zero_when_perfect_predictions():
	y = torch.full((1, 1, 4), 2.0)
	logits = torch.zeros(1, 1, 4)
	# log1p(sum(y)) where sum(y) = 8 -> log(9) ≈ 2.197
	true_logcounts = torch.tensor([[torch.log(torch.tensor(9.0))]])
	_, count_loss = _mixture_loss(y, logits, true_logcounts)
	assert torch.allclose(count_loss, torch.tensor(0.0), atol=1e-5)


def test_mixture_loss_labels_filter_excludes_negatives():
	"""When labels are provided, the profile loss is only computed over
	the labeled-1 examples, but the count loss runs on all examples."""

	y, logits, logcounts = _toy_inputs(n=4)
	labels = torch.tensor([1, 0, 1, 0])

	# All-positive subset should produce the same profile loss as the
	# full call when labels=None — we provide that as a baseline.
	y_pos = y[labels == 1]
	logits_pos = logits[labels == 1]
	logcounts_pos = logcounts[labels == 1]

	prof_with_labels, _ = _mixture_loss(y, logits, logcounts, labels=labels)
	prof_without_labels, _ = _mixture_loss(y_pos, logits_pos, logcounts_pos)

	assert torch.allclose(prof_with_labels, prof_without_labels, atol=1e-5)


def test_mixture_loss_count_loss_uses_all_examples_with_labels():
	"""The count loss should not be filtered by `labels`."""
	y, logits, logcounts = _toy_inputs(n=4)
	labels = torch.tensor([1, 0, 1, 0])

	_, count_with_labels = _mixture_loss(y, logits, logcounts, labels=labels)
	_, count_no_labels = _mixture_loss(y, logits, logcounts)
	assert torch.allclose(count_with_labels, count_no_labels, atol=1e-5)


def test_mixture_loss_is_differentiable():
	y, logits, logcounts = _toy_inputs()
	logits = logits.detach().requires_grad_(True)
	logcounts = logcounts.detach().requires_grad_(True)
	profile_loss, count_loss = _mixture_loss(y, logits, logcounts)
	(profile_loss + count_loss).backward()
	assert logits.grad is not None
	assert logcounts.grad is not None
	assert torch.isfinite(logits.grad).all()
	assert torch.isfinite(logcounts.grad).all()


# --------- Per-group count pooling ----------------------------------------

def test_mixture_loss_signal_groups_pool_counts_per_group():
	"""When signal_groups=[1, 2] is given, both the profile loss and
	the count loss are per-group: the stranded pair contributes ONE
	profile-loss term (a single joint multinomial over its two strands
	and length) and ONE count target (sum of its two strands' counts).
	Each group thus contributes one term regardless of channel count."""

	# 3 channels: 1 unstranded + 1 stranded pair = 2 groups.
	y, logits, _ = _toy_inputs(n_outputs=3)
	# Per-group log counts: shape (n, 2).
	logcounts_grouped = torch.randn(y.shape[0], 2,
		generator=torch.Generator().manual_seed(1))

	profile_loss, count_loss = _mixture_loss(y, logits, logcounts_grouped,
		signal_groups=[1, 2])
	assert profile_loss.shape == (2,)  # one per group
	assert count_loss.shape == (2,)    # one per group

	# Sanity: per-group MSE matches hand-pooled y.
	y_per_track = y.sum(dim=-1)
	y_per_group = torch.stack([
		y_per_track[:, 0],
		y_per_track[:, 1] + y_per_track[:, 2],
	], dim=-1)
	expected_count = ((torch.log(y_per_group + 1) - logcounts_grouped) ** 2
		).mean(dim=0)
	assert torch.allclose(count_loss, expected_count, atol=1e-5)

	# Per-group profile loss is a single joint multinomial over the
	# group's flattened (channels * length) axis. Group 0 (size 1) is a
	# plain per-length multinomial of channel 0; group 1 (size 2) is one
	# multinomial over channels 1 and 2 concatenated.
	n = y.shape[0]
	lp0 = torch.nn.functional.log_softmax(logits[:, 0:1].reshape(n, -1), dim=-1)
	lp1 = torch.nn.functional.log_softmax(logits[:, 1:3].reshape(n, -1), dim=-1)
	expected_profile = torch.stack([
		MNLLLoss(lp0, y[:, 0:1].reshape(n, -1)).mean(dim=0),
		MNLLLoss(lp1, y[:, 1:3].reshape(n, -1)).mean(dim=0),
	])
	assert torch.allclose(profile_loss, expected_profile, atol=1e-5)


def test_mixture_loss_grouped_profile_matches_joint_multinomial():
	"""A stranded (+, -) group's profile loss equals a single multinomial
	over both strands concatenated — i.e. the two strands are normalized
	jointly, not independently (the fix for the one-strand-collapse bug)."""

	y, logits, _ = _toy_inputs(n_outputs=2)
	logcounts = torch.zeros(y.shape[0], 1)

	profile_loss, _ = _mixture_loss(y, logits, logcounts, signal_groups=[2])
	assert profile_loss.shape == (1,)

	n = y.shape[0]
	log_probs = torch.nn.functional.log_softmax(logits.reshape(n, -1), dim=-1)
	expected = MNLLLoss(log_probs, y.reshape(n, -1)).mean(dim=0)
	assert torch.allclose(profile_loss, expected, atol=1e-5)


def test_mixture_loss_grouped_profile_couples_strands():
	"""Regression test for the one-strand-collapse bug. Under the joint
	per-group normalization the profile loss must respond to the RELATIVE
	offset between a group's strands: shifting one strand's logits by a
	constant re-weights the multinomial and changes the loss. The pre-fix
	per-channel normalization left that relative offset a free gauge
	(loss invariant to it), which let a trained model dump nearly all of a
	group's predicted counts onto a single strand at inference. A constant
	shift applied to the WHOLE group is the true softmax gauge and must
	leave the loss unchanged."""

	y, logits, _ = _toy_inputs(n_outputs=2)
	logcounts = torch.zeros(y.shape[0], 1)

	base, _ = _mixture_loss(y, logits, logcounts, signal_groups=[2])

	# Shift only the plus strand: the joint softmax MUST see this.
	one_strand = logits.clone()
	one_strand[:, 0] += 3.0
	shifted, _ = _mixture_loss(y, one_strand, logcounts, signal_groups=[2])
	assert not torch.allclose(base, shifted, atol=1e-4)

	# Shift the whole group (both strands, all positions): pure gauge.
	whole_group, _ = _mixture_loss(y, logits + 5.0, logcounts,
		signal_groups=[2])
	assert torch.allclose(base, whole_group, atol=1e-4)


def test_mixture_loss_single_output_bit_identical_across_group_specs():
	"""Single-output (unstranded, e.g. ATAC/DNase) models must be
	completely unaffected by the grouped-normalization change: passing
	signal_groups=None, [1], or omitting it entirely must all produce the
	exact same profile and count losses, bit-for-bit. This is the
	guarantee that accessibility checkpoints need no retraining."""

	y, logits, logcounts = _toy_inputs(n_outputs=1, n_count_outputs=1)

	prof_none, count_none = _mixture_loss(y, logits, logcounts)
	prof_grp, count_grp = _mixture_loss(y, logits, logcounts,
		signal_groups=[1])

	# Exact equality (not allclose): same ops, same order.
	assert torch.equal(prof_none, prof_grp)
	assert torch.equal(count_none, count_grp)

	# And both equal a hand-written per-channel multinomial.
	log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
	expected_prof = MNLLLoss(log_probs, y).mean(dim=0)
	assert torch.equal(prof_none, expected_prof)


def test_mixture_loss_single_output_matches_legacy_forward_and_grad():
	"""Single-output (ATAC/DNase) profile loss AND its gradient w.r.t. the
	logits are bitwise-identical to the pre-fix per-channel formula, with
	and without the ``labels`` filter and at a realistic output length.
	This guards the accessibility no-op guarantee at the *gradient* level
	(what actually drives training), not just the forward value, and pins
	the reordered slice-then-softmax ``labels`` branch."""

	for labels in (None, torch.tensor([1, 0, 1, 0])):
		g = torch.Generator().manual_seed(3)
		n = 4
		y = torch.randint(0, 60, (n, 1, 200), generator=g).float()
		base_logits = torch.randn(n, 1, 200, generator=g)
		logcounts = torch.randn(n, 1, generator=g)

		logits = base_logits.clone().requires_grad_(True)
		profile_loss, _ = _mixture_loss(y, logits, logcounts, labels=labels,
			signal_groups=[1])
		profile_loss.sum().backward()

		ref_logits = base_logits.clone().requires_grad_(True)
		ref = _legacy_single_output_profile_loss(y, ref_logits, labels=labels)
		ref.sum().backward()

		assert torch.equal(profile_loss, ref)
		assert torch.equal(logits.grad, ref_logits.grad)


def test_mixture_loss_all_size_one_groups_bit_identical():
	"""signal_groups=[1, 1, 1] (several unstranded co-trained tracks) must
	be bit-identical to the per-channel signal_groups=None path for BOTH
	the profile and count losses — the joint-normalization branch is never
	entered when every group is size one."""

	y, logits, logcounts = _toy_inputs(n_outputs=3, n_count_outputs=3)
	prof_none, count_none = _mixture_loss(y, logits, logcounts)
	prof_grp, count_grp = _mixture_loss(y, logits, logcounts,
		signal_groups=[1, 1, 1])
	assert torch.equal(prof_none, prof_grp)
	assert torch.equal(count_none, count_grp)


def test_mixture_loss_signal_groups_size_mismatch_raises():
	y, logits, logcounts = _toy_inputs(n_outputs=3, n_count_outputs=2)
	# sum(signal_groups) must equal y.shape[1] (=3).
	with pytest.raises(ValueError, match="sum.signal_groups"):
		_mixture_loss(y, logits, logcounts, signal_groups=[1, 1])
