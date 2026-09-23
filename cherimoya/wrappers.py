# wrappers.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
A set of wrappers to help in using Cherimoya models.
"""

import torch


class ControlWrapper(torch.nn.Module):
	"""A wrapper that supplies an all-zero control track when none is given.

	Cherimoya models trained with control tracks expect a control tensor at
	every forward pass, but attribution and marginalization tools call the model
	with the sequence alone. This wrapper bridges that gap: when no control is
	passed it checks whether the model expects one and, if so, synthesizes a
	control track of all zeroes with the matching shape, dtype, and device. Models
	without control tracks are forwarded through unchanged. The wrapper returns the
	model's full ``(profile, log-count)`` output, so it is meant to be the inner
	wrapper that :class:`ProfileWrapper`, :class:`LogCountWrapper`, or
	:class:`ExpectedCountsWrapper` are layered on top of. This is a port of
	``bpnetlite.bpnet.ControlWrapper`` so that Cherimoya does not depend on
	bpnet-lite for this behavior.


	Parameters
	----------
	model: cherimoya.Cherimoya
		A Cherimoya model, which makes predictions for basepair resolution profiles
		and also for log counts.
	"""

	def __init__(self, model):
		super().__init__()
		self.model = model

	@property
	def signal_groups(self):
		"""The wrapped model's signal groups.

		``torch.nn.Module`` does not forward attribute lookups to
		submodules, so an outer wrapper that needs to know how the
		model's channels are grouped — :class:`ExpectedCountsWrapper`
		does, to split the profile head — would otherwise find nothing
		here and fail with ``AttributeError``. Forwarding it is what
		makes this class the transparent inner wrapper its docstring
		describes.
		"""

		return self.model.signal_groups

	def forward(self, X, X_ctl=None):
		if X_ctl is not None:
			return self.model(X, X_ctl)

		if self.model.n_control_tracks == 0:
			return self.model(X)

		X_ctl = torch.zeros(X.shape[0], self.model.n_control_tracks,
			X.shape[-1], dtype=X.dtype, device=X.device)
		return self.model(X, X_ctl)


class _ProfileLogitScaling(torch.nn.Module):
	"""A non-linear scaling of profile logits, isolated for Captum.

	This module performs the non-linear part of :class:`ProfileWrapper` —
	multiplying logits by their own softmax — in its own ``forward`` so that
	Captum can register it as a non-linear operation. Captum classifies each
	registered module as linear or non-linear; the wrapper's inputs are the
	one-hot sequence (run through the model) rather than the logits being
	modified, so the non-linear logit operations must be quarantined here
	for attribution methods that walk the module graph.
	"""

	def __init__(self):
		super().__init__()
		self.softmax = torch.nn.Softmax(dim=-1)

	def forward(self, logits):
		y_softmax = self.softmax(logits)
		return logits * y_softmax


class ProfileWrapper(torch.nn.Module):
	"""A wrapper that returns the weighted-softmax of the profile logits.

	This wrapper takes the predicted profile logits and returns the dot product
	between them and their softmaxed values, summed across positions. The
	mean-centering and softmax weighting collapse the per-position profile into
	a single number per example whose attribution reflects the predicted profile
	*shape*. This is a port of ``bpnetlite.bpnet.ProfileWrapper`` so that
	Cherimoya does not depend on bpnet-lite for attribution.


	Parameters
	----------
	model: cherimoya.Cherimoya
		A Cherimoya model, which makes predictions for basepair resolution profiles
		and also for log counts.
	"""

	def __init__(self, model):
		super().__init__()
		self.model = model
		self.flatten = torch.nn.Flatten()
		self.scaling = _ProfileLogitScaling()

	def forward(self, X, X_ctl=None):
		logits = self.model(X, X_ctl=X_ctl)[0]
		logits = self.flatten(logits)
		logits = logits - torch.mean(logits, dim=-1, keepdims=True)
		return self.scaling(logits).sum(dim=-1, keepdims=True)


class LogCountWrapper(torch.nn.Module):
	"""A wrapper that extracts the log count predictions.

	This wraps a Cherimoya and slices out the second prediction, which is for the
	log counts. This is useful when you only care about the log count predictions,
	such as for feature attribution or design methods.


	Parameters
	----------
	model: cherimoya.Cherimoya
		A Cherimoya model, which makes predictions for basepair resolution profiles
		and also for log counts.
	"""

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, X, X_ctl=None):
		return self.model(X, X_ctl=X_ctl)[1]


class ExpectedCountsWrapper(torch.nn.Module):
	"""A wrapper that provides the expected counts per basepair.

	This wrapper combines the profile predictions and the log count predictions to
	give the expected number of reads mapping to each position. This is done by
	exponentiating the log count predictions and multiplying them by the softmaxed
	logit profiles. Essentially, we are distributing counts (not log counts) by the
	predicted probability distribution across positions.

	The distribution is performed jointly within each signal group. A group's
	profile channels and positions are softmaxed together so that the probabilities
	sum to one across the *entire* group, and the group's counts are then spread
	across all of its channels and positions. For a stranded ``(+, -)`` pair this
	means the expected counts summed over both strands and all positions equals the
	predicted count for that group. The count head is trained against
	``log(count + 1)``, so :func:`torch.expm1` is used to recover the counts.


	Parameters
	----------
	model: cherimoya.Cherimoya
		A Cherimoya model, which makes predictions for basepair resolution profiles
		and also for log counts.
	"""

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, X, X_ctl=None):
		y_logits, y_logcounts = self.model(X, X_ctl=X_ctl)
		y_counts = torch.expm1(y_logcounts)

		y_expected = []
		y_logit_groups = torch.split(y_logits, self.model.signal_groups, dim=1)
		for i, logits in enumerate(y_logit_groups):
			batch_size, n_channels, length = logits.shape
			probs = torch.softmax(logits.reshape(batch_size, -1), dim=-1)
			probs = probs.reshape(batch_size, n_channels, length)
			y_expected.append(probs * y_counts[:, i, None, None])

		return torch.cat(y_expected, dim=1)
