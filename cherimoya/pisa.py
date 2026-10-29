# pisa.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
PISA attributions of a Cherimoya model's profile.

`pisa` attributes the profile logits of one channel at every output
position: for each position, the DeepLIFT/SHAP attributions of the logit
there to every input nucleotide::

    from cherimoya import Cherimoya
    from cherimoya.pisa import pisa

    model = Cherimoya.load("model.torch", compile=False)
    X_attr = pisa(model, X, channel=0, random_state=0)

The result has shape ``(n, out_length, 4, length)``: about 34 MB per
sequence in float32 for a model with the default trimming over 2,114 bp.
"""

from __future__ import annotations

import time
import warnings

import numpy
import torch
import tangermeme.pisa

from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.utils import validate_input

from ._pisa_engine import Engine
from .cherimoya import Cherimoya
from .deep_lift_shap import _CHECK_PEARSON_MIN
from .deep_lift_shap import _CHECK_REL_L2_MAX
from .deep_lift_shap import _agreement
from .deep_lift_shap import attribution_ops
from .wrappers import ControlWrapper


_CHECK_SEQUENCES = 2
"""`pisa` attributes the first this many sequences with the default path
before it decides on the fast path: the first to pay the default path's
first-call costs, the second to time it, and both to check the fast path
against."""

_FAST_MIN_SECONDS = 30.0
"""`pisa` takes the fast path only for a job that the default path would
take at least this many seconds to finish. On a GPU the fast path warms up
the first time a process runs it, compiling its passes: on an H200, a first
call took 8 to 15 seconds longer than a second with warm compile caches,
and 41 to 48 seconds longer the first time a model's width was compiled, in
our measurements."""


class _ProfileChannelWrapper(torch.nn.Module):
	"""The profile logits of one channel, (batch, out_length), the
	per-position output tangermeme's `pisa` attributes, of a Cherimoya model
	given all-zero control tracks by `ControlWrapper`."""

	def __init__(self, model, channel):
		super().__init__()
		self.model = ControlWrapper(model)
		self.channel = channel

	def forward(self, X):
		return self.model(X)[0][:, self.channel]


def _check_channel(model, channel):
	"""Raise a ValueError unless `channel` is an index into the model's
	profile channels."""

	n_channels = sum(model.signal_groups)
	if (isinstance(channel, bool) or not isinstance(channel, int)
			or not 0 <= channel < n_channels):
		raise ValueError("channel must be an int in [0, {}) for a model with "
			"signal_groups={}, got {!r}".format(n_channels,
			list(model.signal_groups), channel))


def pisa(model, X, channel=0, batch_size=32, references=dinucleotide_shuffle,
	n_shuffles=20, hypothetical=False, warning_threshold=0.001, device=None,
	random_state=None, verbose=False, fast=True):
	"""PISA attributions of a Cherimoya model's profile logits.

	PISA (pairwise influence by sequence attribution) runs DeepLIFT/SHAP
	once for every output position, giving the attributions of the
	prediction at each position to every input nucleotide. Here the output
	is the profile logits of one channel, as the model returns them before
	any softmax, with all-zero control tracks, the input `ControlWrapper`
	gives a model trained with controls. `attribution_ops()` is registered
	for the model's fused convolution and normalization.

	There are two paths. The default path is tangermeme's `pisa`, which
	attributes through backward hooks on the model's modules. The fast path
	computes the same attributions from a forward and a backward pass
	written out for Cherimoya's architecture, with tangermeme's DeepLIFT
	rules and references: each sequence is forwarded once with its
	references, rather than many times for each reference; the terms of the
	rules that depend only on the activations are computed once for each
	sequence-reference pair rather than once for each output position; and
	the backward pass runs over the sequence half of each pair only. It runs
	several times faster on a GPU. The two paths agree up to floating-point
	rounding, which on a GPU is the rounding of TF32, which ``import
	cherimoya`` turns on for matrix products.

	The fast path is used when `fast` is True, `model` is a `Cherimoya` with
	float32 parameters, `X` is a tensor of more than two sequences, and
	`references` is `dinucleotide_shuffle` or a tensor. All of `X` and the
	references are then checked as tangermeme checks them, so an unknown
	character raises before either path runs, and the first two sequences
	are attributed with the default path, the second of them timed. When the
	whole job would take the default path less than thirty seconds at that
	rate, the default path attributes the rest, since the fast path's first
	compile on a GPU can take longer than it saves. Otherwise the fast path
	attributes every sequence, and its attributions of the first two are
	compared with the default path's. When they disagree beyond rounding,
	with a Pearson correlation below 0.9999 or a relative L2 distance above
	1e-2 on either of them, or when the fast path raises, a
	`RuntimeWarning` gives the reason and the default path attributes the
	rest.

	Both paths draw the same references: with `random_state`, those of every
	sequence are ``dinucleotide_shuffle(X[i:i+1], n=n_shuffles,
	random_state=random_state)``, as tangermeme's `pisa` draws them. With
	`random_state` None, a seed is drawn once per call when the fast path
	may be used.


	Parameters
	----------
	model: cherimoya.Cherimoya
		The model. It is not moved or changed.

	X: torch.Tensor, shape=(n, 4, length)
		The one-hot sequences, such as those `extract_loci` returns, without
		unknown characters.

	channel: int, optional
		The profile channel to attribute, an index into the model's
		``sum(signal_groups)`` profile channels: for a model with one
		stranded group, 0 for the + strand and 1 for the - strand. Default
		is 0.

	batch_size: int, optional
		The number of output positions per backward pass on the default
		path. The fast path sizes its chunks of output positions from the
		memory free on the device. Default is 32.

	references: callable or torch.Tensor, optional
		`dinucleotide_shuffle`; another function that takes ``(X, n=...,
		random_state=...)`` and returns ``(n_examples, n, 4, length)``
		references, which uses the default path; or a tensor of shape
		``(n, n_references, 4, length)``. Default is `dinucleotide_shuffle`.

	n_shuffles: int, optional
		The number of references drawn per sequence when `references` is a
		function. Default is 20.

	hypothetical: bool, optional
		Whether to return the hypothetical attributions of every character
		at every position, rather than those multiplied by `X`. Default is
		False.

	warning_threshold: float or None, optional
		A `RuntimeWarning` is given when a convergence delta exceeds this:
		per chunk of output positions on the default path, once per call on
		the fast path. None gives no such warning. Default is 0.001.

	device: str or torch.device or None, optional
		The device to compute on. None means "cuda" when CUDA is available
		and "cpu" otherwise. Default is None.

	random_state: int or None, optional
		The seed of the references. Default is None.

	verbose: bool, optional
		Whether to show progress bars. Default is False.

	fast: bool, optional
		Whether to use the fast path where it applies. False always uses
		tangermeme's `pisa`. Default is True.


	Returns
	-------
	attributions: torch.Tensor, shape=(n, out_length, 4, length)
		The attributions of each output position, averaged over each
		sequence's references, on the CPU. ``out_length`` is ``length - 2 *
		model.trimming``, the length of the profile the model predicts.
	"""

	_check_channel(model, channel)
	wrapper = _ProfileChannelWrapper(model, channel)
	threshold = float("inf") if warning_threshold is None else warning_threshold
	given = isinstance(references, torch.Tensor)
	dtype = next(model.parameters()).dtype

	# tangermeme's pisa takes gradients with respect to X and the references
	# as they are given, so they must be floating point.
	def default(X_, references_):
		if isinstance(references_, torch.Tensor):
			references_ = references_.to(dtype)

		return tangermeme.pisa.pisa(wrapper, X_.to(dtype),
			batch_size=batch_size, references=references_,
			n_shuffles=n_shuffles, hypothetical=True,
			warning_threshold=threshold,
			additional_nonlinear_ops=attribution_ops(), device=device,
			random_state=random_state, verbose=verbose)

	def default_part(a, b=None):
		return default(X[a:b], references[a:b] if given else references)

	def finish(attributions):
		return attributions if hypothetical else attributions * X.cpu()[:,
			None]

	float32 = all(p.dtype == torch.float32 for p in model.parameters())
	n, half = _CHECK_SEQUENCES, _CHECK_SEQUENCES // 2
	if not (fast and isinstance(model, Cherimoya) and float32
			and isinstance(X, torch.Tensor) and len(X) > n
			and (references is dinucleotide_shuffle or given)):
		return finish(default(X, references))

	# The default path checks only the sequences it is given, so the whole
	# input is checked here, before either path runs, as tangermeme checks
	# it: an unknown character in a sequence the check does not attribute
	# raises the same error with fast=True as with fast=False.
	validate_input(X, "X", shape=(-1, -1, -1), ohe=True)
	if given:
		validate_input(references, "references", shape=(X.shape[0], -1,
			X.shape[1], X.shape[2]), ohe=True, allow_N=True, ohe_dim=-2)

	# Both paths draw the references of every sequence with the seed
	# random_state, so they draw the same references only with a seed.
	if random_state is None:
		random_state = int(numpy.random.randint(0, 9_999_999))

	checked = [default_part(0, half)]
	began = time.perf_counter()
	checked.append(default_part(half, n))
	seconds = time.perf_counter() - began
	checked = torch.cat(checked)

	if seconds * (len(X) - half) / (n - half) >= _FAST_MIN_SECONDS:
		try:
			if device is None:
				device = "cuda" if torch.cuda.is_available() else "cpu"

			engine = Engine(model, channel=channel, device=device,
				n_shuffles=n_shuffles, random_state=random_state,
				warning_threshold=warning_threshold)
			attributions = engine.run(X, references if given else None,
				verbose=verbose).attr
			pearson, rel_l2 = _agreement(attributions[:n], checked)
			if pearson >= _CHECK_PEARSON_MIN and rel_l2 <= _CHECK_REL_L2_MAX:
				return finish(attributions)

			reason = ("its attributions of the first {} sequences had a "
				"Pearson correlation as low as {:.6f} and a relative L2 "
				"distance as high as {:.3g} with the default path's".format(n,
				pearson, rel_l2))
		except Exception as exc:
			reason = "it raised {}: {}".format(type(exc).__name__,
				str(exc).strip().split("\n")[0])

		warnings.warn("pisa is using the default path because the fast path "
			"{}. Pass fast=False to skip the fast path.".format(reason),
			RuntimeWarning, stacklevel=2)

	return finish(torch.cat([checked, default_part(n)]))
