"""Tiny Cherimoya models, one-hot inputs, references and a tangermeme oracle
for the tests of ``cherimoya.fast_deep_lift_shap``.

Models. `CONFIGS` holds three configurations, each built with
``Cherimoya(..., compile=False, random_state=seed)``:

* A: 16 filters, 3 layers, expansion 2, one group, no controls, L=256.
* B: 8 filters, 2 layers, expansion 1, groups [1, 2], one control track,
  L=200. Both heads attribute group 1.
* C: 128 filters, 2 layers, expansion 2, one group, L=512: the default
  model's matrix widths, C=128 and H=256.

The default initialization (std 0.02) leaves every GELU nearly linear, so
`redraw_weights` redraws every weight the forward reads from N(0, std) with
one seeded generator, so that the DeepLIFT rules change the multipliers.

float64. Cherimoya cannot run a float64 forward: the count head casts its
pooled stream with ``.float()`` and hands a float32 tensor to the float64
count Linear. `enable_float64` rebinds the forward of one model instance to
the model's own ``_forward_impl`` source with those two casts widened to the
count Linear's dtype. For float32 parameters that is the same operation, and
every module call is unchanged, so tangermeme's hooks still fire. The
normalization's own float32 cast is untouched; it is what bounds a float64
oracle.

References, each (N, K, 4, L) float32, injected identically into both sides
of a comparison: `shuffled_references` (exactly the dinucleotide shuffles
`deep_lift_shap` draws itself), `identical_references` (r = x),
`mutated_references` (x with 3 positions changed) and `random_references`.

Oracle: `oracle` runs tangermeme's `deep_lift_shap` as `cherimoya attribute`
calls it, but with injected references and on the CPU, and returns its
per-pair convergence deltas, caught through a module-global ``print``
(`capture_deltas`).
"""

import builtins
import contextlib
import importlib
import inspect
import math
import sys
import textwrap
import types
import warnings

from dataclasses import dataclass
from typing import NamedTuple

import numpy
import torch

from tangermeme.ersatz import dinucleotide_shuffle

from cherimoya import Cherimoya
from cherimoya import ControlWrapper
from cherimoya import LogCountWrapper
from cherimoya import ProfileWrapper
from cherimoya.deep_lift_shap import attribution_ops


# The submodule, not the function of the same name.
TDLS = importlib.import_module("tangermeme.deep_lift_shap")


@dataclass(frozen=True)
class TinyConfig:
	"""A tiny Cherimoya architecture, its input length, and the group its
	heads attribute."""

	name: str
	n_filters: int
	n_layers: int
	expansion: int
	signal_groups: tuple
	n_control_tracks: int
	length: int
	group: int


CONFIGS = {
	"A": TinyConfig("A", 16, 3, 2, (1,), 0, 256, 0),
	"B": TinyConfig("B", 8, 2, 1, (1, 2), 1, 200, 1),
	"C": TinyConfig("C", 128, 2, 2, (1,), 0, 512, 0),
}

N_RANDOM, N_AT_RICH = 12, 1


def _config(config):
	return CONFIGS[config] if isinstance(config, str) else config


def redraw_weights(model, seed):
	"""Redraw, in place, every weight the forward reads from N(0, std)."""

	gen = torch.Generator().manual_seed(seed)
	C = model.n_filters
	H = model.expansion * C

	def draw(param, std):
		param.copy_(torch.randn(param.shape, generator=gen,
			dtype=torch.float64) * std)

	with torch.no_grad():
		draw(model.iconv.weight, 0.5)
		draw(model.iconv.bias, 0.3)
		for block in model.blocks:
			draw(block.conv.conv_weight, 0.8)
			draw(block.linear1.weight, 1.5 / math.sqrt(C))
			draw(block.linear2.weight, 1.0 / math.sqrt(H))

		draw(model.fconv.weight, 0.1)
		draw(model.fconv.bias, 0.1)
		draw(model.linear.weight, 1.0)
		draw(model.linear.bias, 0.1)

	return model


def tiny_model(config="A", seed=0, dtype=torch.float32):
	"""A tiny Cherimoya with redrawn weights: float32, or float64 with
	`enable_float64` applied."""

	cfg = _config(config)
	model = Cherimoya(n_filters=cfg.n_filters, n_layers=cfg.n_layers,
		signal_groups=list(cfg.signal_groups),
		n_control_tracks=cfg.n_control_tracks, expansion=cfg.expansion,
		verbose=False, compile=False, random_state=seed)
	redraw_weights(model, seed)
	if dtype == torch.float64:
		return enable_float64(model.double())

	if dtype != torch.float32:
		raise ValueError("tiny models are float32 or float64, not "
			"{}".format(dtype))

	return model


def enable_float64(model):
	"""Let `model` run its own forward in float64: its ``_forward_impl``
	with the two head casts widened. Only this instance changes."""

	source = textwrap.dedent(inspect.getsource(Cherimoya._forward_impl))
	if source.count(".float()") != 2:
		raise RuntimeError("Cherimoya._forward_impl no longer has exactly "
			"the two .float() casts this widens")

	source = source.replace(".float()", ".to(self.linear.weight.dtype)")
	namespace = dict(vars(sys.modules[Cherimoya.__module__]))
	exec(compile(source, "<Cherimoya._forward_impl, widened head casts>",
		"exec"), namespace)
	model._forward_fn = types.MethodType(namespace["_forward_impl"], model)
	return model


def onehot(tokens):
	"""Base indices 0-3 of shape (..., L) to an int8 one-hot (..., 4, L),
	the dtype `extract_loci` returns."""

	expanded = numpy.eye(4, dtype=numpy.int8)[numpy.asarray(tokens)]
	return torch.from_numpy(numpy.ascontiguousarray(numpy.moveaxis(expanded,
		-1, -2)))


def random_onehot(n, length, seed):
	"""n uniformly random sequences, int8 (n, 4, length)."""

	return onehot(numpy.random.RandomState(seed).randint(0, 4, (n, length)))


def at_rich_onehot(n, length, seed, at_fraction=0.9):
	"""n sequences whose positions are A or T with probability
	`at_fraction`, int8 (n, 4, length)."""

	rng = numpy.random.RandomState(seed)
	at = rng.random_sample((n, length)) < at_fraction
	return onehot(numpy.where(at, rng.choice([0, 3], (n, length)),
		rng.choice([1, 2], (n, length))))


def sequences(config, seed=0):
	"""12 random sequences, then 1 AT-rich one, int8 (13, 4, L), at the
	config's length."""

	length = _config(config).length
	return torch.cat([random_onehot(N_RANDOM, length, seed=seed),
		at_rich_onehot(N_AT_RICH, length, seed=seed + 1)])


def _tokens(X):
	"""The base index at each position of a one-hot with exactly one base per
	position."""

	x = torch.as_tensor(X)
	if not bool((x.sum(dim=-2) == 1).all()):
		raise ValueError("expected a one-hot with exactly one base at every "
			"position")

	return x.argmax(dim=-2).numpy()


def shuffled_references(X, k=5, random_state=0):
	"""The references `deep_lift_shap` draws itself, float32 (N, k, 4, L):
	pair (e, j) gets ``dinucleotide_shuffle(X[e:e+1].float(), n=1,
	random_state=random_state + j)[:, 0]``."""

	return torch.stack([torch.cat([dinucleotide_shuffle(X[e:e + 1].float(),
		n=1, random_state=random_state + j)[:, 0] for j in range(k)])
		for e in range(len(X))])


def identical_references(X, k):
	"""r = x for every pair, float32 (N, k, 4, L). Every rule then falls back
	to the ordinary gradient."""

	return X.float().unsqueeze(1).repeat(1, k, 1, 1)


def mutated_references(X, k, n_mutations=3, seed=0):
	"""x with `n_mutations` distinct positions changed to another base,
	float32 (N, k, 4, L)."""

	tokens = _tokens(X)
	n, length = tokens.shape
	rng = numpy.random.RandomState(seed)
	out = numpy.repeat(tokens[:, None], k, axis=1)
	for e in range(n):
		for j in range(k):
			pos = rng.choice(length, n_mutations, replace=False)
			out[e, j, pos] = (out[e, j, pos] + rng.randint(1, 4,
				n_mutations)) % 4

	return onehot(out).float()


def random_references(X, k, seed=0):
	"""Random one-hots, unrelated to X, float32 (N, k, 4, L)."""

	n, _, length = X.shape
	return random_onehot(n * k, length, seed=seed).view(n, k, 4,
		length).float()


def wrapper(model, output="counts", group=0):
	"""The module `cherimoya attribute` attributes."""

	inner = ControlWrapper(model)
	if output == "counts":
		return LogCountWrapper(inner, group=group)

	if output == "profile":
		return ProfileWrapper(inner, group=group)

	raise ValueError("output must be counts or profile, got {!r}".format(
		output))


class OracleResult(NamedTuple):
	"""`oracle`'s output."""

	attributions: torch.Tensor  # multipliers (N, K, 4, L) or hyp (N, 4, L)
	deltas: torch.Tensor  # (N, K) tangermeme's own convergence deltas


@contextlib.contextmanager
def capture_deltas():
	"""Capture the convergence deltas `deep_lift_shap` prints with
	``print_convergence_deltas=True``.

	``print(convergence_deltas)`` is that module's only print, and a
	module-global ``print`` shadows the builtin there. The sink keeps a copy
	of each per-batch delta tensor and passes any other call on to the
	builtin. The global is removed on exit, also when the body raises.
	"""

	if "print" in vars(TDLS):
		raise RuntimeError("tangermeme.deep_lift_shap already has a "
			"module-global print")

	captured = []

	def sink(*args, **kwargs):
		if len(args) == 1 and not kwargs and isinstance(args[0], torch.Tensor):
			captured.append(args[0].detach().clone())
		else:
			builtins.print(*args, **kwargs)

	TDLS.print = sink
	try:
		yield captured
	finally:
		vars(TDLS).pop("print", None)


def oracle(model, X, references, raw_outputs, dtype, batch_size=7,
	warning_threshold=1e-3, num_threads=1):
	"""tangermeme's `deep_lift_shap` on the CPU with injected references
	(N, K, 4, L), plus its convergence deltas.

	The call is `cherimoya attribute`'s, hypothetical=True with cherimoya's
	`attribution_ops()`, except that the references are injected, the device
	is the CPU and the batch size defaults to 7, which splits sequences
	across batches. Pass a float64 model with dtype=torch.float64. torch
	runs `num_threads` intra-op threads during the call.
	"""

	previous = torch.get_num_threads()
	torch.set_num_threads(num_threads)
	try:
		with warnings.catch_warnings(), capture_deltas() as captured:
			warnings.filterwarnings("ignore",
				message="Convergence deltas too high", category=RuntimeWarning)
			out = TDLS.deep_lift_shap(model, X, references=references,
				batch_size=batch_size, hypothetical=True,
				raw_outputs=raw_outputs, warning_threshold=warning_threshold,
				additional_nonlinear_ops=attribution_ops(),
				print_convergence_deltas=True, dtype=dtype, device="cpu")
	finally:
		torch.set_num_threads(previous)

	deltas = torch.cat(captured).reshape(len(X), references.shape[1])
	return OracleResult(out, deltas)
