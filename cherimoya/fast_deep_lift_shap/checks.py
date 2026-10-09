# checks.py
# Author: Eugenio Mattei

"""
The two checks every run of the fast engine through `cherimoya attribute`
makes: a forward self-check before the run, and an audit after it.

* The forward self-check compares the engine's count output with the
  model's own forward, through `LogCountWrapper(ControlWrapper(model))`, on
  a few sequences, in strict float32. It catches a model the engine reads
  wrongly before anything is computed.
* The audit attributes a few evenly spaced sequences again with
  tangermeme's `deep_lift_shap`, as the default engine calls it, given the
  references the engine used for them, and compares the attributions. It
  checks first that those references are bit for bit the ones tangermeme
  draws itself.
"""

import time
import warnings

import numpy
import torch

from .engine import make_wrapper
from .engine import model_forward
from .engine import precision_mode
from .references import onehot_to_tokens
from .references import tokens_to_onehot


SELF_CHECK_PEAKS = 4

SELF_CHECK_MAX_ABS = 2.5e-4
"""The self-check fails if the engine's count output and the model's differ
by more than this, in strict float32, on any of its sequences. It is the
agreement `cherimoya.cheri` documents between the model's float32 forward
paths."""

AUDIT_PEARSON_MIN = 0.9999
AUDIT_REL_L2_MAX = 1e-2
"""The audit fails on a sequence whose attributions have a Pearson
correlation below `AUDIT_PEARSON_MIN`, or a relative L2 distance above
`AUDIT_REL_L2_MAX`, against tangermeme's."""

AUDIT_WARN_MEDIAN_REL_L2 = 3e-4
"""The audit warns when the median relative L2 distance over its sequences
exceeds this."""

AUDIT_MAX_BATCH = 20
"""The audit runs `deep_lift_shap` at min(batch_size, AUDIT_MAX_BATCH). At a
batch of 64, `deep_lift_shap` allocated 15.5 GB on the default 9-layer
architecture over 2114 bp, and about 6 GB at 20."""


class AuditFailed(RuntimeError):
	"""The audit found a sequence beyond its bounds, or references that are
	not tangermeme's. The outputs were written before it was raised."""


def forward_self_check(engine, model, X, peaks=SELF_CHECK_PEAKS,
	bound=SELF_CHECK_MAX_ABS):
	"""Compare the engine's count output with the model's own on the first
	sequences of X.

	The model runs as `deep_lift_shap` runs it: through
	``LogCountWrapper(ControlWrapper(model), group)``, under
	torch.enable_grad(), which on a GPU is the Triton training path. The
	engine runs `model_forward` eagerly with its own weights and statistics.

	The check passes if the two differ by at most `bound` in strict float32
	(`precision_mode("fp32")`). In "tf32" precision the model's own output
	depends on the batch shape it runs at, by more than that bound on the
	default architecture, so when the engine runs in "tf32" the same
	comparison in "tf32" is recorded but does not decide the check.


	Parameters
	----------
	engine: Engine
		The engine.

	model: cherimoya.Cherimoya
		The float32 model the engine was built from, on the engine's device.

	X: torch.Tensor, shape=(N, 4, L)
		One-hot sequences; the first `peaks` are used.

	peaks: int, optional
		The number of sequences. Default is 4.

	bound: float, optional
		The largest absolute difference that passes. Default is 2.5e-4.


	Returns
	-------
	record: dict
		"pass", "max_abs" (the strict float32 difference), and the outputs of
		both sides in each precision compared.
	"""

	began = time.perf_counter()
	n = min(int(peaks), len(X))
	device = engine.weights.device
	X4 = torch.as_tensor(X[:n]).to(device=device, dtype=torch.float32)

	wrapper = make_wrapper(model, "counts", engine.group).to(device).eval()
	record = {"peaks": n, "bound": bound, "gate_precision": "fp32",
		"comparisons": {}}

	modes = ["fp32"]
	if engine.precision != "fp32":
		modes.append(engine.precision)

	for mode in modes:
		with precision_mode(mode, inductor_cache=False), torch.enable_grad():
			module_y = wrapper(X4)[:, 0].detach().cpu().double()

		with precision_mode(mode, inductor_cache=False), torch.no_grad():
			ours = model_forward(X4, engine.weights,
				forward_stats=engine.forward_stats, output=engine.output,
				group=engine.group)

		ours_y = ours.y.detach().cpu().double()
		record["comparisons"][mode] = {
			"max_abs": float((ours_y - module_y).abs().max()),
			"module_y": module_y.tolist(),
			"ours_y": ours_y.tolist(),
		}

		if mode == engine.precision:
			record["module_y_run"] = module_y.tolist()

	record["max_abs"] = record["comparisons"]["fp32"]["max_abs"]
	record["pass"] = bool(record["max_abs"] <= bound)
	record["seconds"] = time.perf_counter() - began
	return record


def compare_run_outputs(check, y_x, precision):
	"""The run's own count outputs for the self-check's sequences against the
	model's, in the run's precision. Recorded, not checked: in "tf32" the
	model's output depends on the batch shape it runs at."""

	n = check["peaks"]
	module_y = torch.tensor(check["module_y_run"], dtype=torch.float64)
	run_y = y_x[:n].detach().cpu().double()
	return {"precision": precision,
		"max_abs": float((run_y - module_y).abs().max()),
		"run_y": run_y.tolist()}


def audit_peaks(n_kept, n_audit):
	"""The sequences the audit attributes again: round(linspace(0,
	n_kept - 1, n_audit)), every sequence when n_audit >= n_kept.

	Consecutive points of the linspace are at least 1 apart, so the rounded
	indices are distinct.


	Returns
	-------
	peaks: numpy.ndarray, dtype=int64
		Increasing indices.
	"""

	if n_audit <= 0 or n_kept <= 0:
		return numpy.zeros(0, numpy.int64)

	if n_audit >= n_kept:
		return numpy.arange(n_kept, dtype=numpy.int64)

	return numpy.round(numpy.linspace(0, n_kept - 1, n_audit)).astype(
		numpy.int64)


def compare_attributions(ours, stock, pearson_min=AUDIT_PEARSON_MIN,
	rel_l2_max=AUDIT_REL_L2_MAX, warn_median=AUDIT_WARN_MEDIAN_REL_L2):
	"""Per-sequence Pearson correlation and relative L2 distance,
	||ours - stock|| / ||stock||, over each sequence's values, in float64.


	Parameters
	----------
	ours, stock: numpy.ndarray, shape=(n, 4, W)
		The two sets of attributions.

	pearson_min, rel_l2_max, warn_median: float, optional
		The bounds. Default are the audit's.


	Returns
	-------
	record: dict
		Per-sequence values and their extremes; "pass" is every sequence
		within both bounds (NaN fails), "warn" a median relative L2 above
		`warn_median`.
	"""

	a = numpy.asarray(ours, dtype=numpy.float64).reshape(len(ours), -1)
	b = numpy.asarray(stock, dtype=numpy.float64).reshape(len(stock), -1)
	if a.shape != b.shape or not len(a):
		raise ValueError("the audit compares equal, non-empty shapes, got {} "
			"and {}".format(a.shape, b.shape))

	rel = numpy.linalg.norm(a - b, axis=1) / numpy.maximum(
		numpy.linalg.norm(b, axis=1), 1e-30)
	ac = a - a.mean(axis=1, keepdims=True)
	bc = b - b.mean(axis=1, keepdims=True)
	r = (ac * bc).sum(axis=1) / numpy.maximum(numpy.linalg.norm(ac, axis=1)
		* numpy.linalg.norm(bc, axis=1), 1e-300)
	ok = (r >= pearson_min) & (rel <= rel_l2_max)
	return {
		"pearson": r.tolist(),
		"rel_l2": rel.tolist(),
		"min_pearson": float(r.min()),
		"median_rel_l2": float(numpy.median(rel)),
		"max_rel_l2": float(rel.max()),
		"failed_positions": numpy.flatnonzero(~ok).tolist(),
		"bounds": {"pearson_min": pearson_min, "rel_l2_max": rel_l2_max,
			"warn_median_rel_l2": warn_median},
		"pass": bool(ok.all()),
		"warn": not float(numpy.median(rel)) <= warn_median,
	}


def tangermeme_references(X, peaks, n_shuffles, random_state):
	"""tangermeme's own draw of the references of some sequences, made as
	`deep_lift_shap` makes it: reference k of sequence e is
	``dinucleotide_shuffle(X[e:e+1].float(), n=1, random_state=random_state
	+ k)[:, 0]``.


	Returns
	-------
	references: list of numpy.ndarray, dtype=uint8, shape=(n_shuffles, L)
		The references of each sequence in `peaks`, as tokens.
	"""

	from tangermeme.ersatz import dinucleotide_shuffle

	out = []
	for e in peaks:
		x = X[int(e):int(e) + 1].float()
		references = [dinucleotide_shuffle(x, n=1,
			random_state=random_state + k)[:, 0] for k in range(n_shuffles)]
		out.append(onehot_to_tokens(torch.cat(references)))

	return out


def audit(engine, model, X, result, peaks, start, end, batch_size=64,
	warning_threshold=1e-3):
	"""Attribute some sequences again with tangermeme's `deep_lift_shap`,
	given the references the engine used, and compare.

	`deep_lift_shap` runs as `cherimoya attribute`'s default engine runs it:
	through the real ``LogCountWrapper(ControlWrapper(model), group)``, with
	`attribution_ops()`, in float32, on the engine's device and in the
	engine's precision mode, at min(batch_size, `AUDIT_MAX_BATCH`). Before
	that, the engine's references for these sequences are compared, bit for
	bit, with tangermeme's own draw (`tangermeme_references`). Its per-batch
	convergence warnings are counted, not shown.


	Parameters
	----------
	engine: Engine
		The engine that made `result`.

	model: cherimoya.Cherimoya
		The float32 model, on the engine's device.

	X: torch.Tensor, shape=(N, 4, L)
		The sequences the engine attributed.

	result: Result
		The engine's result, with `result.references` the references of
		`peaks` (`Engine.run(keep_references=peaks)`).

	peaks: numpy.ndarray
		The sequences to audit, as `audit_peaks` returns them.

	start, end: int
		The window the engine attributed.

	batch_size: int, optional
		The batch size `cherimoya attribute` was given. Default is 64.

	warning_threshold: float or None, optional
		Passed on to `deep_lift_shap`. Default is 1e-3.


	Returns
	-------
	record: dict
		The audited sequences, the comparison (`compare_attributions`),
		whether the references are tangermeme's, and "pass".
	"""

	from tangermeme.deep_lift_shap import deep_lift_shap

	from ..deep_lift_shap import attribution_ops

	began = time.perf_counter()
	device = engine.weights.device
	K = result.references.shape[1]
	bs = max(1, min(int(batch_size), AUDIT_MAX_BATCH))
	record = {
		"peaks": len(peaks),
		"indices": [int(i) for i in peaks],
		"batch_size": bs,
		"dtype": "float32",
		"device": str(device),
		"precision": engine.precision,
		"deterministic": engine.deterministic,
		"n_shuffles": int(K),
		"random_state": engine.random_state,
	}

	drawn = tangermeme_references(X, peaks, K, engine.random_state)
	same = [bool(numpy.array_equal(result.references[i], drawn[i]))
		for i in range(len(peaks))]
	record["references_match_tangermeme"] = all(same)
	record["references_mismatched_positions"] = [i for i, ok
		in enumerate(same) if not ok]

	R = tokens_to_onehot(result.references, torch.float32)
	wrapper = make_wrapper(model, "counts", engine.group).to(device).eval()
	index = torch.as_tensor(numpy.asarray(peaks, numpy.int64))
	if device.type == "cuda":
		# The engine's cached blocks go back before deep_lift_shap allocates.
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats(device)

	if warning_threshold is None:
		threshold = float("inf")
	else:
		threshold = float(warning_threshold)

	t = time.perf_counter()
	with precision_mode(engine.precision, deterministic=engine.deterministic,
			inductor_cache=False), warnings.catch_warnings(
			record=True) as caught:
		warnings.simplefilter("always")
		stock = deep_lift_shap(wrapper, X[index], references=R, n_shuffles=K,
			hypothetical=True, batch_size=bs, warning_threshold=threshold,
			additional_nonlinear_ops=attribution_ops(), dtype="float32",
			device=str(device), random_state=engine.random_state,
			verbose=False)[:, :, start:end].float()

	record["stock_s"] = time.perf_counter() - t

	delta_warnings = [w for w in caught if str(w.message).startswith(
		"Convergence deltas too high")]
	for w in caught:
		if w not in delta_warnings:
			warnings.warn_explicit(w.message, w.category, w.filename,
				w.lineno)

	record["stock_delta_warnings"] = len(delta_warnings)
	if device.type == "cuda":
		record["gpu_max_allocated_bytes"] = torch.cuda.max_memory_allocated(
			device)

	ours = result.attr[index].float()
	record.update(compare_attributions(ours.numpy(), stock.numpy()))
	record["pass"] = bool(record["pass"]
		and record["references_match_tangermeme"])
	record["seconds"] = time.perf_counter() - began
	return record


def audit_summary(record):
	"""One line describing an audit record."""

	return ("{} sequences against tangermeme's deep_lift_shap (batch {}, {} "
		"precision): min Pearson {:.8f}, relative L2 median {:.3g} and max "
		"{:.3g} (bounds {}, {:g}); references {} to tangermeme's; {:.1f} "
		"s".format(record["peaks"], record["batch_size"], record["precision"],
		record["min_pearson"], record["median_rel_l2"], record["max_rel_l2"],
		AUDIT_PEARSON_MIN, AUDIT_REL_L2_MAX, "equal"
		if record["references_match_tangermeme"] else "NOT equal",
		record["seconds"]))
