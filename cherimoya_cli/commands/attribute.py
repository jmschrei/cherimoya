# cherimoya_cli attribute command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import sys
import time
import warnings


ENGINES = ("default", "fast")
PRECISIONS = ("tf32", "fp32")


def run(args):

	from ..defaults import default_attribute_parameters
	from ..utils import merge_parameters

	parameters = merge_parameters(args.parameters, default_attribute_parameters)
	if parameters["skip"]:
		return

	algorithm = parameters["algorithm"]
	if algorithm not in ("deep_lift_shap", "saturation_mutagenesis"):
		raise ValueError("algorithm must be either `deep_lift_shap` or "
			"`saturation_mutagenesis`, got {!r}".format(algorithm))

	# The fast engine's options are checked before anything is loaded, and
	# its cache defaults are set before cherimoya and tangermeme are
	# imported, which is when tangermeme compiles its numba kernel and
	# cherimoya decorates its Triton kernels.
	fast = _fast_engine_selected(parameters)
	environment = _set_cache_defaults() if fast else None

	###

	import numpy

	from tangermeme.deep_lift_shap import deep_lift_shap
	from tangermeme.io import extract_loci
	from tangermeme.saturation_mutagenesis import saturation_mutagenesis

	from cherimoya import Cherimoya
	from cherimoya import ControlWrapper
	from cherimoya import LogCountWrapper
	from cherimoya import ProfileWrapper
	from cherimoya.deep_lift_shap import attribution_ops

	# `compile` defaults to false here: neither algorithm ran faster
	# compiled, and compiling added 6-70 s to the first call. DeepLIFT is
	# never compiled: its backward hooks cause graph breaks and recompiles.
	# The fast engine compiles its own passes on a GPU whatever this says.
	compiled = parameters["compile"] and algorithm == "saturation_mutagenesis"
	model = Cherimoya.load(parameters["model"], device=parameters["device"],
		compile=compiled,
		compile_mode=parameters["compile_mode"])

	# DeepLIFT attributes one output, so a count head with several groups
	# has to be narrowed to one of them first.
	group = parameters["group"]
	if (algorithm == "deep_lift_shap" and parameters["output"] == "counts"
			and group is None and len(model.signal_groups) > 1):
		raise ValueError("deep_lift_shap attributes one output; set `group` "
			"to one of the model's {} signal groups"
			.format(len(model.signal_groups)))

	X, idxs = extract_loci(
		sequences=parameters["sequences"],
		loci=parameters["loci"],
		chroms=parameters["chroms"],
		in_window=parameters["in_window"],
		max_jitter=0,
		exclusion_lists=parameters["exclusion_lists"],
		ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
		return_mask=True,
		verbose=parameters["verbose"],
	)

	n_idxs = X.sum(dim=(1, 2)) == X.shape[-1]
	X = X[n_idxs]
	idxs[idxs.clone()] = n_idxs

	control = ControlWrapper(model)
	if parameters["output"] == "counts":
		wrapper = LogCountWrapper(control, group=group)
	elif parameters["output"] == "profile":
		wrapper = ProfileWrapper(control, group=group)
	else:
		raise ValueError("output must be either `counts` or `profile`.")

	# Only the centred `attr_window` slice is saved. Saturation mutagenesis
	# is one forward pass per alternate base per position, so for it this
	# width sets the cost of the step; DeepLIFT attributes the whole input
	# in one pass per reference either way.
	attr_window = parameters["attr_window"]
	if attr_window > X.shape[-1]:
		raise ValueError(
			"attr_window ({}) is wider than the extracted sequence ({}); "
			"lower attr_window or raise in_window"
			.format(attr_window, X.shape[-1]))

	mid = X.shape[-1] // 2
	start = mid - attr_window // 2
	end = start + attr_window

	if algorithm == "deep_lift_shap" and fast:
		X_attr, fast_run = _fast_deep_lift_shap(parameters, model, X, start,
			end)
	elif algorithm == "deep_lift_shap":
		X_attr = deep_lift_shap(
			wrapper,
			X,
			hypothetical=True,
			n_shuffles=parameters["n_shuffles"],
			batch_size=parameters["batch_size"],
			warning_threshold=parameters["warning_threshold"],
			additional_nonlinear_ops=attribution_ops(),
			dtype=parameters["dtype"],
			device=parameters["device"],
			random_state=parameters["random_state"],
			verbose=parameters["verbose"],
		)[:, :, start:end].float()
	else:
		X_attr = saturation_mutagenesis(
			wrapper,
			X,
			dtype=parameters["dtype"],
			device=parameters["device"],
			batch_size=parameters["batch_size"],
			verbose=parameters["verbose"],
			hypothetical=True,
			start=start,
			end=end,
		).float()

	numpy.savez_compressed(parameters["ohe_filename"], X[:, :, start:end])
	numpy.savez_compressed(parameters["attr_filename"], X_attr)
	numpy.save(parameters["idx_filename"], idxs)

	if algorithm == "deep_lift_shap" and fast:
		_finish_fast_deep_lift_shap(parameters, model, X, idxs, start, end,
			fast_run, environment)


def _fast_engine_selected(parameters):
	"""Whether this run uses the fast engine, after checking `engine` and,
	if the fast engine runs, its options, so that a bad value fails before
	the model is loaded."""

	engine = parameters["engine"]
	if engine not in ENGINES:
		raise ValueError("engine must be either `default` or `fast`, got "
			"{!r}".format(engine))

	if engine == "default":
		return False

	if parameters["algorithm"] != "deep_lift_shap":
		warnings.warn("engine `fast` computes DeepLIFT/SHAP only; saturation "
			"mutagenesis runs as it does with engine `default`.",
			UserWarning, stacklevel=3)
		return False

	if parameters["output"] == "profile":
		warnings.warn("engine `fast` attributes the count head only; the "
			"profile head is attributed with engine `default`, tangermeme's "
			"deep_lift_shap.", UserWarning, stacklevel=3)
		return False

	def bad(key, want):
		return ValueError("`{}` must be {} for engine `fast`, got {!r}".format(
			key, want, parameters[key]))

	if parameters["precision"] not in PRECISIONS:
		raise bad("precision", "`tf32` or `fp32`")

	S = parameters["seqs_per_step"]
	if S != "auto" and (isinstance(S, bool) or not isinstance(S, int) or S < 1):
		raise bad("seqs_per_step", "`auto` or a positive integer")

	budget = parameters["mem_budget_gb"]
	if (isinstance(budget, bool) or not isinstance(budget, (int, float))
			or not budget > 0):
		raise bad("mem_budget_gb", "a positive number")

	workers = parameters["ref_workers"]
	if workers is not None and (isinstance(workers, bool)
			or not isinstance(workers, int) or workers < 0):
		raise bad("ref_workers", "null or a non-negative integer")

	n_audit = parameters["audit"]
	if isinstance(n_audit, bool) or not isinstance(n_audit, int) or n_audit < 0:
		raise bad("audit", "a number of sequences, 0 or more")

	# tangermeme runs bfloat16 and float16 under autocast, which computes a
	# different function from the one the fast engine computes.
	if parameters["dtype"] not in (None, "float32"):
		raise bad("dtype", "`float32` or null")

	return True


def _set_cache_defaults():
	"""Set the fast engine's two cache variables if they are unset, before
	cherimoya and tangermeme are imported; return what was found.

	NUMBA_CACHE_DIR: tangermeme compiles its numba shuffle kernel with
	caching on when it is imported, and the fast engine's reference workers
	load the compiled kernel from that cache. The default puts the cache in
	a per-user directory under the temporary directory rather than next to
	tangermeme's installed sources.

	TRITON_CACHE_AUTOTUNING=1: Triton then keeps the autotuning results of
	Cherimoya's kernels, which the self-check and the audit run, in its
	cache, so a later run on the same machine does not autotune them again.
	"""

	import getpass
	import tempfile

	record = {
		"cherimoya_imported_before": "cherimoya" in sys.modules,
		"tangermeme_imported_before": "tangermeme" in sys.modules,
	}

	set_by_cherimoya = []
	if not os.environ.get("NUMBA_CACHE_DIR"):
		try:
			user = getpass.getuser()
		except Exception:
			user = "user"

		path = os.path.join(tempfile.gettempdir(), "numba_cache_" + user)
		try:
			os.makedirs(path, exist_ok=True)
		except OSError as exc:
			warnings.warn("could not create {} for NUMBA_CACHE_DIR ({}); "
				"numba caches in its default location".format(path, exc),
				UserWarning, stacklevel=3)
		else:
			os.environ["NUMBA_CACHE_DIR"] = path
			set_by_cherimoya.append("NUMBA_CACHE_DIR")

	if "TRITON_CACHE_AUTOTUNING" not in os.environ:
		os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
		set_by_cherimoya.append("TRITON_CACHE_AUTOTUNING")

	for name in ("NUMBA_CACHE_DIR", "TRITON_CACHE_AUTOTUNING",
			"TRITON_CACHE_DIR"):
		record[name] = os.environ.get(name)

	record["set_by_cherimoya"] = set_by_cherimoya
	return record


def _fast_deep_lift_shap(parameters, model, X, start, end):
	"""Attribute X with the fast engine, after its forward self-check;
	return the float32 attributions and what `_finish_fast_deep_lift_shap`
	needs."""

	from cherimoya.fast_deep_lift_shap import Engine
	from cherimoya.fast_deep_lift_shap import audit_peaks
	from cherimoya.fast_deep_lift_shap import forward_self_check
	from cherimoya.fast_deep_lift_shap.checks import compare_run_outputs

	timings = {}
	engine = Engine(
		model,
		output="counts",
		group=parameters["group"],
		device=parameters["device"],
		precision=parameters["precision"],
		n_shuffles=parameters["n_shuffles"],
		random_state=parameters["random_state"],
		head_rows=2 * parameters["batch_size"],
		seqs_per_step=parameters["seqs_per_step"],
		mem_budget_gb=parameters["mem_budget_gb"],
		warning_threshold=parameters["warning_threshold"],
		ref_workers=parameters["ref_workers"],
	)

	t = time.perf_counter()
	check = forward_self_check(engine, model, X)
	timings["self_check"] = time.perf_counter() - t
	if not check["pass"]:
		raise RuntimeError("forward self-check failed: on the first {} "
			"sequences, in strict float32, the fast engine's count output "
			"differs from the model's by {:.4g}, above {:g}. Nothing was "
			"written.".format(check["peaks"], check["max_abs"],
			check["bound"]))

	if parameters["verbose"]:
		print("Fast DeepLIFT/SHAP self-check passed: the count output is "
			"within {:.3g} of the model's (strict float32, {} sequences)."
			.format(check["max_abs"], check["peaks"]))

	peaks = audit_peaks(len(X), parameters["audit"])
	keep = peaks if len(peaks) else None

	t = time.perf_counter()
	result = engine.run(X, start, end, keep_references=keep,
		verbose=parameters["verbose"])
	timings["run"] = time.perf_counter() - t

	check["run_outputs"] = compare_run_outputs(check, result.y_x,
		engine.precision)

	if parameters["verbose"]:
		meta = result.meta
		print("Fast DeepLIFT/SHAP: {} pairs in {:.1f} s ({:.0f} pairs/s), {} "
			"steps of {} sequences.".format(meta["pairs"], meta["seconds"],
			meta["pairs_per_s"], meta["steps"], meta["seqs_per_step"]))

	fast_run = {"engine": engine, "result": result, "check": check,
		"peaks": peaks, "timings": timings}
	return result.attr.float(), fast_run


def _finish_fast_deep_lift_shap(parameters, model, X, idxs, start, end,
	fast_run, environment):
	"""Write the fast engine's two sidecars, run the audit and record it.

	The sidecars sit next to `attr_filename`, without its ``.npz``:
	``.deltas.npy`` holds the convergence delta of every sequence-reference
	pair, and ``.meta.json`` a record of the run. The meta is written before
	the audit and again after it, so a crash in the audit leaves the run's
	record. A failed audit raises AuditFailed after everything is written.
	"""

	import numpy
	import torch

	from cherimoya.fast_deep_lift_shap import AuditFailed
	from cherimoya.fast_deep_lift_shap import audit
	from cherimoya.fast_deep_lift_shap.checks import audit_summary

	engine, result = fast_run["engine"], fast_run["result"]
	meta, timings = result.meta, fast_run["timings"]
	device = engine.weights.device
	cuda = device.type == "cuda"

	base = parameters["attr_filename"]
	if base.endswith(".npz"):
		base = base[:-len(".npz")]

	deltas_file, meta_file = base + ".deltas.npy", base + ".meta.json"
	numpy.save(deltas_file, result.deltas.float().numpy())

	deltas = dict(meta["deltas"])
	if deltas.get("above") is not None:
		deltas["fraction"] = deltas["above"] / deltas["pairs"]

	deltas["finite"] = bool(torch.isfinite(result.deltas).all())
	reference_stats = meta.get("reference_stats") or {}
	timings.update(
		compile=meta["compile_s"],
		warmup=meta["warmup_s"],
		reference_wait=reference_stats.get("wait_s"),
		steady_pairs_per_s=meta["steady_pairs_per_s"],
		pairs_per_s=meta["pairs_per_s"],
	)

	peaks = fast_run["peaks"]
	record = {
		"engine": "fast",
		"finished": _now(),
		"parameters": parameters,
		"versions": _versions(),
		"device": _device_record(device),
		"precision": engine.precision,
		"precision_flags": meta["precision_flags"],
		"seqs_per_step": meta["seqs_per_step"],
		"n_shuffles": meta["n_shuffles"],
		"head_rows": engine.head_rows,
		"compile": engine.compile,
		"forward_stats": engine.forward_stats,
		"random_state": engine.random_state,
		"random_state_drawn": parameters["random_state"] is None,
		"inputs": {
			"loci": int(len(idxs)),
			"kept": int(len(X)),
			"length": int(X.shape[-1]),
			"window": [int(start), int(end)],
		},
		"run": {key: meta.get(key) for key in ("steps", "padded_peaks",
			"host_syncs", "oom", "references", "reference_stats",
			"seconds")},
		"timings_s": timings,
		"memory": {
			"gpu_max_allocated_bytes":
				torch.cuda.max_memory_allocated(device) if cuda else None,
			"gpu_max_reserved_bytes":
				torch.cuda.max_memory_reserved(device) if cuda else None,
			"step_peak_bytes": meta.get("step_peak_bytes"),
			"step_estimate_bytes": meta.get("step_estimate_bytes"),
			"rss_max_bytes": _max_rss(),
		},
		"deltas": deltas,
		"self_check": fast_run["check"],
		"audit": {"peaks": len(peaks), "pass": None,
			"status": "pending" if len(peaks) else "off (audit 0)"},
		"environment": environment,
		"outputs": {
			"ohe": parameters["ohe_filename"],
			"attr": parameters["attr_filename"],
			"idx": parameters["idx_filename"],
			"deltas": deltas_file,
			"meta": meta_file,
		},
	}

	_write_json(meta_file, record)
	if not len(peaks):
		return

	t = time.perf_counter()
	report = audit(engine, model, X, result, peaks, start, end,
		batch_size=parameters["batch_size"],
		warning_threshold=parameters["warning_threshold"])
	timings["audit"] = time.perf_counter() - t
	record["audit"] = report
	record["finished"] = _now()
	_write_json(meta_file, record)

	summary = audit_summary(report)
	if report["warn"]:
		warnings.warn("The fast engine's audit: the median relative L2 "
			"distance to tangermeme's deep_lift_shap, {:.3g}, is above {:g}."
			.format(report["median_rel_l2"],
			report["bounds"]["warn_median_rel_l2"]), RuntimeWarning,
			stacklevel=3)

	if not report["pass"]:
		failed = [report["indices"][i] for i in report["failed_positions"]]
		raise AuditFailed("The fast engine's audit failed: {}. Sequences "
			"beyond the bounds: {}. The outputs are written, and {} records "
			"the audit.".format(summary, failed, meta_file))

	if parameters["verbose"]:
		print("Fast DeepLIFT/SHAP audit passed: {}.".format(summary))


def _now():
	import datetime

	return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _versions():
	import importlib.metadata as metadata

	import torch

	out = {"python": sys.version.split()[0], "torch.version.cuda":
		torch.version.cuda}
	try:
		out["cudnn"] = torch.backends.cudnn.version()
	except Exception as exc:
		out["cudnn"] = "<{}: {}>".format(type(exc).__name__, exc)

	for package in ("cherimoya", "tangermeme", "torch", "numpy", "numba",
			"triton"):
		try:
			out[package] = metadata.version(package)
		except metadata.PackageNotFoundError:
			out[package] = None

	return out


def _device_record(device):
	import torch

	out = {"device": str(device),
		"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
	if device.type != "cuda":
		return out

	index = device.index if device.index is not None else (
		torch.cuda.current_device())
	props = torch.cuda.get_device_properties(index)
	out.update(name=props.name, total_memory_bytes=props.total_memory,
		capability="{}.{}".format(props.major, props.minor))
	return out


def _max_rss():
	"""This process's peak resident set size in bytes, or None where the
	resource module is missing."""

	try:
		import resource
	except ImportError:
		return None

	rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	return rss if sys.platform == "darwin" else rss * 1024


def _write_json(path, obj):
	import json

	tmp = path + ".tmp"
	with open(tmp, "w") as out:
		out.write(json.dumps(obj, indent=2, default=str) + "\n")

	os.replace(tmp, path)
