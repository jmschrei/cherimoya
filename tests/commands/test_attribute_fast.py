"""Tests of `cherimoya attribute` with ``"engine": "fast"`` against the same
command with its default engine, on the CPU.

Both engines run on the same synthetic inputs: a FASTA of 3 contigs of 3 kb
with a run of 20 Ns, and 12 loci, one of which runs off the end of its
contig (`extract_loci` drops it) and one of which covers the N run (the N
filter drops it), so 10 are kept. The model is a tiny Cherimoya with
weights redrawn so that the DeepLIFT rules matter, saved with model.save().
The JSON attributes 4 shuffles over a 256 bp input and keeps 64 bp; batches
of 7 split a locus's pairs across the default engine's batches.

The fast engine writes the default engine's three files with the same
names, keys, dtypes and shapes: the one-hot and index files are equal, and
the attributions are within 1e-5 per locus (relative L2). It also writes a
deltas file and a record of the run, and its audit attributes 8 evenly
spaced loci again with tangermeme's `deep_lift_shap`.
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import warnings

import numpy
import pytest

from pathlib import Path

import cherimoya.fast_deep_lift_shap as fast

from cherimoya.fast_deep_lift_shap import AuditFailed
from cherimoya.fast_deep_lift_shap import engine as fdls
from cherimoya.fast_deep_lift_shap import checks
from cherimoya_cli.commands import attribute

from .. import fast_deep_lift_shap_helpers as tiny


CONTIG_LENGTH = 3000
CHROMS = ("chr1", "chr2", "chr3")
N_RUN = ("chr2", 1500, 1520)
IN_WINDOW, ATTR_WINDOW, N_SHUFFLES = 256, 64, 4
LOCI = (
	("chr1", 300), ("chr1", 900), ("chr1", 1600), ("chr1", 2400),
	("chr2", 400), ("chr2", 1000),
	("chr2", 1510),  # its window covers the N run: dropped by the N filter
	("chr2", 2200), ("chr2", 2700),
	("chr3", 500), ("chr3", 1800),
	("chr3", 2950),  # its window runs off the contig: dropped by extract_loci
)
KEPT = numpy.array([True] * 6 + [False] + [True] * 4 + [False])
REL_L2_MAX = 1e-5
MISSING_MODEL = "/nonexistent/cherimoya.torch"


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
	"""The FASTA, the loci, and two tiny models: config A, and config B with
	two groups."""

	root = tmp_path_factory.mktemp("inputs")
	rng = numpy.random.RandomState(0)
	fasta = root / "genome.fa"
	with open(fasta, "w") as out:
		for name in CHROMS:
			seq = list(rng.choice(list("ACGT"), CONTIG_LENGTH))
			if name == N_RUN[0]:
				seq[N_RUN[1]:N_RUN[2]] = "N" * (N_RUN[2] - N_RUN[1])

			text = "".join(seq)
			out.write(">{}\n".format(name))
			out.write("\n".join(text[i:i + 60] for i in range(0, len(text),
				60)) + "\n")

	loci = root / "peaks.narrowPeak"
	with open(loci, "w") as out:
		for i, (chrom, centre) in enumerate(LOCI):
			out.write("{}\t{}\t{}\tpeak{}\t0\t.\t1.0\t1.0\t1.0\t100\n".format(
				chrom, centre - 100, centre + 100, i))

	model_a, model_b = root / "tiny_a.torch", root / "tiny_b.torch"
	tiny.tiny_model("A", seed=0).save(str(model_a))
	tiny.tiny_model("B", seed=1).save(str(model_b))
	return {"fasta": str(fasta), "loci": str(loci), "model": str(model_a),
		"model_b": str(model_b)}


def write_params(inputs, out_dir, name, **overrides):
	"""An attribute JSON for the synthetic inputs, writing
	<out_dir>/<name>.{ohe.npz,attr.npz,idx.npy}."""

	params = {
		"model": inputs["model"],
		"sequences": inputs["fasta"],
		"loci": [inputs["loci"]],
		"chroms": list(CHROMS),
		"in_window": IN_WINDOW,
		"attr_window": ATTR_WINDOW,
		"n_shuffles": N_SHUFFLES,
		"batch_size": 64,
		"random_state": 0,
		"dtype": "float32",
		"device": "cpu",
		"output": "counts",
		"algorithm": "deep_lift_shap",
		"engine": "fast",
		"ref_workers": 0,
		"verbose": False,
		"ohe_filename": str(out_dir / "{}.ohe.npz".format(name)),
		"attr_filename": str(out_dir / "{}.attr.npz".format(name)),
		"idx_filename": str(out_dir / "{}.idx.npy".format(name)),
	}

	params.update(overrides)
	path = out_dir / "{}.json".format(name)
	path.write_text(json.dumps(params, indent=2))
	return path


def _params(path):
	return json.loads(Path(path).read_text())


def load_outputs(path):
	"""The three files a run of the JSON at `path` wrote, and the npz keys."""

	p = _params(path)
	out = {}
	for name in ("ohe", "attr"):
		with numpy.load(p["{}_filename".format(name)]) as npz:
			out[name + "_keys"] = sorted(npz.files)
			out[name] = npz["arr_0"]

	out["idx"] = numpy.load(p["idx_filename"])
	return out


def output_paths(path):
	p = _params(path)
	base = p["attr_filename"][:-len(".npz")]
	names = [p["ohe_filename"], p["attr_filename"], p["idx_filename"],
		base + ".deltas.npy", base + ".meta.json"]
	return [Path(n) for n in names]


def per_locus_rel_l2(a, b):
	a = a.astype(numpy.float64).reshape(len(a), -1)
	b = b.astype(numpy.float64).reshape(len(b), -1)
	return numpy.linalg.norm(a - b, axis=1) / numpy.maximum(
		numpy.linalg.norm(b, axis=1), 1e-30)


def ns(path):
	return argparse.Namespace(parameters=str(path))


def _forbid_the_fast_engine(monkeypatch):
	def forbidden(*args, **kwargs):
		raise AssertionError("the fast engine must not run")

	monkeypatch.setattr(fdls, "Engine", forbidden)
	monkeypatch.setattr(fast, "Engine", forbidden)


@pytest.fixture(scope="module")
def default_runs(inputs, tmp_path_factory):
	"""The default engine's run of the synthetic JSON, once per batch size
	and set of overrides."""

	cache = {}

	def get(batch_size, **overrides):
		key = (batch_size, json.dumps(overrides, sort_keys=True))
		if key not in cache:
			out = tmp_path_factory.mktemp("default{}".format(batch_size))
			overrides.setdefault("engine", "default")
			path = write_params(inputs, out, "default", batch_size=batch_size,
				**overrides)
			attribute.run(ns(path))
			cache[key] = load_outputs(path)

		return cache[key]

	return get


##


@pytest.mark.parametrize("batch_size, ref_workers", [(64, 2), (7, 0)])
def test_the_fast_engine_matches_the_default_engine(inputs, default_runs,
	tmp_path, batch_size, ref_workers):
	default = default_runs(batch_size)
	path = write_params(inputs, tmp_path, "fast", batch_size=batch_size,
		ref_workers=ref_workers)
	attribute.run(ns(path))
	ours = load_outputs(path)

	for name in ("ohe", "attr"):
		assert ours[name + "_keys"] == default[name + "_keys"] == ["arr_0"]

	assert ours["ohe"].dtype == default["ohe"].dtype == numpy.int8
	assert ours["ohe"].shape == (KEPT.sum(), 4, ATTR_WINDOW)
	assert numpy.array_equal(ours["ohe"], default["ohe"])
	assert ours["idx"].dtype == default["idx"].dtype == numpy.bool_
	assert numpy.array_equal(ours["idx"], default["idx"])
	assert numpy.array_equal(ours["idx"], KEPT)
	assert ours["attr"].dtype == default["attr"].dtype == numpy.float32
	assert ours["attr"].shape == default["attr"].shape
	assert per_locus_rel_l2(ours["attr"], default["attr"]).max() <= REL_L2_MAX

	_, _, _, deltas_file, meta_file = output_paths(path)
	deltas = numpy.load(deltas_file)
	assert deltas.dtype == numpy.float32
	assert deltas.shape == (KEPT.sum(), N_SHUFFLES)
	assert numpy.isfinite(deltas).all()

	meta = json.loads(meta_file.read_text())
	assert meta["engine"] == "fast" and meta["precision"] == "tf32"
	assert meta["n_shuffles"] == N_SHUFFLES
	assert meta["head_rows"] == 2 * batch_size
	assert meta["random_state"] == 0 and meta["random_state_drawn"] is False
	assert meta["forward_stats"] == "cpu_mirror" and meta["compile"] == "none"
	assert meta["inputs"] == {"loci": len(LOCI), "kept": int(KEPT.sum()),
		"length": IN_WINDOW, "window": [IN_WINDOW // 2 - ATTR_WINDOW // 2,
		IN_WINDOW // 2 + ATTR_WINDOW // 2]}
	assert meta["run"]["references"] == "produced"
	assert meta["deltas"]["pairs"] == KEPT.sum() * N_SHUFFLES
	assert meta["deltas"]["finite"]

	# On the CPU both sides of the self-check run the model's CPU path.
	check = meta["self_check"]
	assert check["pass"] and check["peaks"] == 4
	assert check["max_abs"] <= 1e-6
	assert check["run_outputs"]["max_abs"] <= 1e-6

	audit = meta["audit"]
	assert audit["pass"] is True and audit["warn"] is False
	assert audit["indices"] == numpy.round(numpy.linspace(0, KEPT.sum() - 1,
		8)).astype(int).tolist()
	assert audit["batch_size"] == min(batch_size, checks.AUDIT_MAX_BATCH)
	assert audit["references_match_tangermeme"] is True
	assert audit["failed_positions"] == []
	assert audit["max_rel_l2"] <= REL_L2_MAX


def test_the_default_engine_is_the_default(inputs, tmp_path, monkeypatch):
	"""Without `engine`, `cherimoya attribute` runs tangermeme's
	deep_lift_shap and never builds the fast engine or its sidecars."""

	_forbid_the_fast_engine(monkeypatch)
	path = write_params(inputs, tmp_path, "plain")
	params = _params(path)
	del params["engine"]
	path.write_text(json.dumps(params))
	attribute.run(ns(path))

	ohe, attr, idx, deltas, meta = output_paths(path)
	assert ohe.exists() and attr.exists() and idx.exists()
	assert not deltas.exists() and not meta.exists()


def test_the_default_engine_ignores_the_fast_options(default_runs):
	plain = default_runs(64)
	options = default_runs(64, precision="fp32", seqs_per_step=3,
		mem_budget_gb=4, ref_workers=0, audit=0)
	for name in ("ohe", "attr", "idx"):
		assert numpy.array_equal(options[name], plain[name])


def test_the_sidecars_follow_the_three_files(inputs, tmp_path, monkeypatch):
	"""ohe, attr and idx are written in the default engine's order, then the
	deltas, then the record of the run."""

	path = write_params(inputs, tmp_path, "order", audit=0)
	calls = []
	real_save, real_savez = numpy.save, numpy.savez_compressed

	def save(file, *args, **kwargs):
		calls.append(("save", str(file)))
		return real_save(file, *args, **kwargs)

	def savez(file, *args, **kwargs):
		calls.append(("savez_compressed", str(file)))
		return real_savez(file, *args, **kwargs)

	monkeypatch.setattr(numpy, "save", save)
	monkeypatch.setattr(numpy, "savez_compressed", savez)
	attribute.run(ns(path))
	ohe, attr, idx, deltas, meta = output_paths(path)
	assert calls == [("savez_compressed", str(ohe)),
		("savez_compressed", str(attr)), ("save", str(idx)),
		("save", str(deltas))]
	assert meta.exists()


##


def test_skip_returns_before_anything_is_loaded(inputs, tmp_path):
	path = write_params(inputs, tmp_path, "skip", skip=True,
		model=MISSING_MODEL)
	assert attribute.run(ns(path)) is None
	assert not any(p.exists() for p in output_paths(path))


def test_profile_falls_back_to_the_default_engine(inputs, default_runs,
	tmp_path, monkeypatch):
	"""The fast engine attributes the count head only: output "profile"
	warns and runs the default engine, whose files it writes, and no
	sidecar."""

	_forbid_the_fast_engine(monkeypatch)
	default = default_runs(64, output="profile")
	path = write_params(inputs, tmp_path, "profile", output="profile",
		precision="fp32")
	with pytest.warns(UserWarning, match="count head only"):
		attribute.run(ns(path))

	ours = load_outputs(path)
	for name in ("ohe", "attr", "idx"):
		assert ours[name].dtype == default[name].dtype
		assert numpy.array_equal(ours[name], default[name]), name

	_, _, _, deltas_file, meta_file = output_paths(path)
	assert not deltas_file.exists() and not meta_file.exists()


def test_saturation_mutagenesis_ignores_the_engine(inputs, tmp_path,
	monkeypatch):
	_forbid_the_fast_engine(monkeypatch)
	path = write_params(inputs, tmp_path, "ism",
		algorithm="saturation_mutagenesis")
	with pytest.warns(UserWarning, match="DeepLIFT/SHAP only"):
		attribute.run(ns(path))

	assert load_outputs(path)["attr"].shape == (KEPT.sum(), 4, ATTR_WINDOW)


@pytest.mark.parametrize("key, value, message", [
	("engine", "faster", "engine must be"),
	("precision", "bf16", "precision"),
	("precision", "reference", "precision"),
	("seqs_per_step", 0, "seqs_per_step"),
	("seqs_per_step", "4", "seqs_per_step"),
	("seqs_per_step", True, "seqs_per_step"),
	("mem_budget_gb", -1, "mem_budget_gb"),
	("mem_budget_gb", "8", "mem_budget_gb"),
	("ref_workers", -1, "ref_workers"),
	("ref_workers", 1.5, "ref_workers"),
	("audit", -1, "audit"),
	("audit", True, "audit"),
	("dtype", "bfloat16", "dtype"),
	("dtype", "float16", "dtype"),
	("dtype", "float64", "dtype"),
])
def test_bad_options_are_refused_before_the_model_is_loaded(inputs, tmp_path,
	key, value, message):
	path = write_params(inputs, tmp_path, "bad", model=MISSING_MODEL,
		**{key: value})
	with pytest.raises(ValueError, match=message):
		attribute.run(ns(path))


def test_a_multi_group_model_without_a_group_raises_the_default_error(
	inputs, tmp_path):
	ours = write_params(inputs, tmp_path, "ours", model=inputs["model_b"],
		group=None)
	default = write_params(inputs, tmp_path, "default",
		model=inputs["model_b"], group=None, engine="default")
	with pytest.raises(ValueError, match="attributes one output") as fast:
		attribute.run(ns(ours))

	with pytest.raises(ValueError, match="attributes one output") as stock:
		attribute.run(ns(default))

	assert str(fast.value) == str(stock.value)


##


@pytest.mark.parametrize("threshold, expected", [(1e-12, 1), (None, 0)])
def test_one_summary_warning_or_none(inputs, tmp_path, threshold, expected):
	path = write_params(inputs, tmp_path, "warn", warning_threshold=threshold,
		audit=0)
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		attribute.run(ns(path))

	found = [w for w in caught if issubclass(w.category, RuntimeWarning)
		and str(w.message).startswith("Convergence deltas too high")]
	assert len(found) == expected
	meta = json.loads(output_paths(path)[-1].read_text())
	if threshold is None:
		assert meta["deltas"]["above"] is None
	else:
		assert meta["deltas"]["above"] >= 1


def test_random_state_null_records_the_drawn_seed(inputs, tmp_path):
	drawn = write_params(inputs, tmp_path, "drawn", random_state=None, audit=0)
	attribute.run(ns(drawn))
	meta = json.loads(output_paths(drawn)[-1].read_text())
	seed = meta["random_state"]
	assert isinstance(seed, int) and meta["random_state_drawn"] is True

	again = write_params(inputs, tmp_path, "again", random_state=seed, audit=0)
	attribute.run(ns(again))
	assert numpy.array_equal(load_outputs(again)["attr"],
		load_outputs(drawn)["attr"])


def test_a_failed_self_check_aborts_before_anything_is_written(inputs,
	tmp_path, monkeypatch):
	"""Weights that are not the model's (the count Linear scaled by 1.05)
	fail the self-check, before the run and before any file."""

	original = fdls.CheriWeights.from_model

	def perturbed(cls, model, device=None):
		weights = original(model, device)
		weights.Wl.mul_(1.05)
		return weights

	monkeypatch.setattr(fdls.CheriWeights, "from_model",
		classmethod(perturbed))
	runs = []
	monkeypatch.setattr(fdls.Engine, "run", lambda self, *args, **kwargs:
		runs.append(args))
	path = write_params(inputs, tmp_path, "selfcheck")
	with pytest.raises(RuntimeError, match="self-check failed"):
		attribute.run(ns(path))

	assert not runs
	assert not any(p.exists() for p in output_paths(path))


##


def _corrupting_run(monkeypatch, corrupt):
	"""Make Engine.run return its result with corrupt(attr) applied."""

	real_run = fdls.Engine.run
	calls = []

	def run(self, *args, **kwargs):
		result = real_run(self, *args, **kwargs)
		calls.append(kwargs.get("keep_references"))
		return result._replace(attr=corrupt(result.attr.clone()))

	monkeypatch.setattr(fdls.Engine, "run", run)
	return calls


def _scale_one_locus(attr):
	attr[3] *= 1.05  # relative L2 0.05, Pearson still 1
	return attr


def _swap_two_loci(attr):
	attr[[3, 7]] = attr[[7, 3]]  # Pearson far below 0.9999
	return attr


@pytest.mark.parametrize("corrupt", [_scale_one_locus, _swap_two_loci])
def test_a_wrong_engine_output_fails_the_audit_after_writing(inputs, tmp_path,
	monkeypatch, corrupt):
	calls = _corrupting_run(monkeypatch, corrupt)
	path = write_params(inputs, tmp_path, "corrupt")
	with pytest.raises(AuditFailed, match="audit failed"):
		attribute.run(ns(path))

	assert all(p.exists() for p in output_paths(path))
	audit = json.loads(output_paths(path)[-1].read_text())["audit"]
	peaks = numpy.round(numpy.linspace(0, KEPT.sum() - 1, 8)).astype(int)
	assert len(calls) == 1 and numpy.array_equal(calls[0], peaks)
	bad = [peaks.tolist().index(i) for i in (3, 7) if i in peaks
		and (corrupt is _swap_two_loci or i == 3)]
	assert audit["pass"] is False and audit["failed_positions"] == bad
	assert audit["references_match_tangermeme"] is True
	if corrupt is _scale_one_locus:
		assert audit["rel_l2"][bad[0]] == pytest.approx(0.05, rel=1e-3)
		assert audit["min_pearson"] > 0.99999
	else:
		assert max(audit["pearson"][i] for i in bad) < 0.9999


def test_references_unlike_tangermemes_fail_the_audit(inputs, tmp_path,
	monkeypatch):
	"""If the engine's references for the audited loci were not tangermeme's
	draw, the audit fails although the stock run, given the same
	references, agrees with the engine."""

	real = checks.tangermeme_references

	def shifted(X, peaks, n_shuffles, random_state):
		return real(X, peaks, n_shuffles, random_state + 1)

	monkeypatch.setattr(checks, "tangermeme_references", shifted)
	path = write_params(inputs, tmp_path, "refs")
	with pytest.raises(AuditFailed, match="NOT equal"):
		attribute.run(ns(path))

	audit = json.loads(output_paths(path)[-1].read_text())["audit"]
	assert audit["references_match_tangermeme"] is False
	assert audit["failed_positions"] == []
	assert audit["references_mismatched_positions"] == list(range(8))


def test_audit_zero_runs_no_audit(inputs, tmp_path, monkeypatch):
	def forbidden(*args, **kwargs):
		raise AssertionError("audit 0 must not audit")

	monkeypatch.setattr(checks, "audit", forbidden)
	monkeypatch.setattr(fast, "audit", forbidden)
	calls = _corrupting_run(monkeypatch, lambda attr: attr)
	path = write_params(inputs, tmp_path, "noaudit", audit=0)
	attribute.run(ns(path))
	meta = json.loads(output_paths(path)[-1].read_text())
	assert meta["audit"] == {"peaks": 0, "pass": None,
		"status": "off (audit 0)"}
	assert calls == [None]


##


def test_the_cache_defaults_are_set_only_when_unset(tmp_path, monkeypatch):
	import tempfile

	monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
	monkeypatch.delenv("TRITON_CACHE_AUTOTUNING", raising=False)
	monkeypatch.setenv("TMPDIR", str(tmp_path))
	monkeypatch.setattr(tempfile, "tempdir", None)
	record = attribute._set_cache_defaults()
	assert os.environ["NUMBA_CACHE_DIR"].startswith(str(tmp_path))
	assert os.path.isdir(os.environ["NUMBA_CACHE_DIR"])
	assert os.environ["TRITON_CACHE_AUTOTUNING"] == "1"
	assert record["set_by_cherimoya"] == ["NUMBA_CACHE_DIR",
		"TRITON_CACHE_AUTOTUNING"]

	monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "mine"))
	monkeypatch.setenv("TRITON_CACHE_AUTOTUNING", "0")
	record = attribute._set_cache_defaults()
	assert os.environ["NUMBA_CACHE_DIR"] == str(tmp_path / "mine")
	assert os.environ["TRITON_CACHE_AUTOTUNING"] == "0"
	assert record["set_by_cherimoya"] == []


def test_the_cache_defaults_are_set_before_cherimoya_is_imported(tmp_path):
	"""In a fresh process, `cherimoya attribute` with engine "fast" sets the
	two variables before it imports cherimoya and tangermeme, so
	tangermeme's kernel is cached under NUMBA_CACHE_DIR and cherimoya's
	autotuned kernels keep their timings. Importing the command module
	alone imports neither."""

	params = {"model": MISSING_MODEL, "sequences": "x.fa", "loci": "x.bed",
		"engine": "fast", "device": "cpu"}
	path = tmp_path / "p.json"
	path.write_text(json.dumps(params))
	code = textwrap.dedent("""
		import argparse, os, sys
		from cherimoya_cli.commands import attribute
		heavy = sorted(m for m in ("torch", "cherimoya", "tangermeme")
			if m in sys.modules)
		assert not heavy, heavy
		try:
			attribute.run(argparse.Namespace(parameters={!r}))
		except Exception:
			pass
		from tangermeme import ersatz
		import cherimoya.cheri as cheri
		print(os.environ["NUMBA_CACHE_DIR"])
		print(ersatz._fast_shuffle._cache.cache_path)
		print(os.environ["TRITON_CACHE_AUTOTUNING"])
		kernel = getattr(cheri, "_fwd_stats_kernel", None)
		print(getattr(kernel, "cache_results", None))
		""".format(str(path)))
	env = {k: v for k, v in os.environ.items()
		if k not in ("NUMBA_CACHE_DIR", "TRITON_CACHE_AUTOTUNING")}
	env["TMPDIR"] = str(tmp_path)
	done = subprocess.run([sys.executable, "-c", code], env=env,
		capture_output=True, text=True, timeout=600, cwd=str(tmp_path))
	assert done.returncode == 0, done.stderr[-3000:]

	numba_dir, kernel_cache, autotuning, cached = \
		done.stdout.strip().splitlines()[-4:]
	assert numba_dir.startswith(str(tmp_path))
	assert Path(kernel_cache).resolve().is_relative_to(
		Path(numba_dir).resolve())
	assert autotuning == "1"
	assert cached in ("True", "None")
