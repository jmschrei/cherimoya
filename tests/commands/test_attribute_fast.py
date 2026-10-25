"""Tests of `cherimoya attribute`'s ``"fast"`` key on the CPU. With
``"fast": true``, the default, `cherimoya.deep_lift_shap.deep_lift_shap`
takes its fast path for the log-counts, and the command writes the files
that ``"fast": false`` writes.

The inputs: a FASTA of 3 contigs of 3 kb with a run of 20 Ns, and 12 loci,
one of which runs off the end of its contig (`extract_loci` drops it) and
one of which covers the N run (the N filter drops it), so 10 are kept. The
model is a tiny Cherimoya with weights redrawn so that the DeepLIFT rules
matter, saved with model.save(). The JSON attributes 4 shuffles over a 256
bp input and keeps 64 bp.

Every job here would take the default path less than thirty seconds, so the
tests that need the fast path set `_FAST_MIN_SECONDS` to 0 (`forced`).
"""

import argparse
import json

import numpy
import pytest

from pathlib import Path

from cherimoya._deep_lift_shap_engine import Engine
from cherimoya_cli.commands import attribute
from cherimoya_cli.defaults import default_attribute_parameters
from cherimoya_cli.defaults import default_pipeline_parameters

from .. import deep_lift_shap_engine_helpers as tiny


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
	"""The FASTA, the loci, and a tiny model."""

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

	model = root / "tiny.torch"
	tiny.tiny_model("A", seed=0).save(str(model))
	return {"fasta": str(fasta), "loci": str(loci), "model": str(model)}


@pytest.fixture
def forced(monkeypatch):
	monkeypatch.setattr("cherimoya.deep_lift_shap._FAST_MIN_SECONDS", 0.0)


@pytest.fixture
def engines(monkeypatch):
	built = []

	def counting(*args, **kwargs):
		built.append(Engine(*args, **kwargs))
		return built[-1]

	monkeypatch.setattr("cherimoya.deep_lift_shap.Engine", counting)
	return built


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
		"output": "logcounts",
		"algorithm": "deep_lift_shap",
		"verbose": False,
		"ohe_filename": str(out_dir / "{}.ohe.npz".format(name)),
		"attr_filename": str(out_dir / "{}.attr.npz".format(name)),
		"idx_filename": str(out_dir / "{}.idx.npy".format(name)),
	}

	params.update(overrides)
	path = out_dir / "{}.json".format(name)
	path.write_text(json.dumps(params, indent=2))
	return path


def run(path):
	attribute.run(argparse.Namespace(parameters=str(path)))
	p = json.loads(Path(path).read_text())
	out = {}
	for name in ("ohe", "attr"):
		with numpy.load(p["{}_filename".format(name)]) as npz:
			out[name + "_keys"] = sorted(npz.files)
			out[name] = npz["arr_0"]

	out["idx"] = numpy.load(p["idx_filename"])
	return out


def per_locus_rel_l2(a, b):
	a = a.astype(numpy.float64).reshape(len(a), -1)
	b = b.astype(numpy.float64).reshape(len(b), -1)
	return numpy.linalg.norm(a - b, axis=1) / numpy.maximum(
		numpy.linalg.norm(b, axis=1), 1e-30)


##


def test_the_fast_path_writes_the_default_paths_files(inputs, tmp_path, forced,
	engines):
	assert default_attribute_parameters["fast"] is True
	assert default_pipeline_parameters["attribute_parameters"]["fast"] is True
	default = run(write_params(inputs, tmp_path, "default", fast=False))
	assert engines == []
	ours = run(write_params(inputs, tmp_path, "fast"))
	assert len(engines) == 1

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


@pytest.mark.parametrize("value", ["yes", 1])
def test_fast_must_be_a_bool(inputs, tmp_path, value):
	path = write_params(inputs, tmp_path, "bad", model=MISSING_MODEL,
		fast=value)
	with pytest.raises(ValueError, match="`fast` must be true or false"):
		attribute.run(argparse.Namespace(parameters=str(path)))
