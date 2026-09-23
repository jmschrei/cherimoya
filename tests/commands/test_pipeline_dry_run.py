"""Tests that `dry_run` emits the per-step JSONs without running anything.

`dry_run` is how a pipeline config is checked before committing hours of
GPU time to it, so it has to complete on exactly the configurations a
real run would use -- including the one with a motif database, which is
the common case and the only one that reaches the annotation step.
"""

import argparse
import json

import pytest


def _write_json(tmp_path, **overrides):
	"""A pipeline JSON naming real (empty) input files, with `dry_run`
	on."""

	for name in ("g.fa", "x.bed", "n.bed", "s.bw", "m.meme"):
		(tmp_path / name).write_text("")

	cfg = {
		"name": "demo",
		"sequences": str(tmp_path / "g.fa"),
		"loci": [str(tmp_path / "x.bed")],
		"negatives": [str(tmp_path / "n.bed")],
		"signals": [str(tmp_path / "s.bw")],
		"controls": None,
		"model": None,
		"motifs": None,
		"dry_run": True,
		"verbose": False,
		"random_state": 0,
	}
	cfg.update(overrides)

	path = tmp_path / "p.json"
	with open(path, "w") as f:
		json.dump(cfg, f)
	return path


def _run(tmp_path, monkeypatch, **overrides):
	"""Run the pipeline from `tmp_path`, since `dry_run` still writes
	each step's JSON relative to the working directory."""

	from cherimoya_cli.commands import pipeline

	cfg = _write_json(tmp_path, **overrides)
	monkeypatch.chdir(tmp_path)
	pipeline.run(argparse.Namespace(parameters=str(cfg)))


##


def test_dry_run_with_motifs_completes(tmp_path, monkeypatch):
	"""With a motif database set, the run reaches the end. The seqlet
	annotation step used to read its own output with `pandas.read_csv`
	outside the `dry_run` guard, so it raised FileNotFoundError on a
	file no subprocess had been allowed to write."""

	_run(tmp_path, monkeypatch, motifs=str(tmp_path / "m.meme"))


def test_dry_run_with_motifs_writes_no_annotation_outputs(tmp_path,
		monkeypatch):
	"""A dry run emits the step JSONs and nothing else -- in particular
	not the tomtom-lite outputs, which would otherwise be empty files
	standing in for real results."""

	_run(tmp_path, monkeypatch, motifs=str(tmp_path / "m.meme"))

	assert not (tmp_path / "demo.seqlets_annotated.bed").exists()
	assert not (tmp_path / "demo.motif_seqlet_count.tsv").exists()


def test_dry_run_with_motifs_writes_the_step_jsons(tmp_path, monkeypatch):
	"""The point of a dry run: every per-step JSON lands on disk so the
	config can be inspected."""

	_run(tmp_path, monkeypatch, motifs=str(tmp_path / "m.meme"))

	for name in ("demo.fit.json", "demo.attribute.json",
			"demo.seqlets.json", "demo.marginalize.json"):
		assert (tmp_path / name).exists(), name


# Not covered here: a dry run with `motifs` unset. That configuration
# reaches `sys.exit()` at the marginalization guard, which terminates
# the interpreter rather than returning, and is a separate defect from
# the one this file pins.
