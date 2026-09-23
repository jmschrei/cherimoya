"""Shared fixtures for the `cherimoya_cli` subcommand tests.

Four test files each built their own near-identical pipeline JSON. The
config is the same in all of them; what differs is which optional keys
are set explicitly and which are left out, so that is what the factory
below takes as an argument.
"""

import argparse
import json

import pytest


# Written as empty files so the pipeline's pre-flight path check passes.
PIPELINE_INPUTS = ("g.fa", "x.bed", "n.bed", "s.bw", "m.meme")


@pytest.fixture
def pipeline_json(tmp_path):
	"""Factory writing a pipeline JSON that names real input files, with
	`dry_run` on.

	Keyword arguments override config values. `omit` drops keys from the
	JSON entirely, which is how a hand-written config differs from one
	`cherimoya pipeline-json` emitted and is the case worth testing.
	"""

	def build(omit=(), **overrides):
		for name in PIPELINE_INPUTS:
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
		for key in omit:
			cfg.pop(key, None)

		path = tmp_path / "p.json"
		with open(path, "w") as f:
			json.dump(cfg, f)
		return path

	return build


@pytest.fixture
def run_pipeline(pipeline_json, tmp_path, monkeypatch):
	"""Run `cherimoya pipeline` from `tmp_path` against a config built by
	`pipeline_json`.

	The working directory matters: `dry_run` still writes each step's
	JSON, and it writes them relative to the working directory.
	"""

	def run(**kwargs):
		from cherimoya_cli.commands import pipeline

		path = pipeline_json(**kwargs)
		monkeypatch.chdir(tmp_path)
		return pipeline.run(argparse.Namespace(parameters=str(path)))

	return run
