"""Tests for the keys a hand-written pipeline JSON is allowed to omit.

`cherimoya pipeline-json` emits every key, so the generated path never
exercises what happens when one is missing. A hand-written JSON does,
and the README tells users to write and edit these by hand.
"""

import argparse
import json

import pytest

from cherimoya_cli.defaults import default_pipeline_parameters
from cherimoya_cli.utils import merge_parameters


def _minimal_json(tmp_path, **overrides):
	"""The smallest pipeline JSON that names real inputs, written to
	disk along with the files it points at so the pre-flight check
	passes."""

	for name in ("g.fa", "x.bed", "n.bed", "s.bw", "m.meme"):
		(tmp_path / name).write_text("")

	cfg = {
		"name": "demo",
		"sequences": str(tmp_path / "g.fa"),
		"loci": [str(tmp_path / "x.bed")],
		"negatives": [str(tmp_path / "n.bed")],
		"signals": [str(tmp_path / "s.bw")],
		"dry_run": True,
		"verbose": False,
	}
	cfg.update(overrides)

	path = tmp_path / "p.json"
	with open(path, "w") as f:
		json.dump(cfg, f)
	return path


##


def test_motifs_is_a_pipeline_default():
	"""`pipeline.run` reads `parameters["motifs"]` unguarded in four
	places, so it has to be a declared default or a JSON that omits it
	raises KeyError partway through the run."""

	assert "motifs" in default_pipeline_parameters
	assert default_pipeline_parameters["motifs"] is None


def test_pipeline_json_may_omit_motifs(tmp_path):
	"""Merging a JSON without `motifs` fills it in rather than leaving
	the key absent."""

	merged = merge_parameters(str(_minimal_json(tmp_path)),
		default_pipeline_parameters)
	assert merged["motifs"] is None


def test_pipeline_json_may_omit_model(tmp_path):
	"""`model` defaults to None and `pipeline.run` treats None as 'train
	one', so requiring it in the JSON contradicts how it is used."""

	merged = merge_parameters(str(_minimal_json(tmp_path)),
		default_pipeline_parameters)
	assert merged["model"] is None


def test_pipeline_dry_run_without_motifs_or_model(tmp_path, monkeypatch):
	"""The end-to-end symptom: a hand-written JSON that omits both keys
	must get through the run rather than dying on a missing key.

	The run stops at the marginalization step because `motifs` is None,
	which is `pipeline.run`'s own control flow and not an error, so it
	returns rather than raising.

	`dry_run` still writes each step's JSON, and it writes them relative
	to the working directory, so the test runs from `tmp_path`.
	"""

	from cherimoya_cli.commands import pipeline

	cfg = _minimal_json(tmp_path)
	monkeypatch.chdir(tmp_path)

	assert pipeline.run(argparse.Namespace(parameters=str(cfg))) is None


def test_merge_parameters_error_names_the_null_fix(tmp_path):
	"""A key that really is required still raises, and the message says
	what to write -- `null` is accepted and a missing key is not, which
	is not guessable from 'Must provide value'."""

	path = tmp_path / "p.json"
	with open(path, "w") as f:
		json.dump({}, f)

	with pytest.raises(ValueError, match="null"):
		merge_parameters(str(path), {"sequences": None})
