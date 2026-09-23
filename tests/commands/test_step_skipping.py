"""Tests that a skipped step returns instead of killing the process.

`cherimoya pipeline` calls each subcommand's `run(args)` in-process, so
a `sys.exit()` inside one of them ends the whole pipeline rather than
the step. `"skip": true` is documented as no-opping a step, and the
marginalization guard is documented as skipping a stage when no motif
database is given; both terminated the interpreter instead.
"""

import argparse
import json

import pytest


def _write(tmp_path, name, cfg):
	path = tmp_path / "{}.json".format(name)
	with open(path, "w") as f:
		json.dump(cfg, f)
	return str(path)


def _skip_cfg(defaults, **overrides):
	cfg = dict(defaults)
	cfg["skip"] = True
	cfg.update(overrides)
	return cfg


##


@pytest.mark.parametrize("command", ["evaluate", "attribute", "seqlets",
	"marginalize", "fit"])
def test_skip_returns_rather_than_exiting(tmp_path, command):
	"""Every subcommand that honours `skip` must return, so the caller
	decides what happens next."""

	from cherimoya_cli import defaults as D

	defaults = {
		"evaluate": D.default_evaluate_parameters,
		"attribute": D.default_attribute_parameters,
		"seqlets": D.default_seqlet_parameters,
		"marginalize": D.default_marginalize_parameters,
		"fit": D.default_fit_parameters,
	}[command]

	mod = __import__("cherimoya_cli.commands." + command, fromlist=["run"])
	path = _write(tmp_path, command, _skip_cfg(defaults))

	# Returns None rather than raising SystemExit.
	assert mod.run(argparse.Namespace(parameters=path)) is None


def test_pipeline_without_motifs_returns(tmp_path, monkeypatch):
	"""The marginalization guard skips a stage, not the interpreter."""

	from cherimoya_cli.commands import pipeline

	for name in ("g.fa", "x.bed", "n.bed", "s.bw"):
		(tmp_path / name).write_text("")

	cfg = {
		"name": "demo", "sequences": str(tmp_path / "g.fa"),
		"loci": [str(tmp_path / "x.bed")],
		"negatives": [str(tmp_path / "n.bed")],
		"signals": [str(tmp_path / "s.bw")],
		"controls": None, "model": None, "motifs": None,
		"dry_run": True, "verbose": False, "random_state": 0,
	}
	monkeypatch.chdir(tmp_path)

	assert pipeline.run(argparse.Namespace(
		parameters=_write(tmp_path, "p", cfg))) is None


def test_pipeline_json_returns(tmp_path):
	"""`pipeline-json` ended with a bare `sys.exit()` after writing its
	output, which is indistinguishable from a failure to a caller."""

	from cherimoya_cli.commands import pipeline_json

	out = tmp_path / "pipeline.json"
	args = argparse.Namespace(
		sequences="g.fa", peaks=None, negatives=None, inputs=["s.bw"],
		controls=None, name="demo", motifs=None, unstranded=False,
		fragments=False, pos_shift=0, neg_shift=0, paired_end=False,
		scale_factor=1, output=str(out))

	assert pipeline_json.run(args) is None
	assert out.exists()


##
# pipeline-json argument requirements
##


@pytest.mark.parametrize("missing", ["sequences", "inputs", "name",
	"output"])
def test_pipeline_json_requires_its_four_inputs(missing):
	"""Omitting one used to produce a JSON full of nulls, or a
	`TypeError` from `open(None)`. argparse should say which flag is
	missing instead."""

	from cherimoya_cli.__main__ import _setup_parsers

	argv = ["pipeline-json", "-s", "g.fa", "-i", "s.bw", "-n", "demo",
		"-o", "p.json"]
	flag = {"sequences": "-s", "inputs": "-i", "name": "-n",
		"output": "-o"}[missing]
	i = argv.index(flag)
	del argv[i:i + 2]

	with pytest.raises(SystemExit):
		_setup_parsers().parse_args(argv)


def test_pipeline_json_accepts_all_four():
	from cherimoya_cli.__main__ import _setup_parsers

	args = _setup_parsers().parse_args(["pipeline-json", "-s", "g.fa",
		"-i", "s.bw", "-n", "demo", "-o", "p.json"])

	assert args.sequences == "g.fa"
	assert args.output == "p.json"
