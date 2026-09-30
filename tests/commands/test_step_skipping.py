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


def test_pipeline_without_motifs_returns(run_pipeline):
	"""The marginalization guard skips a stage, not the interpreter."""

	assert run_pipeline(motifs=None) is None


def test_pipeline_json_returns_and_writes_the_flags(tmp_path):
	"""`pipeline-json` returns rather than exiting, and the JSON it writes
	holds the flags over the pipeline defaults."""

	from cherimoya_cli.commands import pipeline_json
	from cherimoya_cli.defaults import default_pipeline_parameters

	out = tmp_path / "pipeline.json"
	args = argparse.Namespace(
		sequences="g.fa", peaks=["p.bed"], negatives=None, inputs=["s.bw"],
		controls=None, name="demo", motifs="m.meme", unstranded=True,
		fragments=False, pos_shift=4, neg_shift=-4, paired_end=True,
		scale_factor=2.0, output=str(out))

	assert pipeline_json.run(args) is None
	cfg = json.loads(out.read_text())

	assert (cfg["sequences"], cfg["loci"], cfg["negatives"], cfg["signals"],
		cfg["name"], cfg["motifs"]) == ("g.fa", ["p.bed"], None, ["s.bw"],
		"demo", "m.meme")
	assert {k: cfg["preprocessing_parameters"][k] for k in ("unstranded",
		"pos_shift", "neg_shift", "paired_end", "scale_factor")} == {
		"unstranded": True, "pos_shift": 4, "neg_shift": -4,
		"paired_end": True, "scale_factor": 2.0}
	assert cfg["fit_parameters"] == default_pipeline_parameters["fit_parameters"]


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


def test_pipeline_skip_runs_no_step(tmp_path, run_pipeline):
	"""A top-level `skip` ends the pipeline before any step, including the
	preprocessing and MoDISco commands that have no `skip` of their own."""

	from unittest import mock

	with mock.patch("subprocess.run") as subprocess_run:
		run_pipeline(skip=True, dry_run=False, loci=None)

	subprocess_run.assert_not_called()
	assert not (tmp_path / "demo.fit.json").exists()


def test_pipeline_skips_annotation_when_asked(tmp_path, run_pipeline):
	from unittest import mock

	with mock.patch("subprocess.run") as subprocess_run, \
			mock.patch("cherimoya_cli.commands.fit.run"), \
			mock.patch("cherimoya_cli.commands.attribute.run"), \
			mock.patch("cherimoya_cli.commands.seqlets.run"), \
			mock.patch("cherimoya_cli.commands.marginalize.run"):
		run_pipeline(motifs=str(tmp_path / "m.meme"), dry_run=False,
			annotation_parameters={"skip": True})

	commands = [call.args[0][0] for call in subprocess_run.call_args_list]
	assert "ttl" not in commands
	assert "modisco" in commands
