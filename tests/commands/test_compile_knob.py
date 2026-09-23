"""Tests that the inference subcommands expose `torch.compile` settings.

`Cherimoya.load` compiles the forward with ``mode='max-autotune'`` by
default, and the troubleshooting page and the bundled skill both tell
users to pass ``compile=False`` when they hit a CUDA-graph error or want
to attribute a model. Without a JSON key for it there was no way to act
on that advice from the CLI.
"""

import argparse
import json
from unittest import mock

import pytest

from cherimoya_cli.defaults import default_attribute_parameters
from cherimoya_cli.defaults import default_evaluate_parameters
from cherimoya_cli.defaults import default_marginalize_parameters


ALL_DEFAULTS = {
	"attribute": default_attribute_parameters,
	"evaluate": default_evaluate_parameters,
	"marginalize": default_marginalize_parameters,
}


##


@pytest.mark.parametrize("name", sorted(ALL_DEFAULTS))
def test_compile_keys_are_declared(name):
	"""Both knobs exist on every subcommand that loads a model, with the
	same defaults `Cherimoya.load` uses, so nothing changes for a JSON
	that does not set them."""

	defaults = ALL_DEFAULTS[name]
	assert defaults["compile"] is True
	assert defaults["compile_mode"] == "max-autotune"


def _captured_load_kwargs(command, cfg, tmp_path):
	"""Run a subcommand far enough to capture the kwargs it passes to
	`Cherimoya.load`, then abort."""

	captured = {}

	class _Stop(Exception):
		pass

	def fake_load(path, **kwargs):
		captured.update(kwargs)
		raise _Stop()

	path = tmp_path / "{}.json".format(command)
	with open(path, "w") as f:
		json.dump(cfg, f)

	mod = __import__("cherimoya_cli.commands." + command, fromlist=["run"])

	with mock.patch("cherimoya.Cherimoya") as model_cls:
		model_cls.load.side_effect = fake_load
		with pytest.raises(_Stop):
			mod.run(argparse.Namespace(parameters=str(path)))

	return captured


def _evaluate_cfg(tmp_path, **overrides):
	cfg = dict(default_evaluate_parameters)
	cfg.update({
		"sequences": "f.fa", "loci": "f.bed", "signals": ["s.bw"],
		"model": "m.torch", "device": "cpu",
		"performance_filename": str(tmp_path / "p.tsv"),
	})
	cfg.update(overrides)
	return cfg


@pytest.mark.parametrize("compile_flag", [True, False])
def test_evaluate_forwards_compile(tmp_path, compile_flag):
	"""The value in the JSON is what reaches `Cherimoya.load`."""

	captured = _captured_load_kwargs("evaluate",
		_evaluate_cfg(tmp_path, compile=compile_flag), tmp_path)

	assert captured["compile"] is compile_flag


def test_evaluate_forwards_compile_mode(tmp_path):
	captured = _captured_load_kwargs("evaluate",
		_evaluate_cfg(tmp_path, compile_mode="max-autotune-no-cudagraphs"),
		tmp_path)

	assert captured["compile_mode"] == "max-autotune-no-cudagraphs"


def test_evaluate_default_is_compiled(tmp_path):
	"""Omitting the keys preserves the pre-existing behaviour."""

	captured = _captured_load_kwargs("evaluate", _evaluate_cfg(tmp_path),
		tmp_path)

	assert captured["compile"] is True
	assert captured["compile_mode"] == "max-autotune"


def test_pipeline_shares_the_compile_setting(tmp_path, monkeypatch):
	"""Set once at the pipeline top level, the value reaches the
	per-step JSONs, the same way `dtype` and `device` do."""

	from cherimoya_cli.commands import pipeline
	from cherimoya_cli.defaults import default_pipeline_parameters

	assert default_pipeline_parameters["compile"] is True
	assert default_pipeline_parameters["compile_mode"] == "max-autotune"

	for name in ("g.fa", "x.bed", "n.bed", "s.bw"):
		(tmp_path / name).write_text("")

	cfg = {
		"name": "demo", "sequences": str(tmp_path / "g.fa"),
		"loci": [str(tmp_path / "x.bed")],
		"negatives": [str(tmp_path / "n.bed")],
		"signals": [str(tmp_path / "s.bw")],
		"controls": None, "model": None, "motifs": None,
		"dry_run": True, "verbose": False, "random_state": 0,
		"compile": False, "compile_mode": "reduce-overhead",
	}
	path = tmp_path / "p.json"
	with open(path, "w") as f:
		json.dump(cfg, f)

	# `dry_run` writes each step's JSON relative to the working
	# directory, and stops at the marginalization guard because `motifs`
	# is None.
	monkeypatch.chdir(tmp_path)
	pipeline.run(argparse.Namespace(parameters=str(path)))

	with open(tmp_path / "demo.attribute.json") as f:
		emitted = json.load(f)

	assert emitted["compile"] is False
	assert emitted["compile_mode"] == "reduce-overhead"
