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


# Every subcommand that loads a model, and so has to carry both knobs.
ALL_DEFAULTS = {
	"attribute": default_attribute_parameters,
	"evaluate": default_evaluate_parameters,
	"marginalize": default_marginalize_parameters,
}


def _captured_load_kwargs(command, tmp_path, **overrides):
	"""Run a subcommand far enough to capture the kwargs it passes to
	`Cherimoya.load`, then abort.

	Starting from the command's own defaults means every key is present,
	so `merge_parameters` is satisfied and `Cherimoya.load` -- the first
	thing each of these commands does after the `skip` guard -- is
	reached without needing real inputs.
	"""

	captured = {}

	class _Stop(Exception):
		pass

	def fake_load(path, **kwargs):
		captured.update(kwargs)
		raise _Stop()

	cfg = dict(ALL_DEFAULTS[command])
	cfg["model"] = "m.torch"
	# `evaluate` reads `signals` before it loads the model, and does not
	# declare it as a default, so it has to be supplied here.
	cfg.setdefault("signals", ["s.bw"])
	cfg.update(overrides)

	path = tmp_path / "{}.json".format(command)
	with open(path, "w") as f:
		json.dump(cfg, f)

	mod = __import__("cherimoya_cli.commands." + command, fromlist=["run"])

	with mock.patch("cherimoya.Cherimoya") as model_cls:
		model_cls.load.side_effect = fake_load
		with pytest.raises(_Stop):
			mod.run(argparse.Namespace(parameters=str(path)))

	return captured


##


@pytest.mark.parametrize("command", sorted(ALL_DEFAULTS))
def test_default_is_compiled(command, tmp_path):
	"""Omitting the keys preserves the pre-existing behaviour: the same
	settings `Cherimoya.load` uses on its own."""

	captured = _captured_load_kwargs(command, tmp_path)

	assert captured["compile"] is True
	assert captured["compile_mode"] == "max-autotune"


@pytest.mark.parametrize("command", sorted(ALL_DEFAULTS))
@pytest.mark.parametrize("compile_flag", [True, False])
def test_forwards_compile(command, compile_flag, tmp_path):
	"""The value in the JSON is what reaches `Cherimoya.load`."""

	captured = _captured_load_kwargs(command, tmp_path,
		compile=compile_flag)

	assert captured["compile"] is compile_flag


@pytest.mark.parametrize("command", sorted(ALL_DEFAULTS))
def test_forwards_compile_mode(command, tmp_path):
	captured = _captured_load_kwargs(command, tmp_path,
		compile_mode="max-autotune-no-cudagraphs")

	assert captured["compile_mode"] == "max-autotune-no-cudagraphs"


def test_pipeline_shares_the_compile_setting(tmp_path, run_pipeline):
	"""Set once at the pipeline top level, the value reaches the
	per-step JSONs, the same way `dtype` and `device` do."""

	from cherimoya_cli.defaults import default_pipeline_parameters

	assert default_pipeline_parameters["compile"] is True
	assert default_pipeline_parameters["compile_mode"] == "max-autotune"

	run_pipeline(compile=False, compile_mode="reduce-overhead")

	with open(tmp_path / "demo.attribute.json") as f:
		emitted = json.load(f)

	assert emitted["compile"] is False
	assert emitted["compile_mode"] == "reduce-overhead"
