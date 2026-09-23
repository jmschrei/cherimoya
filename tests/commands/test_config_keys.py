"""Tests that every declared CLI default is a key something reads.

A default that nothing reads is worse than no default: it appears in
the generated JSON and in the CLI reference, so setting it looks like
it should do something.
"""

import pytest

from cherimoya_cli.defaults import default_marginalize_parameters
from cherimoya_cli.defaults import default_pipeline_parameters


##


def test_pipeline_marginalize_keys_are_read_by_marginalize():
	"""Every key the pipeline declares for the marginalize step must be
	one `cherimoya marginalize` actually has a default for, or
	`_extract_set` will carry it into the step JSON where nothing reads
	it."""

	declared = set(default_pipeline_parameters['marginalize_parameters'])
	known = set(default_marginalize_parameters)

	# `loci`, `motifs` and the shared inference keys are supplied by the
	# pipeline and consumed by marginalize; anything else has to match.
	unknown = declared - known
	assert unknown == set(), (
		"pipeline declares marginalize keys nothing reads: {}"
		.format(sorted(unknown)))


def test_marginalize_output_key_is_output_filename():
	"""The pipeline used to declare `output_folder` while the command
	reads `output_filename`, so setting it was a silent no-op."""

	declared = default_pipeline_parameters['marginalize_parameters']
	assert 'output_filename' in declared
	assert 'output_folder' not in declared


def test_count_loss_weight_is_gone():
	"""Nothing has ever read it: not `fit`, not the model, not the
	loss."""

	import cherimoya_cli.utils as utils
	import inspect

	assert 'count_loss_weight' not in default_pipeline_parameters[
		'fit_parameters']
	assert 'count_loss_weight' not in inspect.getsource(
		utils.merge_parameters)


def test_modisco_report_still_declares_output_folder():
	"""The control: `modisco_report_parameters.output_folder` *is*
	read, by `pipeline.run`, and must not be renamed along with the
	marginalize one."""

	assert 'output_folder' in default_pipeline_parameters[
		'modisco_report_parameters']
