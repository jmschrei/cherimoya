"""Tests that every declared CLI default is a key something reads.

A default that nothing reads is worse than no default: it appears in the
generated JSON and in the CLI reference, so setting it looks like it
should do something. Two have already shipped that way --
`marginalize_parameters.output_folder`, which the command spells
`output_filename`, and `fit_parameters.count_loss_weight`, which nothing
has ever read -- so the invariant is pinned for every step rather than
for the two that happened to be found.
"""

import pytest

from cherimoya_cli import defaults as D


# `_extract_set` carries a declared key into the step JSON whether or
# not the command reads it, which is what lets the two drift apart.
# These are exactly the five `_extract_set` calls in `pipeline.run`,
# each paired with the defaults dict that call passes.
FORWARDED_STEPS = {
	"fit_parameters": D.default_fit_parameters,
	"attribute_parameters": D.default_attribute_parameters,
	"seqlet_parameters": D.default_seqlet_parameters,
	"annotation_parameters": D.default_annotation_parameters,
	"marginalize_parameters": D.default_marginalize_parameters,
}


##


@pytest.mark.parametrize("step", sorted(FORWARDED_STEPS))
def test_every_declared_step_key_is_read_by_its_command(step):
	"""Every key the pipeline declares for a step must be one the
	subcommand has a default for, or it lands in the step JSON where
	nothing reads it."""

	declared = set(D.default_pipeline_parameters[step])
	unknown = declared - set(FORWARDED_STEPS[step])

	assert unknown == set(), (
		"pipeline declares {} keys nothing reads: {}"
		.format(step, sorted(unknown)))


def test_marginalize_output_key_is_declared():
	"""The other half of the rename: dropping the key entirely would
	leave the subset check above passing while the output path became
	unsettable from a pipeline JSON."""

	assert "output_filename" in D.default_pipeline_parameters[
		"marginalize_parameters"]
