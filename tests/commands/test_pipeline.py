"""Tests for the keys a hand-written pipeline JSON is allowed to omit.

`cherimoya pipeline-json` emits every key, so the generated path never
exercises what happens when one is missing. A hand-written JSON does,
and the README tells users to write and edit these by hand.
"""

import json

import pytest

from cherimoya_cli.defaults import default_pipeline_parameters
from cherimoya_cli.utils import merge_parameters


# The keys a generated JSON always carries and a hand-written one may
# not. Each has a default, so merging has to fill it in rather than
# leaving the key absent for `pipeline.run` to trip over.
OPTIONAL_KEYS = ("motifs", "model", "controls")


##


@pytest.mark.parametrize("key", OPTIONAL_KEYS)
def test_pipeline_json_may_omit_an_optional_key(key, pipeline_json):
	"""`pipeline.run` reads each of these unguarded, so a JSON that
	omits one has to come out of the merge with it set to None rather
	than raising KeyError partway through the run."""

	merged = merge_parameters(str(pipeline_json(omit=OPTIONAL_KEYS)),
		default_pipeline_parameters)

	assert merged[key] is None


def test_pipeline_dry_run_without_the_optional_keys(run_pipeline):
	"""The end-to-end symptom: a hand-written JSON that omits all three
	must get through the run rather than dying on a missing key.

	The run stops at the marginalization step because `motifs` is None,
	which is `pipeline.run`'s own control flow and not an error, so it
	returns rather than raising.
	"""

	assert run_pipeline(omit=OPTIONAL_KEYS) is None


def test_merge_parameters_error_names_the_null_fix(tmp_path):
	"""A key that really is required still raises, and the message says
	what to write -- `null` is accepted and a missing key is not, which
	is not guessable from 'Must provide value'."""

	path = tmp_path / "p.json"
	with open(path, "w") as f:
		json.dump({}, f)

	with pytest.raises(ValueError, match="null"):
		merge_parameters(str(path), {"sequences": None})
