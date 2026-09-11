"""Tests for the cherimoya_cli utility helpers."""

import json

import pytest

from cherimoya_cli.utils import _check_set, _extract_set, merge_parameters


# --------- merge_parameters -----------------------------------------------

def test_merge_fills_in_missing_defaults(tmp_path):
	defaults = {'a': 1, 'b': 2, 'c': 3}
	user = {'a': 10}
	path = tmp_path / "p.json"
	path.write_text(json.dumps(user))
	out = merge_parameters(str(path), defaults)
	assert out == {'a': 10, 'b': 2, 'c': 3}


def test_merge_accepts_dict_directly():
	defaults = {'a': 1, 'b': 2}
	out = merge_parameters({'a': 5}, defaults)
	assert out['a'] == 5 and out['b'] == 2


def test_merge_raises_on_missing_required(tmp_path):
	"""Defaults of None are treated as required (with a small whitelist of
	exceptions like 'controls' and 'exclusion_lists')."""

	defaults = {'sequences': None, 'a': 1}
	path = tmp_path / "p.json"
	path.write_text(json.dumps({}))
	with pytest.raises(ValueError, match="sequences"):
		merge_parameters(str(path), defaults)


def test_merge_allows_none_for_unset_parameters(tmp_path):
	"""`controls`, `exclusion_lists`, `early_stopping`, etc. may be None."""

	defaults = {'controls': None, 'exclusion_lists': None}
	path = tmp_path / "p.json"
	path.write_text(json.dumps({}))
	out = merge_parameters(str(path), defaults)
	assert out['controls'] is None
	assert out['exclusion_lists'] is None


def test_merge_raises_on_missing_file():
	with pytest.raises(FileNotFoundError):
		merge_parameters("/no/such/file.json", {'a': 1})


# --------- _check_set ------------------------------------------------------

def test_check_set_only_writes_when_missing():
	d = {'a': 5}
	_check_set(d, 'a', 99)
	_check_set(d, 'b', 99)
	assert d == {'a': 5, 'b': 99}


def test_check_set_treats_none_as_missing():
	d = {'a': None}
	_check_set(d, 'a', 7)
	assert d == {'a': 7}


# --------- _extract_set ----------------------------------------------------

def test_extract_set_combines_top_level_with_subdict_overrides():
	defaults = {'in_window': 100, 'out_window': 50}
	parameters = {
		'in_window': 200,
		'out_window': 80,
		'fit_parameters': {
			'in_window': None,        # falls through to top-level (200)
			'out_window': 40,         # explicit override (40)
		},
	}
	out = _extract_set(parameters, defaults, 'fit_parameters')
	assert out['in_window'] == 200
	assert out['out_window'] == 40
