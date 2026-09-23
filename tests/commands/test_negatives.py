"""Wiring tests for `cherimoya negatives`.

The GC matching itself is tangermeme's; what is Cherimoya's is the
argparse surface and the forwarding, which is what these cover.
"""

import argparse
from unittest import mock

import pandas
import pytest

from cherimoya_cli.__main__ import _setup_parsers


def _args(**overrides):
	base = dict(peaks="peaks.bed", fasta="g.fa", bigwig=None,
		output="out.bed", bin_width=0.02, max_n_perc=0.1, beta=0.5,
		in_window=2114, out_window=1000, verbose=False)
	base.update(overrides)
	return argparse.Namespace(**base)


def _captured(args, tmp_path):
	from cherimoya_cli.commands import negatives

	captured = {}
	frame = pandas.DataFrame([["chr1", 100, 200]])

	def fake_match(**kwargs):
		captured.update(kwargs)
		return frame

	args.output = str(tmp_path / "out.bed")
	with mock.patch("tangermeme.match.extract_matching_loci",
			side_effect=fake_match):
		negatives.run(args)

	captured["_written"] = pandas.read_csv(args.output, sep="\t",
		header=None)
	return captured


##


def test_negatives_forwards_every_flag(tmp_path):
	"""Each CLI flag has to reach `extract_matching_loci` under the name
	that function expects, which is not the flag's own name for most of
	them."""

	captured = _captured(_args(bin_width=0.05, max_n_perc=0.2, beta=0.7,
		in_window=1000, out_window=500), tmp_path)

	assert captured["gc_bin_width"] == 0.05
	assert captured["max_n_perc"] == 0.2
	assert captured["signal_beta"] == 0.7
	assert captured["in_window"] == 1000
	assert captured["out_window"] == 500


def test_negatives_writes_a_headerless_bed(tmp_path):
	"""The output feeds straight into `fit` as a locus file, so it must
	be headerless and tab separated."""

	captured = _captured(_args(), tmp_path)

	assert list(captured["_written"].iloc[0]) == ["chr1", 100, 200]
	assert len(captured["_written"].columns) == 3


def test_negatives_bigwig_is_optional(tmp_path):
	"""`--bigwig` sets a minimum-counts threshold; without it the
	threshold is off rather than the call failing."""

	captured = _captured(_args(bigwig=None), tmp_path)

	assert captured["bigwig"] is None


def test_negatives_requires_peaks_and_output():
	"""Both are marked required in the parser, so argparse rejects a
	call without them rather than the command failing later."""

	parser = _setup_parsers()
	with pytest.raises(SystemExit):
		parser.parse_args(["negatives", "-f", "g.fa"])


def test_negatives_parser_defaults_match_the_documented_ones():
	"""The flags carry their defaults in argparse rather than in a JSON,
	so this is the only place they can drift from the CLI reference."""

	args = _setup_parsers().parse_args(
		["negatives", "-i", "peaks.bed", "-o", "out.bed"])

	assert args.bin_width == 0.02
	assert args.max_n_perc == 0.1
	assert args.beta == 0.5
	assert args.in_window == 2114
	assert args.out_window == 1000
	assert args.verbose is False
