"""Tests for the cherimoya seqlets CLI's genomic coordinate conversion.

`attribute` writes attributions over a slice centred inside the
extraction window, not over the whole window, so the offset that turns
a within-slice seqlet position into a genomic one is the slice's own
width. These tests pin that conversion by stubbing
`recursive_seqlets` with a seqlet at a known position and checking
where it lands on the genome.
"""

import argparse
import json
from unittest import mock

import numpy
import pandas
import pytest


# The window `attribute` extracts, and the centred slice it actually
# attributes. `mid` and `attr_start` mirror the arithmetic in
# `cherimoya_cli.commands.attribute`.
IN_WINDOW = 2114
ATTR_WIDTH = 400
MID = IN_WINDOW // 2
ATTR_START = MID - ATTR_WIDTH // 2

# One locus, so the expected genomic coordinates are easy to state.
LOCUS_CHROM = "chr1"
LOCUS_START = 10000
LOCUS_END = 11000
LOCUS_MID = (LOCUS_START + LOCUS_END) // 2


def _write_inputs(tmp_path, n_loci=1, width=ATTR_WIDTH):
	"""Write the three files `attribute` hands to `seqlets`, plus the
	locus BED they index into. The arrays are `width` wide because
	that is what `attribute` saves -- the centred slice, not the whole
	extraction window."""

	bed = tmp_path / "loci.bed"
	rows = [[LOCUS_CHROM, LOCUS_START + i * 5000, LOCUS_END + i * 5000]
		for i in range(n_loci)]
	pandas.DataFrame(rows).to_csv(bed, sep="\t", header=False, index=False)

	rng = numpy.random.RandomState(0)
	X = numpy.zeros((n_loci, 4, width), dtype=numpy.float32)
	for i in range(n_loci):
		X[i, rng.randint(0, 4, width), numpy.arange(width)] = 1

	ohe = tmp_path / "a.ohe.npz"
	attr = tmp_path / "a.attr.npz"
	idx = tmp_path / "a.idx.npy"

	numpy.savez_compressed(ohe, X)
	numpy.savez_compressed(attr, rng.randn(n_loci, 4, width).astype("float32"))
	numpy.save(idx, numpy.ones(n_loci, dtype=bool))

	return bed, ohe, attr, idx


def _run_seqlets(tmp_path, seqlet_frame, n_loci=1, width=ATTR_WIDTH):
	"""Run the seqlets command with `recursive_seqlets` stubbed out to
	return `seqlet_frame`, and return the emitted BED as a DataFrame."""

	from cherimoya_cli.commands import seqlets

	bed, ohe, attr, idx = _write_inputs(tmp_path, n_loci, width)
	out = tmp_path / "seqlets.bed"

	cfg = {
		"threshold": 0.01,
		"min_seqlet_len": 4,
		"max_seqlet_len": 25,
		"additional_flanks": 0,
		"chroms": [LOCUS_CHROM],
		"exclusion_lists": None,
		"verbose": False,
		"loci": str(bed),
		"ohe_filename": str(ohe),
		"attr_filename": str(attr),
		"idx_filename": str(idx),
		"output_filename": str(out),
		"skip": False,
	}
	cfg_path = tmp_path / "seqlets.json"
	with open(cfg_path, "w") as f:
		json.dump(cfg, f)

	with mock.patch("tangermeme.seqlet.recursive_seqlets",
		return_value=seqlet_frame):
		seqlets.run(argparse.Namespace(parameters=str(cfg_path)))

	if out.stat().st_size == 0:
		return pandas.DataFrame(columns=[0, 1, 2])
	return pandas.read_csv(out, sep="\t", header=None)


##


def test_seqlet_coords_are_offset_by_the_attribution_slice(tmp_path):
	"""A seqlet at positions 100-110 of the attribution slice must land
	at the genomic bases those positions actually cover.

	The slice is centred in the extraction window, so its first base is
	`LOCUS_MID - IN_WINDOW // 2 + ATTR_START` and position 100 of the
	slice is 100 bases past that. Converting with the full extraction
	window instead puts every seqlet `ATTR_START` bases too far left,
	which is what this pins.
	"""

	frame = pandas.DataFrame({
		"example_idx": [0], "start": [100], "end": [110],
		"attribution": [1.0], "p-value": [0.0]})

	out = _run_seqlets(tmp_path, frame)

	slice_start = LOCUS_MID - IN_WINDOW // 2 + ATTR_START
	assert out.iloc[0, 0] == LOCUS_CHROM
	assert out.iloc[0, 1] == slice_start + 100
	assert out.iloc[0, 2] == slice_start + 110


def test_seqlet_coords_fall_inside_the_source_peak(tmp_path):
	"""Every seqlet comes from a 400bp slice centred on the peak, so it
	cannot land outside the peak. This is the symptom a user sees when
	the offset is wrong, stated as its own assertion so it survives a
	change to the exact window arithmetic."""

	frame = pandas.DataFrame({
		"example_idx": [0, 0], "start": [0, 390], "end": [10, 400],
		"attribution": [1.0, 0.5], "p-value": [0.0, 0.0]})

	out = _run_seqlets(tmp_path, frame)

	assert (out.iloc[:, 1] >= LOCUS_START).all()
	assert (out.iloc[:, 2] <= LOCUS_END).all()


def test_seqlet_coords_track_a_different_attribution_width(tmp_path):
	"""The offset is read off the attribution array rather than assumed,
	so a run that attributed a wider slice converts correctly too."""

	width = 600
	frame = pandas.DataFrame({
		"example_idx": [0], "start": [50], "end": [60],
		"attribution": [1.0], "p-value": [0.0]})

	out = _run_seqlets(tmp_path, frame, width=width)

	slice_start = LOCUS_MID - width // 2
	assert out.iloc[0, 1] == slice_start + 50
	assert out.iloc[0, 2] == slice_start + 60


def test_seqlet_coords_for_a_second_locus(tmp_path):
	"""`example_idx` indexes the locus table, so the second example must
	resolve against the second locus rather than the first."""

	frame = pandas.DataFrame({
		"example_idx": [1], "start": [100], "end": [110],
		"attribution": [1.0], "p-value": [0.0]})

	out = _run_seqlets(tmp_path, frame, n_loci=2)

	locus_mid = (LOCUS_START + 5000 + LOCUS_END + 5000) // 2
	slice_start = locus_mid - IN_WINDOW // 2 + ATTR_START
	assert out.iloc[0, 1] == slice_start + 100
