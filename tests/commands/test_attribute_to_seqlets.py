"""Tests the `attribute` -> `seqlets` seam, end to end.

`attribute` saves a slice centred inside its extraction window;
`seqlets` turns a position in that saved array back into a genomic
coordinate by assuming it is centred on the locus midpoint. Neither
command can check that on its own, and the two are tested separately:
`test_attribute.py` pins the slice arithmetic against literal offsets
and `test_seqlets.py` re-derives the same offsets as module constants.
Change the centring in `attribute` and both files still pass while the
emitted BED is wrong again, which is what shipped as the 857bp shift.

So this runs the real `attribute`, hands its real output files to the
real `seqlets`, and checks the coordinate that comes out. Only the two
tangermeme calls that need genome files or a model are stubbed.
"""

import argparse
import json
from unittest import mock

import numpy
import pandas
import pytest
import torch

from cherimoya_cli.defaults import default_attribute_parameters
from cherimoya_cli.defaults import default_seqlet_parameters


LOCUS_CHROM = "chr1"
LOCUS_START = 10000
LOCUS_END = 11000
LOCUS_MID = (LOCUS_START + LOCUS_END) // 2

# Extraction window and attributed slice width. The pair is varied so a
# hard-coded offset in either command cannot satisfy all of them.
WINDOWS = [(2114, 400), (2114, 600), (4096, 400), (1000, 1000)]


def _run_attribute(tmp_path, in_window, attr_window):
	"""Run `cherimoya attribute` with `extract_loci` and saturation
	mutagenesis stubbed, leaving its own slice arithmetic real.

	Returns the sequence the stub stood the genome up as, plus the
	genomic coordinate of its first column, so the test has a ground
	truth that does not come from re-running either command's
	arithmetic.
	"""

	from cherimoya_cli.commands import attribute

	captured = {}

	def fake_extract(**kwargs):
		# A random one-hot, so no two sub-slices of it are equal and the
		# offset `attribute` sliced at can be recovered from what it
		# saved rather than recomputed the way the command computes it.
		w = kwargs["in_window"]
		rng = numpy.random.RandomState(0)
		X = torch.zeros(1, 4, w)
		X[0, rng.randint(0, 4, w), numpy.arange(w)] = 1.0

		captured["X"] = X
		# `extract_loci` centres the window on the locus midpoint, which
		# makes this the genomic coordinate of column 0.
		captured["window_start"] = LOCUS_MID - w // 2
		return X, torch.ones(1, dtype=bool)

	def fake_sm(model, X, **kwargs):
		width = kwargs["end"] - kwargs["start"]
		return torch.ones(X.shape[0], 4, width)

	cfg = dict(default_attribute_parameters)
	cfg.update({
		"sequences": "g.fa", "loci": str(tmp_path / "loci.bed"),
		"model": "m.torch", "device": "cpu", "verbose": False,
		"in_window": in_window, "attr_window": attr_window,
		"ohe_filename": str(tmp_path / "a.ohe.npz"),
		"attr_filename": str(tmp_path / "a.attr.npz"),
		"idx_filename": str(tmp_path / "a.idx.npy"),
	})

	path = tmp_path / "attribute.json"
	with open(path, "w") as f:
		json.dump(cfg, f)

	with mock.patch("cherimoya.Cherimoya") as model_cls, \
			mock.patch("tangermeme.io.extract_loci",
				side_effect=fake_extract), \
			mock.patch("tangermeme.saturation_mutagenesis."
				"saturation_mutagenesis", side_effect=fake_sm):
		model_cls.load.return_value = mock.MagicMock(n_control_tracks=0)
		attribute.run(argparse.Namespace(parameters=str(path)))

	captured["ohe"] = numpy.load(tmp_path / "a.ohe.npz")["arr_0"]
	captured["attr"] = numpy.load(tmp_path / "a.attr.npz")["arr_0"]
	return captured


def _genomic_start_of_saved_slice(captured):
	"""The genomic coordinate of column 0 of the array `attribute`
	saved, found by locating that array inside the window it was cut
	from."""

	X = captured["X"].numpy()
	ohe = captured["ohe"]
	width = ohe.shape[-1]

	offsets = [o for o in range(X.shape[-1] - width + 1)
		if numpy.array_equal(X[:, :, o:o + width], ohe)]

	assert len(offsets) == 1, \
		"saved slice matches the window at {} offsets".format(len(offsets))

	return captured["window_start"] + offsets[0]


def _run_seqlets(tmp_path, start, end):
	"""Run `cherimoya seqlets` over whatever `attribute` just wrote,
	with one seqlet at a known position of the attributed slice."""

	from cherimoya_cli.commands import seqlets

	frame = pandas.DataFrame({
		"example_idx": [0], "start": [start], "end": [end],
		"attribution": [1.0], "p-value": [0.0]})

	cfg = dict(default_seqlet_parameters)
	cfg.update({
		"loci": str(tmp_path / "loci.bed"), "verbose": False,
		"chroms": [LOCUS_CHROM],
		"ohe_filename": str(tmp_path / "a.ohe.npz"),
		"attr_filename": str(tmp_path / "a.attr.npz"),
		"idx_filename": str(tmp_path / "a.idx.npy"),
		"output_filename": str(tmp_path / "seqlets.bed"),
	})

	path = tmp_path / "seqlets.json"
	with open(path, "w") as f:
		json.dump(cfg, f)

	with mock.patch("tangermeme.seqlet.recursive_seqlets",
			return_value=frame):
		seqlets.run(argparse.Namespace(parameters=str(path)))

	return pandas.read_csv(tmp_path / "seqlets.bed", sep="\t", header=None)


@pytest.fixture
def loci_bed(tmp_path):
	pandas.DataFrame([[LOCUS_CHROM, LOCUS_START, LOCUS_END]]).to_csv(
		tmp_path / "loci.bed", sep="\t", header=False, index=False)
	return tmp_path / "loci.bed"


##


@pytest.mark.parametrize("in_window,attr_window", WINDOWS)
@pytest.mark.parametrize("position", [0, 10, 137])
def test_seqlet_lands_on_the_base_it_was_attributed_from(tmp_path,
		loci_bed, in_window, attr_window, position):
	"""The contract between the two commands: the coordinate `seqlets`
	reports for position `p` is the genome position of the base that
	sat at column `p` of the array `attribute` saved.

	The left side comes from `seqlets`. The right side comes from
	locating the saved array inside the extracted window, so neither
	side re-derives the other's arithmetic -- off-centre the slice in
	`attribute` and only this comparison notices.
	"""

	captured = _run_attribute(tmp_path, in_window, attr_window)
	out = _run_seqlets(tmp_path, position, position + 10)

	expected = _genomic_start_of_saved_slice(captured) + position
	assert out.iloc[0, 0] == LOCUS_CHROM
	assert out.iloc[0, 1] == expected
	assert out.iloc[0, 2] == expected + 10


@pytest.mark.parametrize("in_window,attr_window", WINDOWS)
def test_every_seqlet_position_stays_inside_the_locus(tmp_path, loci_bed,
		in_window, attr_window):
	"""The user-visible symptom of a broken seam: a seqlet outside the
	peak it came from. Checked at both ends of the slice, since an
	offset error shows at one end first."""

	_run_attribute(tmp_path, in_window, attr_window)

	for start, end in ((0, 5), (attr_window - 5, attr_window)):
		out = _run_seqlets(tmp_path, start, end)
		assert out.iloc[0, 1] >= LOCUS_START
		assert out.iloc[0, 2] <= LOCUS_END


def test_attribute_writes_arrays_seqlets_can_size_itself_from(tmp_path,
		loci_bed):
	"""`seqlets` reads the conversion width off the array rather than
	from a key, so the width `attribute` writes is the whole interface
	between them."""

	captured = _run_attribute(tmp_path, in_window=2114, attr_window=600)

	assert captured["ohe"].shape[-1] == 600
	assert captured["attr"].shape == captured["ohe"].shape
