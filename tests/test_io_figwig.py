# test_io_figwig.py
# Signals given as tracks of a figwig file, "<file>.fw#<track>", read by
# `cherimoya.io.extract_loci` and `PeakGenerator`. The reference is
# tangermeme reading bigWigs of the same tracks, which takes its windows
# without any of the code under test, so every array must be identical.

import numpy
import pandas
import pytest
import torch

figwig = pytest.importorskip("figwig")
if not hasattr(figwig, "FigWigReader"):
	pytest.skip("figwig >= 0.2 reads figwig files", allow_module_level=True)

from cherimoya.io import PeakGenerator
from cherimoya.io import extract_loci
from tangermeme.io import extract_loci as tangermeme_extract_loci

CHROM_SIZES = {"chr1": 30_000, "chr2": 21_017, "chr3": 9_000}
NAMES = ["plus", "minus", "atac"]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
	"""A FASTA, three bigWigs of integer counts (a stranded pair and an
	unstranded track), the same three tracks in one figwig file and the
	stranded pair again in a second, peaks with summits, negatives and an
	exclusion list."""

	root = tmp_path_factory.mktemp("figwig_io")
	rng = numpy.random.default_rng(0)

	fasta = root / "genome.fa"
	with open(fasta, "w") as handle:
		for chrom, length in CHROM_SIZES.items():
			sequence = rng.choice(list("ACGT"), length)
			sequence[100:140] = "N"
			handle.write(">{}\n".format(chrom))
			for k in range(0, length, 60):
				handle.write("".join(sequence[k:k + 60]) + "\n")

	values = {}
	for name in NAMES:
		values[name] = {chrom: (rng.poisson(0.3, length) * (rng.random(length) <
			0.2)).astype(numpy.float64) for chrom, length in CHROM_SIZES.items()}
	# Counts above 255 and 2,048, so that the tracks are wider than uint8 and
	# an int16 read is not trivially exact.
	values["atac"]["chr1"][15_000] = 3_000

	bigwigs = {}
	for name in NAMES:
		bigwigs[name] = str(root / (name + ".bw"))
		with figwig.BigWigWriter(bigwigs[name], CHROM_SIZES, zooms=0) as writer:
			for chrom in CHROM_SIZES:
				writer.write(chrom, [0], values[name][chrom][None].astype(
					numpy.float32))

	path = str(root / "tracks.fw")
	with figwig.FigWigWriter(path, CHROM_SIZES, NAMES, tile=1024) as writer:
		for chrom in CHROM_SIZES:
			writer.write(chrom, [0], numpy.stack([values[name][chrom] for name in
				NAMES])[None])
	pair = str(root / "pair.fw")
	with figwig.FigWigWriter(pair, CHROM_SIZES, ["plus", "minus"],
			tile=512) as writer:
		for chrom in CHROM_SIZES:
			writer.write(chrom, [0], numpy.stack([values[name][chrom] for name in
				["plus", "minus"]])[None])

	# Peaks with a summit column; some near the ends of chromosomes or on the
	# Ns, which tangermeme drops, and some on chr3, left out by `chroms`.
	rows = []
	for chrom, length in CHROM_SIZES.items():
		for start in list(rng.integers(0, length - 400, 30)) + [0, length - 300,
				90]:
			start = int(start)
			end = min(length, start + int(rng.integers(150, 400)))
			rows.append((chrom, start, end, ".", 0, ".", 0, 0, 0,
				int(rng.integers(0, end - start))))
	peaks = root / "peaks.narrowPeak"
	pandas.DataFrame(rows).to_csv(peaks, sep="\t", header=False, index=False)

	negatives = root / "negatives.bed"
	pandas.DataFrame([(chrom, int(s), int(s) + 1000) for chrom, length in
		CHROM_SIZES.items() for s in rng.integers(0, length - 1000, 20)]).to_csv(
		negatives, sep="\t", header=False, index=False)

	exclusion = root / "exclusion.bed"
	pandas.DataFrame([("chr1", 5_000, 6_500), ("chr2", 12_000, 12_400)]).to_csv(
		exclusion, sep="\t", header=False, index=False)

	return {"fasta": str(fasta), "bigwigs": bigwigs, "figwig": path,
		"pair": pair, "peaks": str(peaks), "negatives": str(negatives),
		"exclusion": str(exclusion)}


def track(files, name, key="figwig"):
	return "{}#{}".format(files[key], name)


def assert_identical(a, b):
	assert a.dtype == b.dtype and a.shape == b.shape
	assert torch.equal(a, b)


@pytest.mark.parametrize("kwargs", [
	dict(),
	dict(max_jitter=37, summits=True),
	dict(chroms=["chr1", "chr2"], exclusion_lists="exclusion"),
	dict(in_window=501, out_window=301, max_jitter=5),
	dict(in_window=200, out_window=200),
])
def test_figwig_tracks_read_as_their_bigwigs(files, kwargs):
	kwargs = dict(kwargs)
	if kwargs.get("exclusion_lists") == "exclusion":
		kwargs["exclusion_lists"] = [files["exclusion"]]
	common = dict(loci=files["peaks"], sequences=files["fasta"],
		ignore=list("QWERYUIOPSDFHJKLZXVBNM"), return_mask=True, **kwargs)
	expected = tangermeme_extract_loci(signals=[files["bigwigs"]["plus"],
		files["bigwigs"]["minus"]], in_signals=[files["bigwigs"]["atac"]],
		**common)
	found = extract_loci(signals=[track(files, "plus"), track(files, "minus")],
		in_signals=[track(files, "atac")], **common)
	assert len(found) == len(expected) == 4
	for a, b in zip(found, expected):
		assert_identical(a, b)
	assert int(found[-1].sum()) > 0
	# With the default windows, the loci at a chromosome's ends or on the Ns
	# are dropped, so the mask is tested on both values.
	if not kwargs:
		assert int(found[-1].sum()) < len(found[-1])


def test_tracks_from_several_files_in_any_order(files):
	common = dict(loci=files["peaks"], sequences=files["fasta"], max_jitter=10)
	expected = tangermeme_extract_loci(signals=[files["bigwigs"][name] for name
		in ["atac", "minus", "plus", "minus"]], **common)
	found = extract_loci(signals=[track(files, "atac"), track(files, "minus",
		"pair"), track(files, "plus"), track(files, "minus")], **common)
	for a, b in zip(found, expected):
		assert_identical(a, b)


def test_signals_alone_and_sequences_alone(files):
	common = dict(loci=files["negatives"], sequences=files["fasta"])
	expected = tangermeme_extract_loci(signals=[files["bigwigs"]["atac"]],
		**common)
	found = extract_loci(signals=[track(files, "atac")], **common)
	for a, b in zip(found, expected):
		assert_identical(a, b)
	assert_identical(extract_loci(**common), tangermeme_extract_loci(**common))
	X, y = extract_loci(in_signals=None, signals=[track(files, "atac")], **common)
	assert y.shape == (len(X), 1, 1000)


def test_int16_holds_the_counts(files):
	common = dict(loci=files["peaks"], sequences=files["fasta"], max_jitter=50,
		signals=[track(files, "plus"), track(files, "atac")],
		in_signals=[track(files, "minus")])
	floats = extract_loci(**common)
	counts = extract_loci(signal_dtype="int16", **common)
	assert counts[1].dtype == counts[2].dtype == torch.int16
	assert counts[1].max() == 3_000
	assert_identical(counts[0], floats[0])
	for a, b in zip(counts[1:], floats[1:]):
		assert_identical(a.float(), b)


@pytest.mark.parametrize("kwargs, match", [
	(dict(signals=["{figwig}#plus", "{bigwig}"]), "all be figwig tracks"),
	(dict(signals=["{figwig}#plus"], min_counts=1), "min_counts"),
	(dict(signals=["{figwig}#plus"], n_loci=3), "n_loci"),
	(dict(signals=["{figwig}#plus"], return_packed=True), "return_packed"),
	(dict(signals=["{figwig}#plus"], in_window=100, out_window=300),
		"output window wider"),
	(dict(signals=["{bigwig}"], signal_dtype="int16"), "only with figwig"),
])
def test_what_figwig_tracks_refuse(files, kwargs, match):
	kwargs = dict(kwargs)
	kwargs["signals"] = [s.format(figwig=files["figwig"],
		bigwig=files["bigwigs"]["plus"]) for s in kwargs["signals"]]
	with pytest.raises(ValueError, match=match):
		extract_loci(loci=files["peaks"], sequences=files["fasta"], **kwargs)


def test_a_track_the_file_lacks_raises(files):
	with pytest.raises(ValueError):
		extract_loci(loci=files["peaks"], sequences=files["fasta"],
			signals=[track(files, "absent")])


def test_names_that_are_not_figwig_tracks_go_to_tangermeme(files):
	# A bigWig whose path holds '#' is not a figwig track.
	from cherimoya.io import _figwig_track
	assert _figwig_track("a/b#c.bw") is None
	assert _figwig_track("x.fw#") is None
	assert _figwig_track("x.fw") is None
	assert _figwig_track({"chr1": 0}) is None
	assert _figwig_track("dir#1/x.fw#atac") == ("dir#1/x.fw", "atac")


def sampler(files, signals, controls, signal_dtype="float32"):
	return PeakGenerator(peaks=files["peaks"], negatives=files["negatives"],
		sequences=files["fasta"], signals=signals, controls=controls,
		chroms=["chr1", "chr2"], in_window=500, out_window=300, max_jitter=20,
		summits=True, exclusion_lists=[files["exclusion"]], random_state=0,
		negative_ratio=0.5, num_workers=0, signal_dtype=signal_dtype).dataset


@pytest.mark.parametrize("signal_dtype", ["float32", "int16"])
def test_peak_generator_from_figwig_tracks(files, signal_dtype):
	bigwigs = files["bigwigs"]
	expected = sampler(files, [[bigwigs["plus"], bigwigs["minus"]]],
		[bigwigs["atac"]])
	found = sampler(files, [[track(files, "plus"), track(files, "minus")]],
		[track(files, "atac")], signal_dtype)

	for key in ("peak_sequences", "negative_sequences"):
		numpy.testing.assert_array_equal(numpy.asarray(getattr(found, key)),
			numpy.asarray(getattr(expected, key)))
	for key in ("peak_signals", "negative_signals", "peak_controls",
			"negative_controls"):
		a, b = numpy.asarray(getattr(found, key)), numpy.asarray(getattr(expected,
			key))
		assert a.dtype == numpy.dtype(signal_dtype)
		numpy.testing.assert_array_equal(a.astype(numpy.float32), b)

	# The examples drawn, jittered, reverse-complemented and strand-swapped,
	# are the bigWig sampler's once cast.
	for idx in range(len(expected)):
		for a, b in zip(found[idx], expected[idx]):
			if isinstance(b, torch.Tensor):
				assert torch.equal(a.float(), b.float())
			else:
				assert a == b
