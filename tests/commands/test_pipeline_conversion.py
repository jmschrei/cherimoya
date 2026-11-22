"""Tests for the pipeline's conversion of reads to bigWigs (step 0.2).

Signals and controls given as SAM/BAM or BED/tsv files are converted by
`figwig bam2bw`, and `signals`/`controls` are rewritten to the bigWigs it
writes before the fit JSON is made. The first tests check the command line
the pipeline builds; the last runs the real command on a three-read BAM.
"""

import contextlib
import json
import subprocess
from unittest import mock

import numpy
import pytest

from cherimoya_cli.defaults import default_pipeline_parameters


STEPS = ("fit", "attribute", "seqlets", "marginalize")


def _run(run_pipeline, real_conversion=False, **kwargs):
	"""Run the pipeline with every step after the conversion mocked, and
	return the commands it passed to subprocess.run. With
	`real_conversion`, the `figwig bam2bw` calls run for real."""

	real_run = subprocess.run

	def fake_run(cmd, *args, **kw):
		if real_conversion and cmd[:2] == ["figwig", "bam2bw"]:
			return real_run(cmd, *args, **kw)

	with contextlib.ExitStack() as stack:
		subprocess_run = stack.enter_context(mock.patch("subprocess.run",
			side_effect=fake_run))
		for step in STEPS:
			stack.enter_context(mock.patch(
				"cherimoya_cli.commands.{}.run".format(step)))

		run_pipeline(dry_run=False, **kwargs)

	return [call.args[0] for call in subprocess_run.call_args_list]


def _conversions(commands):
	return [cmd for cmd in commands if cmd[:2] == ["figwig", "bam2bw"]]


def _preprocessing(**kwargs):
	"""Every preprocessing key, since a partial dict must still name
	`callpeaks_format`, whose default is None."""

	return dict(default_pipeline_parameters["preprocessing_parameters"],
		**kwargs)


##


def test_conversion_runs_figwig_bam2bw_on_signals_and_controls(tmp_path,
		run_pipeline):
	for name in ("s.bam", "c.bam"):
		(tmp_path / name).write_text("")

	commands = _run(run_pipeline, signals=[str(tmp_path / "s.bam")],
		controls=[str(tmp_path / "c.bam")])

	assert _conversions(commands) == [
		["figwig", "bam2bw", "-s", str(tmp_path / "g.fa"), "-n", "demo",
			"-ps", "0", "-ns", "0", "-sf", "1", "-p", "8", "-v",
			str(tmp_path / "s.bam")],
		["figwig", "bam2bw", "-s", str(tmp_path / "g.fa"), "-n",
			"demo.control", "-ps", "0", "-ns", "0", "-p", "8", "-v",
			str(tmp_path / "c.bam")],
	]


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_conversion_passes_n_jobs_as_cores(n_jobs, tmp_path, run_pipeline):
	"""`n_jobs` is figwig's `-p`, a number of cores, for the signals and
	the controls alike."""

	for name in ("s.bam", "c.bam"):
		(tmp_path / name).write_text("")

	commands = _run(run_pipeline, signals=[str(tmp_path / "s.bam")],
		controls=[str(tmp_path / "c.bam")],
		preprocessing_parameters=_preprocessing(n_jobs=n_jobs))

	for cmd in _conversions(commands):
		assert cmd[cmd.index("-p") + 1] == str(n_jobs)
	assert len(_conversions(commands)) == 2


def test_conversion_skipped_for_bigwig_signals(run_pipeline):
	assert _conversions(_run(run_pipeline)) == []


@pytest.mark.parametrize("unstranded", [False, True])
def test_conversion_writes_the_bigwigs_the_fit_json_names(unstranded,
		tmp_path, run_pipeline):
	"""The real `figwig bam2bw` writes the files that `signals` is
	rewritten to, with each read's 5' end counted on its strand: the start
	of a + strand read, and the last base of a - strand read."""

	pysam = pytest.importorskip("pysam")
	from figwig import read_bigwig

	# Not g.fa, which the `pipeline_json` fixture empties on every call.
	(tmp_path / "genome.fa").write_text(">chr1\n" + "ACGT" * 250 + "\n")

	header = {"HD": {"VN": "1.6", "SO": "coordinate"},
		"SQ": [{"SN": "chr1", "LN": 1000}]}
	with pysam.AlignmentFile(str(tmp_path / "s.bam"), "wb",
			header=header) as f:
		for i, (start, flag) in enumerate([(100, 0), (100, 0), (300, 16)]):
			read = pysam.AlignedSegment(f.header)
			read.query_name = "r{}".format(i)
			read.flag = flag
			read.reference_id = 0
			read.reference_start = start
			read.mapping_quality = 60
			read.cigarstring = "50M"
			read.query_sequence = "A" * 50
			f.write(read)

	_run(run_pipeline, real_conversion=True,
		sequences=str(tmp_path / "genome.fa"),
		signals=[str(tmp_path / "s.bam")],
		preprocessing_parameters=_preprocessing(unstranded=unstranded,
			n_jobs=1))

	with open(tmp_path / "demo.fit.json") as f:
		signals = json.load(f)["signals"]

	expected = numpy.zeros((2, 1000), dtype=numpy.float32)
	expected[0, 100] = 2
	expected[1, 349] = 1

	if unstranded:
		assert signals == ["demo.bw"]
		values = read_bigwig(str(tmp_path / "demo.bw"), "chr1", [0], 1000)[0]
		numpy.testing.assert_array_equal(values, expected.sum(axis=0))
	else:
		assert signals == [["demo.+.bw", "demo.-.bw"]]
		values = read_bigwig([str(tmp_path / name) for name in signals[0]],
			"chr1", [0], 1000)[0]
		numpy.testing.assert_array_equal(values, expected)


def test_figwig_tracks_pass_the_check_and_skip_conversion(tmp_path,
		run_pipeline):
	"""A signal named as a track of a figwig file is checked by its file, and
	is read as it is, like a bigWig."""

	(tmp_path / "tracks.fw").write_text("")
	signals = [[str(tmp_path / "tracks.fw") + "#plus", str(tmp_path /
		"tracks.fw") + "#minus"]]
	assert _conversions(_run(run_pipeline, signals=signals)) == []
	with open(tmp_path / "demo.fit.json") as handle:
		assert json.load(handle)["signals"] == signals


def test_a_missing_figwig_file_is_named(tmp_path, run_pipeline):
	missing = str(tmp_path / "absent.fw")
	with pytest.raises(FileNotFoundError, match="signals: " + missing + "$"):
		run_pipeline(signals=[missing + "#atac"])
