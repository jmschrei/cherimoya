"""Tests that `dry_run` emits the per-step JSONs without running anything.

`dry_run` is how a pipeline config is checked before committing hours of
GPU time to it, so it has to complete on exactly the configurations a
real run would use -- including the one with a motif database, which is
the common case and the only one that reaches the annotation step.
"""


##


def test_dry_run_with_motifs_completes(tmp_path, run_pipeline):
	"""With a motif database set, the run reaches the end. The seqlet
	annotation step used to read its own output with `pandas.read_csv`
	outside the `dry_run` guard, so it raised FileNotFoundError on a
	file no subprocess had been allowed to write."""

	run_pipeline(motifs=str(tmp_path / "m.meme"))


def test_dry_run_with_motifs_writes_no_annotation_outputs(tmp_path,
		run_pipeline):
	"""A dry run emits the step JSONs and nothing else -- in particular
	not the tomtom-lite outputs, which would otherwise be empty files
	standing in for real results."""

	run_pipeline(motifs=str(tmp_path / "m.meme"))

	assert not (tmp_path / "demo.seqlets_annotated.bed").exists()
	assert not (tmp_path / "demo.motif_seqlet_count.tsv").exists()


def test_dry_run_with_motifs_writes_the_step_jsons(tmp_path, run_pipeline):
	"""The point of a dry run: every per-step JSON lands on disk so the
	config can be inspected."""

	run_pipeline(motifs=str(tmp_path / "m.meme"))

	for name in ("demo.fit.json", "demo.attribute.json",
			"demo.seqlets.json", "demo.marginalize.json"):
		assert (tmp_path / name).exists(), name
