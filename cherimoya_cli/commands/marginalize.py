# cherimoya_cli marginalize command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(args):

	import numpy
	import torch

	from bpnetlite.marginalize import marginalization_report
	from tangermeme.io import extract_loci

	from cherimoya import Cherimoya
	from cherimoya import ControlWrapper
	from ..defaults import default_marginalize_parameters
	from ..utils import merge_parameters

	parameters = merge_parameters(args.parameters, default_marginalize_parameters)
	if parameters["skip"]:
		return

	###

	model = Cherimoya.load(parameters["model"], device=parameters["device"],
		compile=parameters["compile"],
		compile_mode=parameters["compile_mode"])

	if model.n_control_tracks > 0:
		model = ControlWrapper(model)

	# `extract_loci` stops at `n_loci`, i.e. returns the first `n_loci`
	# rows, so capping it here would leave the shuffle below permuting a
	# set already chosen by file order. Costs the whole file in memory,
	# which is why the unshuffled path still caps.
	extract_n_loci = None if parameters["shuffle"] else parameters["n_loci"]

	X = extract_loci(
		sequences=parameters["sequences"],
		loci=parameters["loci"],
		chroms=parameters["chroms"],
		max_jitter=0,
		ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
		n_loci=extract_n_loci,
		verbose=parameters["verbose"],
	).float()

	if parameters["shuffle"] == True:
		# Seeded from `random_state` so that which loci the report is
		# built from is reproducible. `RandomState(None)` draws from
		# system entropy, so a null seed keeps the unseeded behavior.
		idxs = numpy.arange(X.shape[0])
		numpy.random.RandomState(parameters["random_state"]).shuffle(idxs)
		X = X[idxs]

	if parameters["n_loci"] is not None:
		X = X[: parameters["n_loci"]]

	marginalization_report(
		model,
		parameters["motifs"],
		X,
		parameters["output_filename"],
		attributions=parameters["attributions"],
		batch_size=parameters["batch_size"],
		minimal=parameters["minimal"],
		device=parameters["device"],
		verbose=parameters["verbose"],
	)
