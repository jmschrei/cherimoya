# cherimoya_cli seqlets command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(args):

	import numpy
	import torch

	from tangermeme.io import _interleave_loci
	from tangermeme.seqlet import recursive_seqlets
	from tangermeme.utils import example_to_fasta_coords

	from ..defaults import default_seqlet_parameters
	from ..utils import merge_parameters

	parameters = merge_parameters(args.parameters, default_seqlet_parameters)
	if parameters["skip"]:
		return

	###

	idxs = numpy.load(parameters["idx_filename"])

	loci = _interleave_loci(parameters["loci"], parameters["chroms"])
	loci = loci.iloc[idxs]

	X = numpy.load(parameters["ohe_filename"])["arr_0"]
	X = torch.from_numpy(X)

	X_attr = numpy.load(parameters["attr_filename"])["arr_0"]
	X_attr = torch.from_numpy(X_attr)
	X_attr = (X_attr * X).sum(dim=1)

	seqlets = recursive_seqlets(
		X_attr,
		threshold=parameters["threshold"],
		min_seqlet_len=parameters["min_seqlet_len"],
		max_seqlet_len=parameters["max_seqlet_len"],
		additional_flanks=parameters["additional_flanks"],
	).sort_values("attribution", ascending=False)

	# The attributed slice is centred inside the extraction window and
	# narrower than it, so the window to convert against is the slice's
	# own width, read off the array rather than from a parameter.
	attr_window = X.shape[-1]

	# An empty result has `object` dtype columns, which
	# `example_to_fasta_coords` cannot index the locus table with. The
	# empty BED is still written; the annotation step opens it either way.
	if len(seqlets) > 0:
		seqlets = example_to_fasta_coords(seqlets, loci, attr_window)

	seqlets.to_csv(parameters["output_filename"], sep="\t", index=False, header=False)
