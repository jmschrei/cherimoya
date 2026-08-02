# cherimoya_cli seqlets command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(args):
    import sys

    import numpy
    import torch

    from tangermeme.io import _interleave_loci
    from tangermeme.seqlet import recursive_seqlets
    from tangermeme.utils import example_to_fasta_coords

    from ..defaults import default_seqlet_parameters
    from ..utils import merge_parameters

    parameters = merge_parameters(args.parameters, default_seqlet_parameters)
    if parameters["skip"]:
        sys.exit()

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

    seqlets = example_to_fasta_coords(seqlets, loci, parameters["in_window"])
    seqlets.to_csv(parameters["output_filename"], sep="\t", index=False, header=False)
