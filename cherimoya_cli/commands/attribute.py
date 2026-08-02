# cherimoya_cli attribute command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(args):
    import sys

    import numpy
    import torch

    from tangermeme.io import extract_loci
    from tangermeme.saturation_mutagenesis import saturation_mutagenesis

    from cherimoya import Cherimoya
    from cherimoya import ControlWrapper
    from cherimoya import LogCountWrapper
    from cherimoya import ProfileWrapper
    from ..defaults import default_attribute_parameters
    from ..utils import merge_parameters

    parameters = merge_parameters(args.parameters, default_attribute_parameters)
    if parameters["skip"]:
        sys.exit()

    ###

    model = Cherimoya.load(parameters["model"], device=parameters["device"])

    X, idxs = extract_loci(
        sequences=parameters["sequences"],
        loci=parameters["loci"],
        chroms=parameters["chroms"],
        max_jitter=0,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        return_mask=True,
        verbose=parameters["verbose"],
    )

    n_idxs = X.sum(dim=(1, 2)) == X.shape[-1]
    X = X[n_idxs]
    idxs[idxs.clone()] = n_idxs

    model = ControlWrapper(model)
    if parameters["output"] == "counts":
        wrapper = LogCountWrapper(model)
    elif parameters["output"] == "profile":
        wrapper = ProfileWrapper(model)
    else:
        raise ValueError("output must be either `counts` or `profile`.")

    mid = X.shape[-1] // 2
    start, end = mid - 200, mid + 200

    X_attr = saturation_mutagenesis(
        wrapper,
        X,
        dtype=parameters["dtype"],
        device=parameters["device"],
        batch_size=parameters["batch_size"],
        verbose=parameters["verbose"],
        hypothetical=True,
        start=start,
        end=end,
    ).float()

    numpy.savez_compressed(parameters["ohe_filename"], X[:, :, start:end])
    numpy.savez_compressed(parameters["attr_filename"], X_attr)
    numpy.save(parameters["idx_filename"], idxs)
