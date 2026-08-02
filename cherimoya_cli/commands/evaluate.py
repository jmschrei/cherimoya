# cherimoya_cli evaluate command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(args):
    import sys

    import torch

    from tangermeme.io import extract_loci
    from tangermeme.predict import predict

    from cherimoya import Cherimoya
    from cherimoya import ControlWrapper
    from cherimoya.io import normalize_signal_groups
    from cherimoya.performance import calculate_performance_measures
    from ..defaults import default_evaluate_parameters
    from ..utils import merge_parameters

    parameters = merge_parameters(args.parameters, default_evaluate_parameters)
    if parameters["skip"]:
        sys.exit()

    # Flatten any structured (list-of-lists) signals/controls so they
    # can be handed to extract_loci. The model's own signal_groups
    # (recovered below from the loaded checkpoint) drives count pooling
    # under calculate_performance_measures.
    signal_files, signal_groups = normalize_signal_groups(parameters["signals"])
    control_files, _ = normalize_signal_groups(parameters["controls"])
    parameters["signals"] = signal_files
    parameters["controls"] = control_files

    measure_names = [
        "profile_mnll",
        "profile_jsd",
        "profile_pearson",
        "profile_spearman",
        "count_pearson",
        "count_spearman",
        "count_mse",
    ]

    ###

    model = Cherimoya.load(parameters["model"], device=parameters["device"])

    examples = extract_loci(
        sequences=parameters["sequences"],
        signals=parameters["signals"],
        in_signals=parameters["controls"],
        loci=parameters["loci"],
        chroms=parameters["chroms"],
        in_window=parameters["in_window"],
        out_window=parameters["out_window"],
        exclusion_lists=parameters["exclusion_lists"],
        max_jitter=0,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        verbose=parameters["verbose"],
    )

    if parameters["controls"] == None:
        X, y = examples
        X_ctl = None
        if model.n_control_tracks > 0:
            model = ControlWrapper(model)
    else:
        X, y, X_ctl = examples
        X_ctl = (X_ctl,)

    y_hat_logits, y_hat_logcounts = predict(
        model,
        X,
        args=X_ctl,
        batch_size=parameters["batch_size"],
        device=parameters["device"],
        dtype=parameters["dtype"],
        verbose=parameters["verbose"],
    )

    if parameters["reverse_complement_average"]:
        X_rc = torch.flip(X, dims=(-1, -2))
        X_ctl_rc = None if X_ctl is None else (torch.flip(X_ctl[0], dims=(-1, -2)),)

        y_hat_logits_rc, y_hat_logcounts_rc = predict(
            model,
            X_rc,
            args=X_ctl_rc,
            batch_size=parameters["batch_size"],
            device=parameters["device"],
            dtype=parameters["dtype"],
            verbose=parameters["verbose"],
        )

        y_hat_logits_rc = torch.flip(y_hat_logits_rc, dims=(-1, -2))
        y_hat_logits = (y_hat_logits + y_hat_logits_rc) / 2
        y_hat_logcounts = (y_hat_logcounts + y_hat_logcounts_rc) / 2

    # Prefer the model's own grouping over whatever the caller passed
    # in the JSON — the checkpoint is authoritative about how its count
    # head is laid out.
    model_signal_groups = getattr(model, "signal_groups", None)
    if model_signal_groups is None:
        model_signal_groups = signal_groups

    measures = calculate_performance_measures(
        y_hat_logits, y, y_hat_logcounts, signal_groups=model_signal_groups
    )

    # Build one row per signal group. Profile metrics come back shape
    # (n_loci, sum(signal_groups)) — average over each group's channel
    # slice and the locus dim. Count metrics already arrive at
    # (n_groups,) when signal_groups is given, so the per-group value
    # is just the i-th element.
    #
    # For a single-group model the output reduces to one row that is
    # byte-identical to the legacy `.mean()`-over-everything line:
    # the profile slice is the whole tensor (one group spans every
    # channel), and the count vector has length 1.
    groups = model_signal_groups or [y_hat_logits.shape[1]]
    rows = []
    offset = 0
    for i, g in enumerate(groups):
        row = []
        for name in measure_names:
            value = measures[name]
            if name.startswith("profile_"):
                row.append(value[:, offset : offset + g].mean().item())
            else:
                row.append(value[i].item() if value.ndim >= 1 else value.item())
        rows.append(row)
        offset += g

    def _format_rows():
        yield "\t".join(measure_names)
        for row in rows:
            yield "\t".join(str(v) for v in row)

    with open(parameters["performance_filename"], "w") as outfile:
        outfile.write("\n".join(_format_rows()))

    if parameters["verbose"]:
        for line in _format_rows():
            print(line)
