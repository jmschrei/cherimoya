# cherimoya_cli pipeline-json command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(args):
    import sys
    import json
    import copy

    from ..defaults import default_pipeline_parameters

    parameters = copy.deepcopy(default_pipeline_parameters)

    parameters["sequences"] = args.sequences
    parameters["loci"] = args.peaks
    parameters["negatives"] = args.negatives
    parameters["signals"] = args.inputs
    parameters["controls"] = args.controls
    parameters["name"] = args.name
    parameters["motifs"] = args.motifs
    parameters["preprocessing_parameters"]["unstranded"] = args.unstranded
    parameters["preprocessing_parameters"]["fragments"] = args.fragments
    parameters["preprocessing_parameters"]["pos_shift"] = args.pos_shift
    parameters["preprocessing_parameters"]["neg_shift"] = args.neg_shift
    parameters["preprocessing_parameters"]["paired_end"] = args.paired_end
    parameters["preprocessing_parameters"]["scale_factor"] = args.scale_factor

    with open(args.output, "w") as outfile:
        outfile.write(json.dumps(parameters, indent=4))

    sys.exit()
