# cherimoya_cli batch command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(args):
    import copy
    import glob
    import json
    import subprocess

    import torch

    from joblib import Parallel
    from joblib import delayed

    from ..defaults import default_pipeline_parameters
    from ..utils import merge_parameters

    parameters = merge_parameters(args.parameters, default_pipeline_parameters)
    pname = parameters["name"]

    # Automatically extract parameters if wildcards are provided
    if parameters["device"] == "*":
        parameters["device"] = [
            "cuda:{}".format(i) for i in range(torch.cuda.device_count())
        ]

    if "*" in parameters["signals"]:
        parameters["signals"] = list(glob.glob(parameters["signals"]))

        if parameters["name"] is None:
            parameters["name"] = []
            for name in parameters["signals"]:
                if name.endswith(".gz"):
                    name = ".".join(name.split("/")[-1].split(".")[:-2])
                else:
                    name = ".".join(name.split("/")[-1].split(".")[:-1])

                parameters["name"].append(name)

    print(parameters["signals"])

    # Check that the parameters are all correctly formatted
    assert isinstance(parameters["name"], list)
    assert isinstance(parameters["signals"], list)
    assert len(parameters["name"]) == len(parameters["signals"])
    for key in "loci", "controls", "negatives":
        if parameters[key] is not None:
            assert isinstance(parameters[key], list)
            assert len(parameters["name"]) == len(parameters[key])

    # Create and run each of the JSONs
    def _create_and_run_json(parameters, i):
        name = parameters["name"][i]

        pipeline_json = copy.deepcopy(parameters)
        pipeline_json["name"] = name
        pipeline_json["device"] = parameters["device"][i % len(parameters["device"])]
        pipeline_json["signals"] = parameters["signals"][i]

        for key in "loci", "negatives", "controls":
            parameter = parameters[key]
            pipeline_json[key] = None if parameter is None else parameter[i]

        for key in "signals", "loci", "negatives", "controls":
            value = pipeline_json[key]

            if value is not None and not isinstance(value, list):
                pipeline_json[key] = [value]

        jname = "{}.pipeline.json".format(name)
        with open(jname, "w") as outfile:
            outfile.write(json.dumps(pipeline_json, indent=4))

        subprocess.run(["cherimoya", "pipeline", "-p", jname], check=True)

    n = len(parameters["name"])
    n_devices = len(parameters["device"])
    f = delayed(_create_and_run_json)

    Parallel(n_jobs=n_devices)(f(parameters, i) for i in range(n))
