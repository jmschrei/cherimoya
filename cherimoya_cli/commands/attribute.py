# cherimoya_cli attribute command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(args):

	import numpy
	import torch

	from tangermeme.deep_lift_shap import deep_lift_shap
	from tangermeme.io import extract_loci
	from tangermeme.saturation_mutagenesis import saturation_mutagenesis

	from cherimoya import Cherimoya
	from cherimoya import ControlWrapper
	from cherimoya import LogCountWrapper
	from cherimoya import ProfileWrapper
	from cherimoya.deep_lift_shap import attribution_ops
	from ..defaults import default_attribute_parameters
	from ..utils import merge_parameters

	parameters = merge_parameters(args.parameters, default_attribute_parameters)
	if parameters["skip"]:
		return

	algorithm = parameters["algorithm"]
	if algorithm not in ("deep_lift_shap", "saturation_mutagenesis"):
		raise ValueError("algorithm must be either `deep_lift_shap` or "
			"`saturation_mutagenesis`, got {!r}".format(algorithm))

	###

	# DeepLIFT's backward hooks make the compiled forward recompile per
	# block until it falls back to eager, so compiling only adds warm-up.
	compiled = parameters["compile"] and algorithm == "saturation_mutagenesis"
	model = Cherimoya.load(parameters["model"], device=parameters["device"],
		compile=compiled,
		compile_mode=parameters["compile_mode"])

	# DeepLIFT attributes one output, so a count head with several groups
	# has to be narrowed to one of them first.
	group = parameters["group"]
	if (algorithm == "deep_lift_shap" and parameters["output"] == "counts"
			and group is None and len(model.signal_groups) > 1):
		raise ValueError("deep_lift_shap attributes one output; set `group` "
			"to one of the model's {} signal groups"
			.format(len(model.signal_groups)))

	X, idxs = extract_loci(
		sequences=parameters["sequences"],
		loci=parameters["loci"],
		chroms=parameters["chroms"],
		in_window=parameters["in_window"],
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
		wrapper = LogCountWrapper(model, group=group)
	elif parameters["output"] == "profile":
		wrapper = ProfileWrapper(model, group=group)
	else:
		raise ValueError("output must be either `counts` or `profile`.")

	# Only the centred `attr_window` slice is saved. Saturation mutagenesis
	# is one forward pass per alternate base per position, so for it this
	# width sets the cost of the step; DeepLIFT attributes the whole input
	# in one pass per reference either way.
	attr_window = parameters["attr_window"]
	if attr_window > X.shape[-1]:
		raise ValueError(
			"attr_window ({}) is wider than the extracted sequence ({}); "
			"lower attr_window or raise in_window"
			.format(attr_window, X.shape[-1]))

	mid = X.shape[-1] // 2
	start = mid - attr_window // 2
	end = start + attr_window

	if algorithm == "deep_lift_shap":
		X_attr = deep_lift_shap(
			wrapper,
			X,
			hypothetical=True,
			n_shuffles=parameters["n_shuffles"],
			batch_size=parameters["batch_size"],
			warning_threshold=parameters["warning_threshold"],
			additional_nonlinear_ops=attribution_ops(),
			dtype=parameters["dtype"],
			device=parameters["device"],
			random_state=parameters["random_state"],
			verbose=parameters["verbose"],
		)[:, :, start:end].float()
	else:
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
