# cherimoya_cli utilities
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import json


def _extract_set(parameters, defaults, name):
	subparameters = {
		key: parameters.get(key, None) for key in defaults if key in parameters
	}

	for parameter, value in parameters[name].items():
		if value is not None:
			subparameters[parameter] = value

	return subparameters


def _check_set(parameters, parameter, value):
	if parameters.get(parameter, None) == None:
		parameters[parameter] = value


def merge_parameters(parameters, default_parameters):
	"""Merge the provided parameters with the default parameters.


	Parameters
	----------
	parameters: str
		Name of the JSON folder with the provided parameters

	default_parameters: dict
		The default parameters for the operation.


	Returns
	-------
	params: dict
		The merged set of parameters.
	"""

	if isinstance(parameters, str):
		if not os.path.exists(parameters):
			raise FileNotFoundError("Parameter file not found: '{}'"
				.format(parameters))

		with open(parameters, "r") as infile:
			parameters = json.load(infile)

	unset_parameters = ("controls", "warning_threshold", "early_stopping",
		"count_loss_weight", "exclusion_lists")
	for parameter, value in default_parameters.items():
		if parameter not in parameters:
			if value is None and parameter not in unset_parameters:
				raise ValueError("Must provide value for '{}'".format(parameter))

			parameters[parameter] = value

	return parameters
