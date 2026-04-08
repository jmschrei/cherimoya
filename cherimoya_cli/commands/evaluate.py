# cherimoya_cli evaluate command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def add_parser(subparsers):
	parser = subparsers.add_parser("evaluate",
		help="Evaluate a trained Cherimoya model.")
	parser.add_argument("-p", "--parameters", type=str, required=True,
		help="A JSON file containing the parameters for making predictions.")
	return parser


def run(args):
	import sys

	import torch

	from bpnetlite.bpnet import ControlWrapper
	from bpnetlite.performance import calculate_performance_measures
	from tangermeme.io import extract_loci
	from tangermeme.predict import predict

	from ..defaults import default_evaluate_parameters
	from ..utils import merge_parameters

	parameters = merge_parameters(args.parameters, default_evaluate_parameters)
	if parameters['skip']:
		sys.exit()

	measure_names = ['profile_mnll', 'profile_jsd', 'profile_pearson',
		'profile_spearman', 'count_pearson', 'count_spearman', 'count_mse']

	###

	model = torch.load(parameters['model'], weights_only=False).to(
		parameters['device'])

	examples = extract_loci(
		sequences=parameters['sequences'],
		signals=parameters['signals'],
		in_signals=parameters['controls'],
		loci=parameters['loci'],
		chroms=parameters['chroms'],
		in_window=parameters['in_window'],
		out_window=parameters['out_window'],
		exclusion_lists=parameters['exclusion_lists'],
		max_jitter=0,
		ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
		verbose=parameters['verbose']
	)

	if parameters['controls'] == None:
		X, y = examples
		X_ctl = None
		if model.n_control_tracks > 0:
			model = ControlWrapper(model)
	else:
		X, y, X_ctl = examples
		X_ctl = (X_ctl,)

	y_hat_logits, y_hat_logcounts = predict(model, X, args=X_ctl,
		batch_size=parameters['batch_size'], device=parameters['device'],
		dtype=parameters['dtype'], verbose=parameters['verbose'])

	if parameters['reverse_complement_average']:
		X_rc = torch.flip(X, dims=(-1, -2))
		X_ctl_rc = None if X_ctl is None else (torch.flip(X_ctl[0], dims=(-1, -2)),)

		y_hat_logits_rc, y_hat_logcounts_rc = predict(model, X_rc, args=X_ctl_rc,
			batch_size=parameters['batch_size'], device=parameters['device'],
			dtype=parameters['dtype'], verbose=parameters['verbose'])

		y_hat_logits_rc = torch.flip(y_hat_logits_rc, dims=(-1, -2))
		y_hat_logits = (y_hat_logits + y_hat_logits_rc) / 2
		y_hat_logcounts = (y_hat_logcounts + y_hat_logcounts_rc) / 2

	measures = calculate_performance_measures(y_hat_logits, y,
		y_hat_logcounts, kernel_sigma=7, kernel_width=81)

	with open(parameters['performance_filename'], "w") as outfile:
		outfile.write("\t".join(measure_names))
		outfile.write("\n")
		outfile.write("\t".join([str(measures[name].mean().item())
			for name in measure_names]))

	if parameters['verbose']:
		print("\t".join(measure_names))
		print("\t".join([str(measures[name].mean().item())
			for name in measure_names]))
