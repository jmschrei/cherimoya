# cherimoya_cli marginalize command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def add_parser(subparsers):
	parser = subparsers.add_parser("marginalize",
		help="Run marginalizations given motifs.")
	parser.add_argument("-p", "--parameters", type=str, required=True,
		help="A JSON file containing the parameters for running marginalizations.")
	return parser


def run(args):
	import sys

	import numpy
	import torch

	from bpnetlite.bpnet import ControlWrapper
	from bpnetlite.marginalize import marginalization_report
	from tangermeme.io import extract_loci

	from ..defaults import default_marginalize_parameters
	from ..utils import merge_parameters

	parameters = merge_parameters(args.parameters,
		default_marginalize_parameters)
	if parameters['skip']:
		sys.exit()

	###

	model = torch.load(parameters['model'], weights_only=False).to(
		parameters['device'])

	if model.n_control_tracks > 0:
		model = ControlWrapper(model)

	X = extract_loci(
		sequences=parameters['sequences'],
		loci=parameters['loci'],
		chroms=parameters['chroms'],
		max_jitter=0,
		ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
		n_loci=parameters['n_loci'],
		verbose=parameters['verbose']
	).float()

	if parameters['shuffle'] == True:
		idxs = numpy.arange(X.shape[0])
		numpy.random.shuffle(idxs)
		X = X[idxs]

	if parameters['n_loci'] is not None:
		X = X[:parameters['n_loci']]

	marginalization_report(model, parameters['motifs'], X,
		parameters['output_filename'],
		attributions=parameters['attributions'],
		batch_size=parameters['batch_size'],
		minimal=parameters['minimal'],
		device=parameters['device'],
		verbose=parameters['verbose'])
