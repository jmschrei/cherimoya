# cherimoya_cli pipeline-json command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def add_parser(subparsers):
	parser = subparsers.add_parser("pipeline-json",
		help="Make a pipeline JSON file given the provided information.")
	parser.add_argument("-s", "--sequences", type=str,
		help="The FASTA file of sequences.")
	parser.add_argument("-i", "--inputs", type=str, action='append',
		help="A BAM or bigwig file. Repeatable.")
	parser.add_argument("-c", "--controls", type=str, action='append',
		help="A BAM or bigwig file. Repeatable.")
	parser.add_argument("-p", "--peaks", type=str, action='append',
		help="A BED-formatted file of peaks to use. Repeatable.")
	parser.add_argument("-neg", "--negatives", type=str, action='append',
		help="A BED-formatted file of negative loci to use. Repeatable.")
	parser.add_argument("-n", "--name", type=str,
		help="Name to use as a suffix in intermediary files.")
	parser.add_argument("-u", "--unstranded", action='store_true',
		default=False, help="Whether the input is stranded")
	parser.add_argument("-f", "--fragments", action='store_true',
		default=False, help='Whether the input are fragments or reads.')
	parser.add_argument("-ps", "--pos_shift", type=int,
		default=0, help="How many bp to shift the + strand reads.")
	parser.add_argument("-ns", "--neg_shift", type=int,
		default=0, help="how many bp to shift the - strand reads.")
	parser.add_argument("-m", "--motifs", type=str, default=None,
		help="A motif database for marginalization and TF-MoDISco.")
	parser.add_argument("-o", "--output", type=str,
		help="The filename for the pipeline JSON.")
	parser.add_argument("-pe", "--paired_end", action='store_true',
		default=False, help="Whether the input is paired-end.")
	return parser


def run(args):
	import sys
	import json
	import copy

	from ..defaults import default_pipeline_parameters

	parameters = copy.deepcopy(default_pipeline_parameters)

	parameters['sequences'] = args.sequences
	parameters['loci'] = args.peaks
	parameters['negatives'] = args.negatives
	parameters['signals'] = args.inputs
	parameters['controls'] = args.controls
	parameters['name'] = args.name
	parameters['motifs'] = args.motifs
	parameters['preprocessing_parameters']['unstranded'] = args.unstranded
	parameters['preprocessing_parameters']['fragments'] = args.fragments
	parameters['preprocessing_parameters']['pos_shift'] = args.pos_shift
	parameters['preprocessing_parameters']['neg_shift'] = args.neg_shift

	with open(args.output, 'w') as outfile:
		outfile.write(json.dumps(parameters, indent=4))

	sys.exit()
