# cherimoya_cli negatives command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def add_parser(subparsers):
	parser = subparsers.add_parser("negatives",
		help="Sample GC-matched negatives.")
	parser.add_argument("-i", "--peaks", required=True,
		help="Peak bed file.")
	parser.add_argument("-f", "--fasta", help="Genome FASTA file.")
	parser.add_argument("-b", "--bigwig", help="Optional signal bigwig.")
	parser.add_argument("-o", "--output", required=True,
		help="Output bed file.")
	parser.add_argument("-l", "--bin_width", type=float, default=0.02,
		help="GC bin width to match.")
	parser.add_argument("-n", "--max_n_perc", type=float, default=0.1,
		help="Maximum percentage of Ns allowed in each locus.")
	parser.add_argument("-a", "--beta", type=float, default=0.5,
		help="Multiplier on the minimum counts in peaks.")
	parser.add_argument("-w", "--in_window", type=int, default=2114,
		help="Width for calculating GC content.")
	parser.add_argument("-x", "--out_window", type=int, default=1000,
		help="Non-overlapping stride to use for loci.")
	parser.add_argument("-v", "--verbose", default=False,
		action='store_true')
	return parser


def run(args):
	from tangermeme.match import extract_matching_loci

	matched_loci = extract_matching_loci(
		loci=args.peaks,
		fasta=args.fasta,
		gc_bin_width=args.bin_width,
		max_n_perc=args.max_n_perc,
		bigwig=args.bigwig,
		signal_beta=args.beta,
		in_window=args.in_window,
		out_window=args.out_window,
		chroms=None,
		verbose=args.verbose,
		n_jobs=1
	)

	matched_loci.to_csv(args.output, header=False, sep='\t', index=False)
