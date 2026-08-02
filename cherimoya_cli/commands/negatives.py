# cherimoya_cli negatives command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


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
        n_jobs=1,
    )

    matched_loci.to_csv(args.output, header=False, sep="\t", index=False)
