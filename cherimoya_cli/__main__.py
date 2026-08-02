# cherimoya_cli entry point
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import argparse
import importlib
from importlib.metadata import version, PackageNotFoundError

desc = """A command-line tool for the training and usage of Cherimoya models."""

_help = """Must be either 'negatives', 'fit', 'evaluate',
    'attribute', 'seqlets', 'marginalize', 'pipeline', or 'install-skill'."""

try:
    __version__ = version("cherimoya")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


def _setup_parsers() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--version",
        action="version",
        version="cherimoya {}".format(__version__),
    )
    subparsers = parser.add_subparsers(help=_help, required=True, dest="cmd")

    # Negatives
    negatives_parser = subparsers.add_parser(
        "negatives", help="Sample GC-matched negatives."
    )
    negatives_parser.add_argument("-i", "--peaks", required=True, help="Peak bed file.")
    negatives_parser.add_argument("-f", "--fasta", help="Genome FASTA file.")
    negatives_parser.add_argument("-b", "--bigwig", help="Optional signal bigwig.")
    negatives_parser.add_argument(
        "-o", "--output", required=True, help="Output bed file."
    )
    negatives_parser.add_argument(
        "-l", "--bin_width", type=float, default=0.02, help="GC bin width to match."
    )
    negatives_parser.add_argument(
        "-n",
        "--max_n_perc",
        type=float,
        default=0.1,
        help="Maximum percentage of Ns allowed in each locus.",
    )
    negatives_parser.add_argument(
        "-a",
        "--beta",
        type=float,
        default=0.5,
        help="Multiplier on the minimum counts in peaks.",
    )
    negatives_parser.add_argument(
        "-w",
        "--in_window",
        type=int,
        default=2114,
        help="Width for calculating GC content.",
    )
    negatives_parser.add_argument(
        "-x",
        "--out_window",
        type=int,
        default=1000,
        help="Non-overlapping stride to use for loci.",
    )
    negatives_parser.add_argument("-v", "--verbose", default=False, action="store_true")

    # Pipeline JSON
    pipeline_json_parser = subparsers.add_parser(
        "pipeline-json",
        help="Make a pipeline JSON file given the provided information.",
    )
    pipeline_json_parser.add_argument(
        "-s", "--sequences", type=str, help="The FASTA file of sequences."
    )
    pipeline_json_parser.add_argument(
        "-i",
        "--inputs",
        type=str,
        action="append",
        help="A BAM or bigwig file. Repeatable.",
    )
    pipeline_json_parser.add_argument(
        "-c",
        "--controls",
        type=str,
        action="append",
        help="A BAM or bigwig file. Repeatable.",
    )
    pipeline_json_parser.add_argument(
        "-p",
        "--peaks",
        type=str,
        action="append",
        help="A BED-formatted file of peaks to use. Repeatable.",
    )
    pipeline_json_parser.add_argument(
        "-neg",
        "--negatives",
        type=str,
        action="append",
        help="A BED-formatted file of negative loci to use. Repeatable.",
    )
    pipeline_json_parser.add_argument(
        "-n", "--name", type=str, help="Name to use as a suffix in intermediary files."
    )
    pipeline_json_parser.add_argument(
        "-u",
        "--unstranded",
        action="store_true",
        default=False,
        help="Whether the input is stranded",
    )
    pipeline_json_parser.add_argument(
        "-f",
        "--fragments",
        action="store_true",
        default=False,
        help="Whether the input are fragments or reads.",
    )
    pipeline_json_parser.add_argument(
        "-ps",
        "--pos_shift",
        type=int,
        default=0,
        help="How many bp to shift the + strand reads.",
    )
    pipeline_json_parser.add_argument(
        "-ns",
        "--neg_shift",
        type=int,
        default=0,
        help="how many bp to shift the - strand reads.",
    )
    pipeline_json_parser.add_argument(
        "-m",
        "--motifs",
        type=str,
        default=None,
        help="A motif database for marginalization and TF-MoDISco.",
    )
    pipeline_json_parser.add_argument(
        "-o", "--output", type=str, help="The filename for the pipeline JSON."
    )
    pipeline_json_parser.add_argument(
        "-pe",
        "--paired_end",
        action="store_true",
        default=False,
        help="Whether the input is paired-end.",
    )
    pipeline_json_parser.add_argument(
        "-sf",
        "--scale_factor",
        default=1,
        help="Whether to scale the read counts. 1 is no scaling.",
    )

    # Fit
    fit_parser = subparsers.add_parser("fit", help="Fit a Cherimoya model.")
    fit_parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        required=True,
        help="A JSON file containing the parameters for fitting the model.",
    )

    # Evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a trained Cherimoya model."
    )
    evaluate_parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        required=True,
        help="A JSON file containing the parameters for making predictions.",
    )

    # Attribute
    attribute_parser = subparsers.add_parser(
        "attribute", help="Calculate attributions using a trained Cherimoya model."
    )
    attribute_parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        required=True,
        help="A JSON file containing the parameters for calculating attributions.",
    )

    # Seqlets
    seqlets_parser = subparsers.add_parser(
        "seqlets", help="Identify seqlets from attributions."
    )
    seqlets_parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        required=True,
        help="A JSON file containing the parameters for identifying seqlets.",
    )

    # Marginalize
    marginalize_parser = subparsers.add_parser(
        "marginalize", help="Run marginalizations given motifs."
    )
    marginalize_parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        required=True,
        help="A JSON file containing the parameters for running marginalizations.",
    )

    # Pipeline
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run each step on the given files."
    )
    pipeline_parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        required=True,
        help="A JSON file containing the parameters used for each step.",
    )

    # Batch
    batch_parser = subparsers.add_parser(
        "batch", help="Run the pipeline in parallel using multiple GPUs."
    )
    batch_parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        required=True,
        help="A JSON file containing the parameters for fitting the model.",
    )

    # Install skill
    install_skill_parser = subparsers.add_parser(
        "install-skill",
        help="Install the bundled Cherimoya agent skill for Claude Code.",
    )
    install_skill_parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=None,
        help="Skills directory to install into. Default is ~/.claude/skills.",
    )
    install_skill_parser.add_argument(
        "--symlink",
        action="store_true",
        default=False,
        help="Symlink the packaged skill instead of copying it. Reflects "
        "in-place edits, but breaks if the install location moves.",
    )
    install_skill_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite an existing installation at the destination.",
    )

    return parser


def main():
    parser = _setup_parsers()
    args = parser.parse_args()

    COMMANDS = {
        "negatives": "negatives",
        "pipeline-json": "pipeline_json",
        "batch": "batch",
        "fit": "fit",
        "evaluate": "evaluate",
        "attribute": "attribute",
        "seqlets": "seqlets",
        "marginalize": "marginalize",
        "pipeline": "pipeline",
        "install-skill": "install_skill",
    }

    modname = COMMANDS.get(args.cmd)
    if modname is None:
        parser.error(f"unknown command: {args.cmd}")
    mod = importlib.import_module(f".commands.{modname}", package=__package__)
    mod.run(args)


if __name__ == "__main__":
    main()
