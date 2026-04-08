# cherimoya_cli entry point
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import argparse

from cherimoya import __version__

from .commands import negatives
from .commands import pipeline_json
from .commands import fit
from .commands import evaluate
from .commands import attribute
from .commands import seqlets
from .commands import marginalize
from .commands import pipeline
from .commands import batch


desc = """A command-line tool for the training and usage of Cherimoya models."""

_help = """Must be either 'negatives', 'fit', 'evaluate',
	'attribute', 'seqlets', 'marginalize', or 'pipeline'."""


def main():
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--version', action='version',
		version='cherimoya {}'.format(__version__))
	subparsers = parser.add_subparsers(help=_help, required=True, dest='cmd')

	negatives.add_parser(subparsers)
	pipeline_json.add_parser(subparsers)
	batch.add_parser(subparsers)
	fit.add_parser(subparsers)
	evaluate.add_parser(subparsers)
	attribute.add_parser(subparsers)
	seqlets.add_parser(subparsers)
	marginalize.add_parser(subparsers)
	pipeline.add_parser(subparsers)

	args = parser.parse_args()

	commands = {
		'negatives': negatives.run,
		'pipeline-json': pipeline_json.run,
		'batch': batch.run,
		'fit': fit.run,
		'evaluate': evaluate.run,
		'attribute': attribute.run,
		'seqlets': seqlets.run,
		'marginalize': marginalize.run,
		'pipeline': pipeline.run,
	}

	commands[args.cmd](args)


if __name__ == '__main__':
	main()
