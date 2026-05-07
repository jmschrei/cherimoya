# cherimoya_cli pipeline command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def add_parser(subparsers):
	parser = subparsers.add_parser("pipeline",
		help="Run each step on the given files.")
	parser.add_argument("-p", "--parameters", type=str, required=True,
		help="A JSON file containing the parameters used for each step.")
	return parser


def _validate_inputs(parameters):
	"""Validate the inputs to the pipeline before any expensive work.

	Checks that local files referenced by the JSON exist. Remote files
	(http://, https://, gs://, s3://) are skipped because resolving them
	requires network access. Raises FileNotFoundError listing every
	missing path so users see all problems at once.
	"""

	import os

	def _is_local(path):
		return not path.startswith(("http://", "https://", "gs://", "s3://"))

	missing = []

	def _check(path, label):
		if path is None:
			return
		if isinstance(path, (list, tuple)):
			for p in path:
				_check(p, label)
			return
		if _is_local(path) and not os.path.exists(path):
			missing.append("{}: {}".format(label, path))

	_check(parameters.get('sequences'), 'sequences')
	_check(parameters.get('loci'), 'loci')
	_check(parameters.get('negatives'), 'negatives')
	_check(parameters.get('signals'), 'signals')
	_check(parameters.get('controls'), 'controls')
	_check(parameters.get('motifs'), 'motifs')
	_check(parameters.get('exclusion_lists'), 'exclusion_lists')

	if missing:
		raise FileNotFoundError(
			"Pipeline cannot start; the following inputs are missing:\n  - "
			+ "\n  - ".join(missing)
		)


def run(args):
	import argparse
	import json
	import os
	import subprocess
	import sys

	import pandas

	from . import attribute as attribute_cmd
	from . import fit as fit_cmd
	from . import marginalize as marginalize_cmd
	from . import negatives as negatives_cmd
	from . import seqlets as seqlets_cmd
	from ..defaults import (
		default_fit_parameters,
		default_attribute_parameters,
		default_seqlet_parameters,
		default_annotation_parameters,
		default_marginalize_parameters,
		default_pipeline_parameters,
	)
	from ..utils import _extract_set, _check_set, merge_parameters

	parameters = merge_parameters(args.parameters, default_pipeline_parameters)
	preprocess_parameters = merge_parameters(
		parameters['preprocessing_parameters'],
		default_pipeline_parameters['preprocessing_parameters']
	)

	_validate_inputs(parameters)

	pname = parameters['name']

	def _run_step(cmd_fn, json_path):
		"""Invoke a CLI step in-process by calling its run(args) directly."""
		if not parameters['dry_run']:
			cmd_fn(argparse.Namespace(parameters=json_path))

	###
	# Step 0.1: Run MACS3 to call peaks if not provided
	###

	if parameters['loci'] is None:
		if preprocess_parameters['verbose']:
			print("\nStep 0.1: Call peaks using MACS3.")

		if preprocess_parameters['callpeaks_format'] is None:
			if preprocess_parameters['fragments']:
				file_format = 'FRAG'
			else:
				fname = parameters['signals'][0]

				if fname.endswith('.gz'):
					file_format = fname.split(".")[-2].upper()
				else:
					file_format = fname.split(".")[-1].upper()

				if preprocess_parameters['paired_end']:
					file_format += 'PE'

			preprocess_parameters['callpeaks_format'] = file_format


		cmd_args = [
			"macs3", "callpeak",
			"-f", preprocess_parameters['callpeaks_format'],
			"-g", str(preprocess_parameters['callpeaks_gsize']),
			"-n", pname,
			"-q", str(preprocess_parameters['callpeaks_q']),
			"-t"
		]

		cmd_args.extend(parameters['signals'])

		if parameters['controls'] is not None:
			cmd_args += ['-c']
			cmd_args.extend(parameters['controls'])

		if preprocess_parameters['fragments']:
			cmd_args += ['--max-count', '1']

		parameters['loci'] = [pname + '_peaks.narrowPeak']

		if not parameters['dry_run']:
			subprocess.run(cmd_args, check=True)


	###
	# Step 0.2: Convert from SAM/BAMs to bigwigs if provided
	###

	ftypes = '.sam', '.bam', '.bed', '.bed.gz', '.tsv', '.tsv.gz'

	if parameters['signals'][0].endswith(ftypes):
		if preprocess_parameters['verbose']:
			print("Step 0.2: Convert data to bigWigs")

		cmd_args = [
			"bam2bw",
			"-s", parameters['sequences'],
			"-n", pname,
			"-ps", str(preprocess_parameters['pos_shift']),
			"-ns", str(preprocess_parameters['neg_shift']),
			"-sf", str(preprocess_parameters['scale_factor']),
			"-p", "-1",
		]

		if preprocess_parameters['read_depth']:
			cmd_args += ["-r"]

		if preprocess_parameters["unstranded"]:
			cmd_args += ["-u"]

		if preprocess_parameters['fragments']:
			cmd_args += ["-f"]

		if preprocess_parameters["verbose"]:
			cmd_args += ["-v"]

		cmd_args += parameters['signals']
		if not parameters['dry_run']:
			subprocess.run(cmd_args, check=True)

		if preprocess_parameters["unstranded"]:
			parameters['signals'] = [pname + ".bw"]
		else:
			parameters['signals'] = [pname + ".+.bw", pname + ".-.bw"]


	if parameters['controls'] is not None:
		if parameters['controls'][0].endswith(ftypes):
			cmd_args = [
				"bam2bw",
				"-s", parameters['sequences'],
				"-n", pname + ".control",
				"-ps", str(preprocess_parameters['pos_shift']),
				"-ns", str(preprocess_parameters['neg_shift']),
				"-p", "-1"
			]

			if preprocess_parameters['read_depth']:
				cmd_args += ["-r"]

			if preprocess_parameters["unstranded"]:
				cmd_args += ["-u"]

			if preprocess_parameters["fragments"]:
				cmd_args += ["-f"]

			if preprocess_parameters["verbose"]:
				cmd_args += ["-v"]

			cmd_args += parameters['controls']
			if not parameters['dry_run']:
				subprocess.run(cmd_args, check=True)

			if preprocess_parameters["unstranded"]:
				parameters['controls'] = [pname + ".control.bw"]
			else:
				parameters['controls'] = [pname + ".control.+.bw", pname + ".control.-.bw"]


	###
	# Step 0.3: Identify GC-matched negative regions
	###

	if parameters['negatives'] is None:
		if preprocess_parameters['verbose']:
			print("\nStep 0.3: Find GC-matched negative regions.")

		negatives_args = argparse.Namespace(
			peaks=parameters["loci"][0],
			fasta=parameters["sequences"],
			bigwig=None,
			output=pname + ".negatives.bed",
			bin_width=0.02,
			max_n_perc=0.1,
			beta=0.5,
			in_window=parameters['in_window'],
			out_window=parameters['out_window'],
			verbose=preprocess_parameters['verbose'],
		)

		parameters["negatives"] = [pname + ".negatives.bed"]

		if not parameters['dry_run']:
			negatives_cmd.run(negatives_args)


	###
	# Step 1: Fit a Cherimoya model to the provided data
	###

	if parameters['verbose']:
		print("\nStep 1: Fitting a Cherimoya model")

	fit_parameters = _extract_set(parameters, default_fit_parameters,
		'fit_parameters')

	if parameters.get('model', None) == None:
		name = pname + '.fit.json'
		parameters['model'] = pname + '.torch'
		_check_set(fit_parameters, 'performance_filename', pname + '.performance.tsv')

		with open(name, 'w') as outfile:
			outfile.write(json.dumps(fit_parameters, sort_keys=True, indent=4))

		_run_step(fit_cmd.run, name)


	###
	# Step 2: Calculate attributions
	###

	if parameters['verbose']:
		print("\nStep 2: Calculating attributions")

	attribute_parameters = _extract_set(parameters,
		default_attribute_parameters, 'attribute_parameters')
	_check_set(attribute_parameters, 'ohe_filename',  pname+'.attributions.ohe.npz')
	_check_set(attribute_parameters, 'attr_filename', pname+'.attributions.attr.npz')
	_check_set(attribute_parameters, 'idx_filename',  pname+'.attributions.idxs.npy')

	name = '{}.attribute.json'.format(parameters['name'])
	with open(name, 'w') as outfile:
		outfile.write(json.dumps(attribute_parameters, sort_keys=True, indent=4))

	_run_step(attribute_cmd.run, name)


	###
	# Step 3.1: Identify seqlets from attributions
	###

	if parameters['verbose']:
		print("\nStep 3.1: Seqlet identification")

	seqlet_parameters = _extract_set(parameters,
		default_seqlet_parameters, 'seqlet_parameters')
	_check_set(seqlet_parameters, "ohe_filename",  pname+'.attributions.ohe.npz')
	_check_set(seqlet_parameters, "attr_filename", pname+'.attributions.attr.npz')
	_check_set(seqlet_parameters, "idx_filename",  pname+'.attributions.idxs.npy')
	_check_set(seqlet_parameters, "output_filename", pname+".seqlets.bed")
	_check_set(seqlet_parameters, "chroms", attribute_parameters['chroms'])

	name = '{}.seqlets.json'.format(parameters['name'])
	with open(name, 'w') as outfile:
		outfile.write(json.dumps(seqlet_parameters, sort_keys=True, indent=4))

	_run_step(seqlets_cmd.run, name)


	###
	# Step 3.2: Annotate seqlets using motif database
	###

	annotation_parameters = _extract_set(parameters,
		default_annotation_parameters, "annotation_parameters")
	_check_set(annotation_parameters, "seqlet_filename", pname+".seqlets.bed")
	_check_set(annotation_parameters, "output_filename", pname+".seqlets_annotated.bed")
	_check_set(annotation_parameters, "motifs", parameters["motifs"])

	annotation_parameters = merge_parameters(annotation_parameters,
		default_annotation_parameters)

	if annotation_parameters['motifs'] is not None:
		if parameters['verbose']:
			print("\nStep 3.2: Seqlet annotation")

		cmd = ["ttl"]
		cmd += ["-f", annotation_parameters["sequences"]]
		cmd += ["-b", annotation_parameters["seqlet_filename"]]
		cmd += ["-s", str(annotation_parameters["n_score_bins"])]
		cmd += ["-m", str(annotation_parameters["n_median_bins"])]
		cmd += ["-a", str(annotation_parameters["n_target_bins"])]
		cmd += ["-c", str(annotation_parameters["n_cache"])]
		cmd += ["-j", str(annotation_parameters["n_jobs"])]

		if not annotation_parameters["reverse_complement"]:
			cmd += ["-r"]

		if annotation_parameters['motifs'] is not None:
			cmd += ["-t", annotation_parameters["motifs"]]

		if not parameters['dry_run']:
			with open(annotation_parameters['output_filename'], "w") as f:
				subprocess.run(cmd, check=True, stdout=f)

		annotated_seqlets = pandas.read_csv(annotation_parameters['output_filename'],
			sep="\t", header=None, usecols=(3,), names=['motifs'])

		seqlet_count = annotated_seqlets.value_counts()
		seqlet_count.to_csv(pname+".motif_seqlet_count.tsv", sep="\t")


	###
	# Step 4.1: Run TF-MoDISco
	###

	if parameters['verbose']:
		print("\nStep 4.1: TF-MoDISco motifs")

	modisco_parameters = parameters['modisco_motifs_parameters']

	_check_set(modisco_parameters, "output_filename",
		pname+'_modisco_results.h5')
	_check_set(modisco_parameters, "verbose",
		parameters['verbose'])

	modisco_parameters = merge_parameters(modisco_parameters,
		default_pipeline_parameters['modisco_motifs_parameters'])

	cmd = "modisco motifs -s {} -a {} -n {} -o {}".format(
		attribute_parameters['ohe_filename'],
		attribute_parameters['attr_filename'],
		modisco_parameters['n_seqlets'],
		modisco_parameters['output_filename'])

	if 'verbose' in modisco_parameters and modisco_parameters['verbose']:
		cmd += ' -v'
	elif parameters['verbose']:
		cmd += ' -v'

	if not parameters['dry_run']:
		subprocess.run(cmd.split(), check=True)


	###
	# Step 4.2: Generate the tf-modisco report
	###

	report_parameters = parameters['modisco_report_parameters']
	_check_set(report_parameters, "verbose", parameters["verbose"])
	_check_set(report_parameters, "output_folder", pname+"_modisco/")
	_check_set(report_parameters, "motifs", parameters['motifs'])

	if report_parameters['verbose']:
		print("\nStep 4.2: TF-MoDISco reports")

	if not parameters['dry_run']:
		if report_parameters['motifs'] is not None:
			subprocess.run(["modisco", "report",
				"-i", modisco_parameters['output_filename'],
				"-o", report_parameters['output_folder'],
				"-s", './',
				"-m", report_parameters['motifs']
				], check=True)
		else:
			subprocess.run(["modisco", "report",
				"-i", modisco_parameters['output_filename'],
				"-o", report_parameters['output_folder'],
				"-s", './'
				], check=True)


	###
	# Step 5: Marginalization experiments
	###

	if parameters['motifs'] is None:
		sys.exit()

	if parameters['verbose']:
		print("\nStep 5: Run marginalizations")

	marginalize_parameters = _extract_set(parameters,
		default_marginalize_parameters, "marginalize_parameters")

	_check_set(marginalize_parameters, "loci", parameters["negatives"])
	_check_set(marginalize_parameters, 'output_filename', pname+"_marginalize/")
	_check_set(marginalize_parameters, 'motifs', parameters['motifs'])
	_check_set(marginalize_parameters, 'negatives', parameters['negatives'])

	name = '{}.marginalize.json'.format(parameters['name'])

	with open(name, 'w') as outfile:
		outfile.write(json.dumps(marginalize_parameters, sort_keys=True,
			indent=4))

	_run_step(marginalize_cmd.run, name)
