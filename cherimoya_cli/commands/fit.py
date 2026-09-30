# cherimoya_cli fit command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def _max_epochs_for_min_steps(max_epochs, steps_per_epoch, min_total_steps):
	"""Raise `max_epochs` until the run reaches `min_total_steps` steps.

	An epoch is one pass over the peaks, so `max_epochs` buys a number of
	optimizer steps proportional to how many peaks an experiment has -- an
	experiment with 14 batches of peaks gets 280 steps out of 20 epochs while
	one with 2,700 batches gets 54,000. This puts a floor under that by
	extending the run, and returns `max_epochs` unchanged when the floor is
	already met or `min_total_steps` is None.


	Parameters
	----------
	max_epochs: int
		The number of epochs requested in the parameters.

	steps_per_epoch: int
		Batches in one pass over the training data, i.e. `len(training_data)`.

	min_total_steps: int or None
		The minimum number of optimizer steps the run should take. None
		disables the floor.


	Returns
	-------
	max_epochs: int
		The number of epochs to train for.
	"""

	if min_total_steps is None or steps_per_epoch <= 0:
		return max_epochs

	if steps_per_epoch * max_epochs >= min_total_steps:
		return max_epochs

	return -(-min_total_steps // steps_per_epoch)


def _loss_balance_summary(loss_weights, lw_lr, lw_wd, lw_momentum):
	"""Describe how the profile and count losses are balanced.

	The two loss terms are balanced either by the learned Kendall weights
	`lw0` and `lw1`, which the `lw_optimizer` trains, or by the constants
	given in `loss_weights`. In the second case `lw0` and `lw1` stop
	receiving gradient and the optimizer takes no effective step, so
	reporting its learning rate describes something that is not happening;
	the constants actually in use are reported instead.


	Parameters
	----------
	loss_weights: tuple or None
		The fixed `(profile, count)` weights, or None to use the learned
		Kendall weights.

	lw_lr: float
		The learning rate of the `lw_optimizer`.

	lw_wd: float
		The weight decay of the `lw_optimizer`.

	lw_momentum: float
		The momentum of the `lw_optimizer`.


	Returns
	-------
	summary: str
		One line describing the loss balancing, for `verbose` output.
	"""

	if loss_weights is None:
		return "SGD Optimizer (lw): lr={}, wd={}, momentum={}".format(
			lw_lr, lw_wd, lw_momentum
		)

	w0, w1 = loss_weights
	return (
		"Fixed Loss Weights: profile={}, count={} "
		"(learned lw0/lw1 disabled)".format(w0, w1)
	)


def run(args):
	import argparse
	import copy
	import hashlib
	import itertools
	import os
	import json

	os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

	import numpy
	import torch
	import lightning
	from lightning.pytorch.utilities import rank_zero_only

	from cherimoya import Cherimoya
	from cherimoya.io import PeakGenerator, normalize_signal_groups
	from cherimoya.training import fit

	from tangermeme.io import _interleave_loci
	from tangermeme.io import extract_loci

	from . import evaluate as evaluate_cmd
	from ..defaults import default_fit_parameters
	from ..utils import merge_parameters

	# With more than one device, Lightning starts every rank after the
	# first by re-running this command, so everything up to the fit runs
	# once per rank. Only rank 0 prints.
	say = rank_zero_only(print)

	parameters = merge_parameters(args.parameters, default_fit_parameters)
	if parameters["skip"]:
		return

	# A chromosome in two splits would score the model on data it was
	# trained on or selected with.
	splits = {name: parameters[name] or [] for name in ("training_chroms",
		"validation_chroms", "test_chroms")}
	for (a, chroms_a), (b, chroms_b) in itertools.combinations(splits.items(), 2):
		shared = sorted(set(chroms_a) & set(chroms_b))
		if shared:
			raise ValueError("{} and {} share {}. Give each chromosome to at "
				"most one split, or set test_chroms to null to skip the test "
				"evaluation.".format(a, b, shared))

	# Resolve the seed before anything draws from an RNG. A null
	# `random_state` means "pick one and tell me" rather than "stay
	# unseeded": the run still varies between invocations, but the seed
	# that produced it is printed and written into the evaluate JSONs, so
	# the run can be repeated afterwards. Drawing it here and storing it
	# back into `parameters` is what puts it in those JSONs, which are
	# deepcopies of this dict. The ranks Lightning launches after the first
	# read rank 0's draw from `PL_GLOBAL_SEED`, which `seed_everything`
	# sets before they start; rank 0 itself always draws, so a value left
	# in the environment by an earlier run in the same process is ignored.
	# Under `srun` every rank starts at once, so there is no draw to
	# inherit, and each derives the same seed from the job step instead.
	if parameters["random_state"] is None:
		seed = None
		if int(os.environ.get("LOCAL_RANK", 0)) > 0:
			seed = os.environ.get("PL_GLOBAL_SEED")
		elif int(os.environ.get("SLURM_NTASKS", 1)) > 1:
			step = "{}.{}".format(os.environ.get("SLURM_JOB_ID"),
				os.environ.get("SLURM_STEP_ID", 0))
			seed = int(hashlib.sha256(step.encode()).hexdigest(), 16) % (2**31 - 1)
			say("Derived random_state={} from SLURM job step {}; set it in the "
				"JSON to repeat this run.".format(seed, step))

		if seed is None:
			seed = int(numpy.random.randint(0, 2**31 - 1))

			# Printed whether or not `verbose` is set: a drawn seed is
			# the one part of the run that cannot be recovered afterwards
			# if training dies before the evaluate JSONs are written.
			say("Drew random_state={}; set it in the JSON to repeat this run."
				.format(seed))

		parameters["random_state"] = int(seed)

	# The sampler and the model each take the seed directly; this covers
	# everything else that training touches.
	lightning.seed_everything(parameters["random_state"], verbose=False)

	# Resolve grouped/flat signal specs into a flat list of files plus
	# the per-group sizes. The flat list is what extract_loci and the
	# `bam2bw`-style tooling need; the group sizes determine the
	# channel permutation used under RC and the number of count
	# predictions. ``parameters`` keeps the structured form (e.g.
	# ``[[plus.bw, minus.bw]]``) because the evaluate JSONs are copied from
	# it, and evaluate reads the control grouping from there for
	# reverse-complement averaging. The signal grouping it takes from the
	# checkpoint.
	signal_files, signal_groups = normalize_signal_groups(parameters["signals"])
	control_files, control_groups = normalize_signal_groups(parameters["controls"])

	if parameters["verbose"]:
		say("Training Chroms: ", parameters["training_chroms"])
		say("Validation Chroms: ", parameters["validation_chroms"])
		say("Test Chroms: ", parameters["test_chroms"])

		say("\nLoading peaks from: ", parameters["loci"])
		say("Loading negatives from: ", parameters["negatives"])
		say("Loading sequence from: ", parameters["sequences"])
		say("Loading signal from: ", parameters["signals"])
		say("Loading controls from: ", parameters["controls"])
		say("Loading exclusion list from: ", parameters["exclusion_lists"])
		say("Random State: ", parameters["random_state"])
		say()

	###

	# The training module builds its own loader around this dataset, so
	# that it can split each batch across devices.
	training_data = PeakGenerator(
		peaks=parameters["loci"],
		negatives=parameters["negatives"],
		sequences=parameters["sequences"],
		signals=parameters["signals"],
		controls=parameters["controls"],
		chroms=parameters["training_chroms"],
		in_window=parameters["in_window"],
		out_window=parameters["out_window"],
		max_jitter=parameters["max_jitter"],
		negative_ratio=parameters["negative_ratio"],
		reverse_complement=parameters["reverse_complement"],
		summits=parameters["summits"],
		exclusion_lists=parameters["exclusion_lists"],
		random_state=parameters["random_state"],
		verbose=parameters["verbose"],
		signal_groups=signal_groups,
		control_groups=control_groups,
	).dataset

	# Centered as the training peaks are; negatives have no summit column.
	valid_data = extract_loci(
		sequences=parameters["sequences"],
		signals=signal_files,
		in_signals=control_files,
		loci=parameters["loci"],
		chroms=parameters["validation_chroms"],
		in_window=parameters["in_window"],
		out_window=parameters["out_window"],
		max_jitter=0,
		summits=parameters["summits"],
		exclusion_lists=parameters["exclusion_lists"],
		ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
		verbose=parameters["verbose"],
	)

	# Every negative on the validation chromosomes joins the validation
	# set, labeled 0, for the measures that separate peaks from negatives.
	# `extract_loci` raises when none falls on them, hence the check.
	n_valid_peaks, n_valid_negatives = len(valid_data[0]), 0
	valid_labels = None
	if parameters["negatives"] is not None and len(_interleave_loci(
		parameters["negatives"], parameters["validation_chroms"])) > 0:
		negative_data = extract_loci(
			sequences=parameters["sequences"],
			signals=signal_files,
			in_signals=control_files,
			loci=parameters["negatives"],
			chroms=parameters["validation_chroms"],
			in_window=parameters["in_window"],
			out_window=parameters["out_window"],
			max_jitter=0,
			exclusion_lists=parameters["exclusion_lists"],
			ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
			verbose=parameters["verbose"],
		)
		n_valid_negatives = len(negative_data[0])
		valid_data = [torch.cat(pair) for pair in zip(valid_data, negative_data)]
		valid_labels = torch.cat([torch.ones(n_valid_peaks),
			torch.zeros(n_valid_negatives)])

	if parameters["verbose"]:
		say("\nTraining Set Peaks: ", training_data.peak_sequences.shape[0])
		say("Training Set Negatives: ", training_data.negative_sequences.shape[0])
		say("Validation Set Size: ", n_valid_peaks)
		say("Validation Set Negatives: ", n_valid_negatives, "\n")
		say("Negative Ratio: 1:{:4.4} pos:neg\n".format(
			parameters["negative_ratio"]))

	###

	if control_files is not None:
		valid_sequences, valid_signals, valid_controls = valid_data
		n_control_tracks = len(control_files)
	else:
		valid_sequences, valid_signals = valid_data
		valid_controls = None
		n_control_tracks = 0

	trimming = (parameters["in_window"] - parameters["out_window"]) // 2

	model = Cherimoya(
		n_filters=parameters["n_filters"],
		n_layers=parameters["n_layers"],
		signal_groups=signal_groups,
		n_control_tracks=n_control_tracks,
		expansion=parameters["expansion"],
		residual_scale=parameters["residual_scale"],
		trimming=trimming,
		name=parameters["name"],
		verbose=parameters["verbose"],
		compile=parameters["compile"],
		compile_mode=parameters["compile_mode"],
		random_state=parameters["random_state"],
	)

	if parameters["verbose"]:
		say("Model has {} dilated layers and {} filters".format(
			parameters["n_layers"], parameters["n_filters"]))
		say("Model has {} trainable parameters.\n".format(
			sum(p.numel() for p in model.parameters() if p.requires_grad)))

	n_warmup_epochs = parameters["n_warmup_epochs"]
	max_epochs = parameters["max_epochs"]

	# The learning rate schedules count a trailing partial batch as a
	# step, as the length of a plain DataLoader does, although training
	# drops it.
	steps_per_epoch = -(-len(training_data) // parameters["batch_size"])

	# Both learning rate schedules are built from `max_epochs` just below, so
	# extending the run here stretches them with it rather than leaving them
	# to decay inside the original budget.
	floored_epochs = _max_epochs_for_min_steps(max_epochs, steps_per_epoch,
		parameters["min_total_steps"])

	if floored_epochs != max_epochs:
		max_epochs = floored_epochs

		if parameters["verbose"]:
			say("Raising max_epochs to {} to reach the {} step minimum "
				"({} steps/epoch, {} total)\n".format(max_epochs,
					parameters["min_total_steps"], steps_per_epoch,
					steps_per_epoch * max_epochs))

	if parameters["verbose"]:
		say("Muon Optimizer: lr={}, wd={}".format(parameters["muon_lr"],
			parameters["muon_wd"]))
		say("AdamW Optimizer: lr={}, wd={}".format(parameters["adam_lr"],
			parameters["adam_wd"]))
		say(_loss_balance_summary(parameters["loss_weights"],
			parameters["lw_lr"], parameters["lw_wd"],
			parameters["lw_momentum"]) + "\n")

	###

	accelerator = {"cuda": "gpu"}.get(parameters["device"],
		parameters["device"])

	trainer = fit(
		model,
		training_data,
		valid_sequences,
		valid_signals,
		X_ctl_valid=valid_controls,
		labels_valid=valid_labels,
		max_epochs=max_epochs,
		early_stopping=parameters["early_stopping"],
		dtype=parameters["dtype"],
		accelerator=accelerator,
		devices=parameters["devices"],
		verbose=parameters["verbose"],
		progress_bar=parameters["progress_bar"],
		batch_size=parameters["batch_size"],
		num_workers=parameters["num_workers"],
		n_warmup_steps=steps_per_epoch * n_warmup_epochs,
		n_decay_steps=steps_per_epoch * max(1, max_epochs - n_warmup_epochs),
		muon_lr=parameters["muon_lr"],
		muon_wd=parameters["muon_wd"],
		adam_lr=parameters["adam_lr"],
		adam_wd=parameters["adam_wd"],
		lw_lr=parameters["lw_lr"],
		lw_wd=parameters["lw_wd"],
		lw_momentum=parameters["lw_momentum"],
		loss_weights=parameters["loss_weights"],
	)

	if not trainer.is_global_zero:
		return

	### Evaluate Model

	model_name = parameters["name"] or model.name

	# The validation chromosomes chose the checkpoint, so only the test
	# chromosomes give an estimate that took no part in training.
	for split in ("validation", "test"):
		if not parameters[split + "_chroms"]:
			continue

		evaluate_parameters = copy.deepcopy(parameters)
		evaluate_parameters["chroms"] = parameters[split + "_chroms"]
		evaluate_parameters["model"] = model_name + ".torch"
		evaluate_parameters["performance_filename"] = "{}.{}.performance.tsv".format(
			model_name, split)

		fname = "{}.{}.evaluate.json".format(model_name, split)
		with open(fname, "w") as outfile:
			outfile.write(json.dumps(evaluate_parameters, sort_keys=True, indent=4))

		evaluate_cmd.run(argparse.Namespace(parameters=fname))
