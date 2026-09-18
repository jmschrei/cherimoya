# cherimoya_cli defaults
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
#
# A note on `signals` / `controls`:
#
# These accept either a flat list of file paths or a structured list
# whose entries are each ``str`` (a one-channel unstranded group) or
# ``list[str]`` (a multi-channel group such as a stranded ``(+, -)``
# pair). The grouping decides how reverse-complement augmentation
# permutes channels and how many predictions the count head emits.
# Examples::
#
#     "signals": ["atac.bw"]
#         # one unstranded group, one count prediction
#
#     "signals": [["ctcf.+.bw", "ctcf.-.bw"]]
#         # one stranded group, one count prediction shared across +/-
#
#     "signals": ["atac.bw", ["ctcf.+.bw", "ctcf.-.bw"]]
#         # one unstranded group + one stranded group; two count predictions
#
# A flat list of N files is now interpreted as N independent unstranded
# groups. BPNet-style callers that previously passed ``["plus.bw",
# "minus.bw"]`` as a stranded pair must update to the nested form
# ``[["plus.bw", "minus.bw"]]`` to keep the +/- swap on RC.

# A note on `random_state`:
#
# Training is seeded by default so that a run can be repeated: the seed
# fixes the model's initialization and the peak/negative sampler's draw
# order. Set it to any other integer to get an independent run -- rerunning
# the same JSON unchanged reproduces the same model rather than giving an
# independent replicate. Setting it to null does not turn seeding off; it
# means "draw a seed, print it, and record it in the evaluate JSON", so an
# unplanned run can still be repeated afterwards.
#
# Seeding does not make CUDA training bitwise reproducible. The fused
# conv+norm kernel reduces with relaxed atomics, so the summation order
# varies between launches. What the seed fixes is the initialization and
# the sequence of examples, not the last bits of the arithmetic.

training_chroms = ["chr2", "chr4", "chr5", "chr7", "chr9", "chr10", "chr11",
    "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
    "chr21", "chr22", "chrX", "chrY"]

validation_chroms = ['chr8', 'chr20']


default_fit_parameters = {
	'n_filters': 128,
	'n_layers': 9,
	'expansion': 2,
	'residual_scale': 0.15,
	'name': None,
	'batch_size': 64,
	'in_window': 2114,
	'out_window': 1000,
	'max_jitter': 500,
	'reverse_complement': True,
	'reverse_complement_average': False,
	'summits': False,
	'max_epochs': 20,
	'muon_lr': 0.025,
	'muon_wd': 0.03,
	'adam_lr': 0.001,
	'adam_wd': 0.0,
	'lw_lr': 0.001,
	'lw_wd': 0.0,
	'lw_momentum': 0.9,
	'n_warmup_epochs': 2,
	'negative_ratio': 0.25,
	'num_workers': 1,
	'dtype': 'float32',
	'device': 'cuda',
	'early_stopping': 5,
	'verbose': False,
	'training_chroms': training_chroms,
	'validation_chroms': validation_chroms,
	'sequences': None,
	'loci': None,
	'exclusion_lists': None,
	'negatives': None,
	'signals': None,
	'controls': None,
	'random_state': 0,
	'performance_filename': 'performance.tsv',
	'skip': False,
}


default_evaluate_parameters = {
	'batch_size': 512,
	'in_window': 2114,
	'out_window': 1000,
	'verbose': False,
	'chroms': validation_chroms,
	'reverse_complement_average': False,
	'device': 'cuda',
	'dtype': 'float32',
	'exclusion_lists': None,
	'sequences': None,
	'loci': None,
	'controls': None,
	'model': None,
	'performance_filename': 'performance.tsv',
	'skip': False,
}


default_attribute_parameters = {
	'batch_size': 512,
	'in_window': 2114,
	'out_window': 1000,
	'verbose': False,
	'chroms': training_chroms + validation_chroms,
	'exclusion_lists': None,
	'sequences': None,
	'loci': None,
	'model': None,
	'output': 'counts',
	'ohe_filename': 'attributions.ohe.npz',
	'attr_filename': 'attributions.attr.npz',
	'idx_filename': 'attributions.idx.npy',
	'dtype': 'float32',
	'device': 'cuda',
	'skip': False,
}


default_seqlet_parameters = {
	'threshold': 0.01,
	'min_seqlet_len': 4,
	'max_seqlet_len': 25,
	'additional_flanks': 3,
	'in_window': 2114,
	'chroms': training_chroms + validation_chroms,
	'exclusion_lists': None,
	'verbose': False,
	'loci': None,
	'ohe_filename': None,
	'attr_filename': None,
	'idx_filename': None,
	'output_filename': 'seqlets.bed',
	'skip': False,
}


default_annotation_parameters = {
	'motifs': None,
	'sequences': None,
	'seqlet_filename': None,
	'n_score_bins': 100,
	'n_median_bins': 1000,
	'n_target_bins': 100,
	'n_cache': 250,
	'reverse_complement': True,
	'n_jobs': -1,
	'output_filename': 'seqlets_annotated.bed',
	'skip': False,
}


default_marginalize_parameters = {
	'batch_size': 512,
	'in_window': 2114,
	'out_window': 1000,
	'verbose': False,
	'chroms': training_chroms,
	'exclusion_lists': None,
	'sequences': None,
	'motifs': None,
	'loci': None,
	'attributions': False,
	'n_loci': 100,
	'shuffle': False,
	'model': None,
	'output_filename':'marginalize/',
	'random_state':0,
	'minimal': True,
	'device': 'cuda',
	'skip': False,
}


default_pipeline_parameters = {
	# Shared parameters
	'in_window': 2114,
	'out_window': 1000,
	'name': None,
	'model': None,
	'dtype': 'float32',
	'device': 'cuda',

	# Data parameters
	'batch_size': 512,
	'verbose': True,
	'random_state': 0,

	'exclusion_lists': None,
	'sequences': None,
	'loci': None,
	'negatives': None,
	'signals': None,
	'controls': None,

	'skip': False,
	'dry_run': False,

	# Data processing parameters
	'preprocessing_parameters': {
		'unstranded': False,
		'fragments': False,
		'paired_end': False,
		'pos_shift': 0,
		'neg_shift': 0,
		'scale_factor': 1,
		'read_depth': False,
		'callpeaks_format': None,
		'callpeaks_gsize': 'hs',
		'callpeaks_q': 0.05,
		'verbose': True
	},

	# Fit parameters
	'fit_parameters': {
		'n_filters': 128,
		'n_layers': 9,
		'expansion': 2,
		'residual_scale': 0.15,
		'batch_size': 64,
		'muon_lr': 0.025,
		'muon_wd': 0.03,
		'adam_lr': 0.001,
		'adam_wd': 0.0,
		'lw_lr': 0.001,
		'lw_wd': 0.0,
		'lw_momentum': 0.9,
		'n_warmup_epochs': 2,
		'negative_ratio': 0.25,
		'count_loss_weight': None,
		'num_workers': 1,
		'early_stopping': 5,
		'max_jitter': 500,
		'reverse_complement': True,
		'reverse_complement_average': False,
		'max_epochs': 20,
		'training_chroms': training_chroms,
		'validation_chroms': validation_chroms,
		'sequences': None,
		'loci': None,
		'negatives': None,
		'signals': None,
		'controls': None,
		'verbose': None,
		'random_state': None,
		'summits': False,
		'performance_filename': None,
	},

	# Attribution parameters
	'attribute_parameters': {
		'batch_size': None,
		'chroms': training_chroms + validation_chroms,
		'output': 'counts',
		'loci': None,
		'dtype': None,
		'device': None,
		'ohe_filename': None,
		'attr_filename': None,
		'idx_filename': None,
		'verbose': None
	},


	# Seqlet Parameters
	'seqlet_parameters': {
		'threshold': 0.01,
		'min_seqlet_len': 4,
		'max_seqlet_len': 25,
		'additional_flanks': 3,
		'in_window': None,
		'chroms': None,
		'verbose': None,
		'loci': None,
		'ohe_filename': None,
		'attr_filename': None,
		'idx_filename': None,
		'output_filename': None
	},


	# Seqlet Annotation Parameters
	'annotation_parameters': {
		'motifs': None,
		'sequences': None,
		'seqlet_filename': None,
		'n_score_bins': 100,
		'n_median_bins': 1000,
		'n_target_bins': 100,
		'n_cache': 250,
		'reverse_complement': True,
		'n_jobs': -1,
		'output_filename': None
	},


	# Modisco parameters
	'modisco_motifs_parameters': {
		'n_seqlets': 100000,
		'output_filename': None,
		'verbose': None
	},


	# Modisco report parameters
	'modisco_report_parameters': {
		'motifs': None,
		'output_folder': None,
		'verbose': None
	},


	# Marginalization parameters
	'marginalize_parameters': {
		'loci': None,
		'n_loci': 100,
		'attributions': False,
		'batch_size': None,
		'shuffle': False,
		'random_state': None,
		'output_folder': None,
		'motifs': None,
		'minimal': True,
		'device': None,
		'verbose': None
	}
}
