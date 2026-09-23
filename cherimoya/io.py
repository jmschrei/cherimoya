# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Code adapted from Alex Tseng, Avanti Shrikumar, and Ziga Avsec

import numpy
import torch

from tangermeme.io import extract_loci


def _validate_signal_groups(groups, label='signal_groups'):
	"""Validate a signal-group spec used by the model, loss, and data path.

	Every entry must be a positive integer (size 0 is meaningless — a
	group with no channels can't be permuted, summed, or matched
	against a count head) and the list itself must be non-empty (an
	empty list means "zero output channels", which would produce a
	degenerate model). Raises a ``ValueError`` with a clear message
	naming the bad value so the user can fix the JSON or kwarg.


	Parameters
	----------
	groups : sequence
		The candidate ``signal_groups`` / ``control_groups`` value to
		validate.

	label : str, optional
		The kwarg or field name to mention in the error message — lets
		the same helper produce useful errors from different call sites
		(``signal_groups``, ``control_groups``, etc.). Default is
		``'signal_groups'``.
	"""

	if not isinstance(groups, (list, tuple)):
		raise ValueError(
			"{} must be a list of positive ints; got {!r}"
			.format(label, groups))
	if len(groups) == 0:
		raise ValueError(
			"{} must be non-empty; got an empty list".format(label))
	for g in groups:
		if (not isinstance(g, int)) or isinstance(g, bool) or g < 1:
			raise ValueError(
				"{} entries must be positive ints; got {!r} in {!r}"
				.format(label, g, list(groups)))


def normalize_signal_groups(signals):
	"""Normalize a structured ``signals``/``controls`` spec into flat files.

	Cherimoya organizes signals into *groups* so reverse-complement
	augmentation can swap channels correctly. Each group is one biological
	modality whose channels share an orientation: an unstranded track is a
	single-channel group; a stranded ``(+, -)`` pair is a two-channel group.

	Accepted inputs::

		None
			Returns ``(None, [])``.

		[str, str, ...]
			A flat list of N files. Each file is treated as its **own**
			single-channel (unstranded) group. (Note: this is a breaking
			change relative to pre-grouping cherimoya, where a flat
			two-element list was implicitly a stranded pair. Old BPNet
			callers should now wrap the pair: ``[["plus.bw", "minus.bw"]]``.)

		[entry, entry, ...]
			A list whose entries are each either a single ``str`` (a
			one-channel group, shorthand for ``[str]``) or a ``list[str]``
			(a multi-channel group). Mix freely, e.g.::

				signals = ["atac.bw", ["ctcf.+.bw", "ctcf.-.bw"]]

			declares two groups: one unstranded ATAC channel and one
			stranded CTCF pair.

	Parameters
	----------
	signals : None, list of str, or list of (str or list of str)
		The structured signals spec as described above.


	Returns
	-------
	flat_files : list of str or None
		The files in concatenation order across groups. ``None`` if
		``signals`` was ``None``. This is the form that ``extract_loci``
		consumes.

	group_sizes : list of int
		The number of channels in each group, in the same order. Empty
		list if ``signals`` was ``None``.
	"""

	if signals is None:
		return None, []

	if not isinstance(signals, (list, tuple)):
		raise TypeError("signals must be a list/tuple or None; got {!r}"
			.format(type(signals).__name__))

	flat = []
	groups = []
	for entry in signals:
		if isinstance(entry, str):
			flat.append(entry)
			groups.append(1)
		elif isinstance(entry, (list, tuple)):
			if len(entry) == 0:
				raise ValueError("signal group is empty")
			for f in entry:
				if not isinstance(f, str):
					raise TypeError(
						"signal group entries must be strings; got {!r}"
						.format(type(f).__name__))
				flat.append(f)
			groups.append(len(entry))
		else:
			raise TypeError(
				"signal entries must be str or list[str]; got {!r}"
				.format(type(entry).__name__))

	return flat, groups


def channel_permutation_from_groups(group_sizes):
	"""Build the channel permutation applied to per-locus signals under RC.

	Within each group the channels are reversed in place — a 1-channel
	group stays put (identity), a 2-channel ``(+, -)`` group becomes
	``(-, +)``. Groups keep their relative order: they're independent
	modalities and must not bleed into each other under augmentation.


	Parameters
	----------
	group_sizes : sequence of int
		The size of each group, e.g. ``[1, 2]`` for an unstranded ATAC
		track followed by a stranded CTCF pair.


	Returns
	-------
	perm : torch.LongTensor, shape=(sum(group_sizes),)
		Index tensor such that ``y[perm].flip(-1)`` is the correct RC of a
		``(n, sum(group_sizes), L)`` signal tensor.
	"""

	_validate_signal_groups(group_sizes, label='group_sizes')

	perm = []
	offset = 0
	for g in group_sizes:
		perm.extend(range(offset + g - 1, offset - 1, -1))
		offset += g
	return torch.tensor(perm, dtype=torch.long)


class PeakNegativeSampler(torch.utils.data.Dataset):
	"""A data generator mimicking the BPNet data loading procedure.

	Here, a set of peaks and negatives are separately loaded. These sets can be
	any size. From these sets, batches of given size are sampled that are a
	mixture of peaks and negatives.

	Sampling is fully deterministic given ``random_state`` and the epoch
	number. ``__getitem__(idx)`` is a pure function of ``idx`` and the
	current epoch, so ``num_workers > 1`` produces the same per-index
	data tuples as ``num_workers = 1`` — the DataLoader yields identical
	batch sequences, just faster.

	Each peak is drawn exactly once per epoch; the peak/negative
	interleaving and all augmentations are reproducible from
	``(random_state, epoch)``.

	In the documentation below, ``mj`` = max_jitter.

	Parameters
	----------
	peak_sequences: torch.tensor, shape=(n_peaks, 4, in_window+2*mj)
		A tensor of peak sequences that are one-hot encoded.

	peak_signals: torch.tensor, shape=(n_peaks, t, out_window+2*mj)
		A tensor of signals to predict, usually base-pair resolution
		integer counts.

	peak_controls: torch.tensor, shape=(n, t, out_window+2*mj) or None,
			optional
		Optional control input track for peak examples.

	negative_sequences: torch.tensor, shape=(n, 4, in_window+2*mj)
		One-hot encoded negative sequences.

	negative_signals: torch.tensor, shape=(n, t, out_window+2*mj)
		Negative sequence signals.

	negative_controls: torch.tensor or None, optional
		Optional control input track for negative examples.

	negative_ratio: float, optional
		Ratio of negatives to peaks per epoch. ``0`` means no negative
		draws. Default 0.1.

	in_window: int, optional
		The input window size. Default 2114.

	out_window: int, optional
		The output window size. Default 1000.

	max_jitter: int, optional
		Maximum jitter (in either direction) applied to peaks. Default 0.

	reverse_complement: bool, optional
		Whether to reverse complement-augment half of the data. Default
		False.

	shuffle: bool, optional
		Whether to shuffle the peak ordering each epoch. Default True.

	random_state: int or None, optional
		Base seed for the deterministic per-epoch RNG. If None, a random
		seed is captured once at construction time so that all forked
		worker processes share it.

	signal_perm: torch.LongTensor or None, optional
		The channel permutation to apply to ``peak_signals`` /
		``negative_signals`` under reverse-complement augmentation. If
		``None``, no channel permutation is applied (only the length
		dimension is flipped). This must be precomputed by the caller
		from the signal-group structure — see
		:func:`channel_permutation_from_groups`. Default is None.

	control_perm: torch.LongTensor or None, optional
		Same as ``signal_perm`` but applied to the control tracks. Default
		is None.
	"""

	def __init__(self, peak_sequences, peak_signals, negative_sequences,
		negative_signals, peak_controls=None, negative_controls=None,
		negative_ratio=0.1, in_window=2114, out_window=1000, max_jitter=0,
		reverse_complement=False, shuffle=True, random_state=None,
		signal_perm=None, control_perm=None):
		if max_jitter < 0:
			raise ValueError("max_jitter must be non-negative, got {}"
				.format(max_jitter))
		if negative_ratio < 0:
			raise ValueError("negative_ratio must be non-negative, got {}"
				.format(negative_ratio))

		self.peak_sequences = peak_sequences.numpy(force=True)
		self.peak_signals = peak_signals.numpy(force=True)
		self.n_peaks = len(self.peak_sequences)

		self.negative_sequences = negative_sequences.numpy(force=True)
		self.negative_signals = negative_signals.numpy(force=True)
		self.n_negatives = len(self.negative_sequences)

		if negative_ratio > 0 and self.n_negatives == 0:
			raise ValueError(
				"negative_ratio is {} but no negative sequences were "
				"given; pass negatives or set negative_ratio to 0"
				.format(negative_ratio))

		if peak_controls is not None:
			self.peak_controls = peak_controls.numpy(force=True)
			self.negative_controls = negative_controls.numpy(force=True)
		else:
			self.peak_controls = None
			self.negative_controls = None

		self.negative_ratio = negative_ratio
		self.in_window = in_window
		self.out_window = out_window
		self.max_jitter = max_jitter
		self.reverse_complement = reverse_complement
		self.shuffle = shuffle

		# Stored as torch.long tensors so they can index the per-sample
		# torch tensors built in `__getitem__`. The perm tensors are
		# tiny (one element per signal channel) so the construction-
		# time copy is negligible.
		self.signal_perm = (None if signal_perm is None
			else torch.as_tensor(signal_perm, dtype=torch.long))
		self.control_perm = (None if control_perm is None
			else torch.as_tensor(control_perm, dtype=torch.long))

		if self.signal_perm is not None:
			if self.signal_perm.shape[0] != self.peak_signals.shape[1]:
				raise ValueError(
					"signal_perm length ({}) does not match peak_signals "
					"channel count ({})".format(
						self.signal_perm.shape[0],
						self.peak_signals.shape[1]))
		if self.control_perm is not None:
			if self.peak_controls is None:
				raise ValueError(
					"control_perm was supplied but peak_controls is None")
			if self.control_perm.shape[0] != self.peak_controls.shape[1]:
				raise ValueError(
					"control_perm length ({}) does not match peak_controls "
					"channel count ({})".format(
						self.control_perm.shape[0],
						self.peak_controls.shape[1]))

		# Capture one base seed at construction so every forked worker
		# inherits the same value (2654435761 is Knuth's hash constant,
		# spreading small epoch values across the 32-bit seed space).
		if random_state is None:
			random_state = int(numpy.random.randint(0, 2**31 - 1))
		self._base_seed = int(random_state) % (2**31 - 1)

		# _last_idx detects epoch boundaries by wrap-around (idx jumping
		# backward). Each forked worker maintains its own copy.
		self._last_idx = -1
		self._epoch = -1
		self._prepare_epoch(0)

	def __len__(self):
		return self.n_peaks + int(self.n_peaks * self.negative_ratio)

	def _prepare_epoch(self, epoch):
		"""Recompute per-epoch arrays from the (base_seed, epoch) RNG."""

		self._epoch = epoch
		seed = (self._base_seed + epoch * 2654435761) % (2**31 - 1)
		rng = numpy.random.RandomState(seed)
		n = len(self)

		# Peak ordering — each peak appears exactly once. Kept as an
		# attribute for introspection.
		self.peak_ordering = (rng.permutation(self.n_peaks) if self.shuffle
			else numpy.arange(self.n_peaks))

		# Per-position label: True at exactly n_peaks slots, False at the
		# remaining negative slots.
		labels = rng.permutation(numpy.arange(n) < self.n_peaks)
		self._labels = labels

		# Per-position source index into the peak or negative tensor.
		# max(1, n_negatives) keeps randint's bounds valid even when
		# n_negatives == 0; the size is also 0 in that case so no values
		# are actually written.
		source = numpy.empty(n, dtype=numpy.int64)
		source[labels] = self.peak_ordering
		source[~labels] = rng.randint(0, max(1, self.n_negatives),
			size=int((~labels).sum()))
		self._source_idx = source

		# Per-position jitter (0 at negative positions) and rc flag.
		if self.max_jitter > 0:
			jitters = rng.randint(0, self.max_jitter * 2, size=n)
			jitters[~labels] = 0
			self._jitters = jitters
		else:
			self._jitters = numpy.zeros(n, dtype=numpy.int64)

		if self.reverse_complement:
			self._rc_flags = rng.randint(0, 2, size=n).astype(bool)
		else:
			self._rc_flags = numpy.zeros(n, dtype=bool)

	def __getitem__(self, idx):
		if idx < self._last_idx:
			self._prepare_epoch(self._epoch + 1)
		self._last_idx = idx

		is_peak = bool(self._labels[idx])
		src = int(self._source_idx[idx])
		j = int(self._jitters[idx])

		if is_peak:
			X, y, X_ctl = (self.peak_sequences, self.peak_signals,
				self.peak_controls)
		else:
			X, y, X_ctl = (self.negative_sequences, self.negative_signals,
				self.negative_controls)

		Xi = torch.from_numpy(X[src][:, j:j+self.in_window])
		yi = torch.from_numpy(y[src][:, j:j+self.out_window])
		Xi_ctl = (torch.from_numpy(X_ctl[src][:, j:j+self.in_window])
			if self.peak_controls is not None else None)

		# Reverse-complement: channel permutation + length flip. For the
		# DNA sequence the channel perm is the simple ACGT -> TGCA flip
		# (which `torch.flip` on dim 0 of a length-4 axis does
		# correctly). For signals/controls the channel permutation
		# depends on the group structure — see
		# `channel_permutation_from_groups`. The permutation is
		# precomputed once at construction so the per-call cost is just
		# an indexed slice plus a length flip — same big-O as the legacy
		# single-track behavior, even when many groups are stacked.
		if self._rc_flags[idx]:
			Xi = torch.flip(Xi, [0, 1])
			if self.signal_perm is not None:
				yi = yi[self.signal_perm]
			yi = torch.flip(yi, [-1])
			if Xi_ctl is not None:
				if self.control_perm is not None:
					Xi_ctl = Xi_ctl[self.control_perm]
				Xi_ctl = torch.flip(Xi_ctl, [-1])

		if Xi_ctl is not None:
			return Xi, Xi_ctl, yi, int(is_peak)
		return Xi, yi, int(is_peak)



def PeakGenerator(peaks, negatives, sequences, signals, controls=None,
	chroms=None, in_window=2114, out_window=1000, max_jitter=500,
	negative_ratio=0.25, reverse_complement=True, shuffle=True, min_counts=None,
	max_counts=None, summits=False, exclusion_lists=None, random_state=None,
	pin_memory=True, num_workers=1, batch_size=64, verbose=False,
	signal_groups=None, control_groups=None):
	"""This is a constructor function that handles all IO.

	This function will extract signal from all signal and control files,
	pass that into a DataGenerator, and wrap that using a PyTorch data
	loader. This is the only function that needs to be used.


	Parameters
	----------
	peaks: str or pandas.DataFrame or list/tuple of such
		A BED-formatted file containing peak coordinates. This can be either
		the string path to the BED file or a pandas DataFrame object containing
		three columns: chrom, start, and end. Alternatively, this can be a list
		of such objects whose coordinates will be interleaved.

	negatives: str or pandas.DataFrame or list/tuple of such
		A BED-formatted file containing negative coordinates. This can be either
		the string path to the BED file or a pandas DataFrame object containing
		three columns: chrom, start, and end. Alternatively, this can be a list
		of such objects whose coordinates will be interleaved.

	sequences: str or dictionary
		Either the path to a fasta file to read from or a dictionary where the
		keys are the unique set of chromosoms and the values are one-hot
		encoded sequences as numpy arrays or memory maps.

	signals: list of strs, list of (str or list of str), or list of dictionaries
		The signal-track specification. There are two accepted shapes:

		1. A flat list of strings — each element is a one-channel
		   (unstranded) group. This is the common ATAC/DNase case.

		2. A list whose entries are each either a ``str`` (a one-channel
		   group, shorthand for ``[str]``) or a ``list[str]`` (a multi-
		   channel group, e.g. a stranded ``(+, -)`` pair). Mix freely::

		       signals = ["atac.bw", ["ctcf.+.bw", "ctcf.-.bw"]]

		Each group's channels share an orientation and get swapped under
		reverse-complement augmentation; groups never bleed into one
		another. See :func:`normalize_signal_groups`.

		A list of per-chromosome dictionaries is also accepted (one
		dictionary per channel, in the same concatenation order produced
		by :func:`normalize_signal_groups`). In that case the caller
		must pass ``signal_groups`` explicitly so this function knows
		how to group the channels.

	controls: same shape as ``signals``, or None, optional
		The control-track specification, sharing the same grouping rule
		as ``signals``. If None, no control tensor is returned. Default
		is None.

	chroms: list or None, optional
		A set of chromosomes to extact loci from. Loci in other chromosomes
		in the locus file are ignored. If None, all loci are used. Default is
		None.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 500.

	negative_ratio: float, optional
		The ratio of negatives compared to peaks in each batch. A value of 1 means
		that each batch is balanced, and a value of 10 means that there would be 10
		negatives for each positive. Note that this is independent of the number of
		peaks and negatives provided. Even if the `peaks` input has 10x the number
		of coordinates as the `negatives` one, if the ratio is 1 each batch during
		training will be balanced (on average).

	reverse_complement: bool, optional
		Whether to reverse complement-augment half of the data. Default is True.

	shuffle: bool, optional
		Whether to randomly sample peaks, if True, or to proceed sequentially
		through them, if False. Negatives are always randomly sampled. Default
		is True.

	min_counts: float or None, optional
		The minimum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no minimum. Default
		is None.

	max_counts: float or None, optional
		The maximum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no maximum. Default
		is None.

	summits: bool, optional
		Whether to return a region centered around the summit instead of the center
		between the start and end. If True, it will add the 10th column (index 9)
		to the start to get the center of the window, and so the data must be in
		narrowPeak format.

	exclusion_lists: list or None, optional
		A list of strings of filenames to BED-formatted files containing exclusion
		lists, i.e., regions where overlapping loci should be filtered out. If None,
		no filtering is performed based on exclusion zones. Default is None.

	random_state: int or None, optional
		Base seed for the sampler's deterministic per-epoch RNG. If None,
		a seed is captured once from system entropy.

	pin_memory: bool, optional
		Whether to pin page memory to make data loading onto a GPU easier.
		Default is True.

	num_workers: int, optional
		The number of processes fetching data at a time to feed into a
		model. If 0, data is fetched from the main process (synchronous,
		can become a bottleneck because each batch blocks the GPU).
		Default is 1, which runs one async prefetch worker. Higher values
		are safe and produce the **same** sequence of batches as
		``num_workers = 1``, just faster: ``__getitem__(idx)`` is a pure
		function of ``idx`` and the current epoch, so all workers compute
		the same data for any given index.

	batch_size: int, optional
		The number of data elements per batch. Default is 64.

	verbose: bool, optional
		Whether to display a progress bar while loading. Default is False.

	signal_groups: list of int or None, optional
		The size of each group in ``signals``. When ``signals`` is given
		as a list of strings or as a structured (mixed) list this is
		derived automatically. Must be supplied explicitly only when
		``signals`` is a flat list of per-chromosome dictionaries (no
		group information can be inferred from that form). Default is
		None.

	control_groups: list of int or None, optional
		Same as ``signal_groups`` but for ``controls``. Default is None.


	Returns
	-------
	X: torch.utils.data.DataLoader
		A PyTorch DataLoader wrapped DataGenerator object.
	"""

	# Normalize structured (list-of-lists / mixed) signal specs into a
	# flat file list plus per-group sizes. When signals are already
	# provided as a list of dictionaries we leave them alone and require
	# the caller to pass signal_groups explicitly — there's no way to
	# infer grouping from raw dicts.
	def _resolve(spec, explicit_groups, label):
		if spec is None:
			return None, []
		# A list of per-chromosome dicts (the in-memory shortcut some
		# callers use to avoid bigWig IO) can't be parsed by
		# `normalize_signal_groups`, so handle it explicitly. The
		# isinstance check is tight on purpose — anything that isn't a
		# dict, str, or sub-list is a user mistake (e.g. an int passed
		# by accident) and is better surfaced here than 200 lines later
		# inside `extract_loci`.
		if isinstance(spec, (list, tuple)) and len(spec) > 0 \
				and isinstance(spec[0], dict):
			if explicit_groups is None:
				explicit_groups = [1] * len(spec)
			else:
				_validate_signal_groups(explicit_groups,
					label='{}_groups'.format(label))
			if sum(explicit_groups) != len(spec):
				raise ValueError(
					"{}_groups sum ({}) does not match number of {} "
					"channels ({})".format(label, sum(explicit_groups),
						label, len(spec)))
			return list(spec), list(explicit_groups)
		flat, groups = normalize_signal_groups(spec)
		if explicit_groups is not None:
			_validate_signal_groups(explicit_groups,
				label='{}_groups'.format(label))
			if list(explicit_groups) != groups:
				raise ValueError(
					"{}_groups={} disagrees with the grouping inferred "
					"from {}={}".format(label, list(explicit_groups),
						label, groups))
		return flat, groups

	signals, signal_groups = _resolve(signals, signal_groups, 'signal')
	controls, control_groups = _resolve(controls, control_groups, 'control')

	signal_perm = (channel_permutation_from_groups(signal_groups)
		if signal_groups else None)
	control_perm = (channel_permutation_from_groups(control_groups)
		if control_groups else None)

	X_peaks = extract_loci(loci=peaks, sequences=sequences,
		signals=signals, in_signals=controls, chroms=chroms, in_window=in_window,
		out_window=out_window, max_jitter=max_jitter, min_counts=min_counts,
		max_counts=max_counts, summits=summits, exclusion_lists=exclusion_lists,
		ignore=list('QWERYUIOPSDFHJKLZXVBNM'), return_mask=True, verbose=verbose)

	# Per-modality outlier filtering. The legacy code summed across
	# every channel and the full length to get a single per-locus
	# total, then dropped loci above 1.2x the 99th-percentile total.
	# That collapses biologically distinct modalities into one number
	# — a stranded TF with peaks two orders of magnitude higher than a
	# co-trained ATAC track would dominate the threshold and skew the
	# filter for both. Compute one threshold per signal group and OR
	# the per-group outlier masks: a locus that's an outlier in *any*
	# group is dropped. For the common single-group case this reduces
	# exactly to the legacy behavior.
	peak_signals = X_peaks[1]
	outlier_idxs = torch.zeros(peak_signals.shape[0], dtype=torch.bool)
	offset = 0
	for g in (signal_groups if signal_groups else [peak_signals.shape[1]]):
		group_counts = peak_signals[:, offset:offset+g].sum(dim=(1, 2))
		group_threshold = torch.quantile(group_counts, 0.99) * 1.2
		outlier_idxs |= group_counts > group_threshold
		offset += g

	X_bg = extract_loci(loci=negatives, sequences=sequences,
		signals=signals, in_signals=controls, chroms=chroms, in_window=in_window,
		out_window=out_window, max_jitter=0, min_counts=min_counts,
		max_counts=max_counts, summits=False, exclusion_lists=exclusion_lists,
		ignore=list('QWERYUIOPSDFHJKLZXVBNM'), return_mask=True, verbose=verbose)

	if verbose:
		n_filtered_peaks = len(X_peaks[-1]) - X_peaks[-1].sum() + outlier_idxs.sum()
		n_filtered_negatives = len(X_bg[-1]) - X_bg[-1].sum()

		print("\nFiltered Peaks: {}".format(n_filtered_peaks))
		print("Filtered Negatives: {}".format(n_filtered_negatives))

	###

	X_gen = PeakNegativeSampler(
		peak_sequences=X_peaks[0][~outlier_idxs],
		peak_signals=X_peaks[1][~outlier_idxs],
		peak_controls=None if controls is None else X_peaks[2][~outlier_idxs],
		negative_sequences=X_bg[0],
		negative_signals=X_bg[1],
		negative_controls=None if controls is None else X_bg[2],
		negative_ratio=negative_ratio,
		in_window=in_window,
		out_window=out_window,
		max_jitter=max_jitter,
		reverse_complement=reverse_complement,
		shuffle=shuffle,
		random_state=random_state,
		signal_perm=signal_perm,
		control_perm=control_perm,
	)

	X_gen = torch.utils.data.DataLoader(X_gen, pin_memory=pin_memory,
		num_workers=num_workers, batch_size=batch_size,
		persistent_workers=num_workers > 0)

	return X_gen
