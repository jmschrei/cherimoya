# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Code adapted from Alex Tseng, Avanti Shrikumar, and Ziga Avsec

import numpy
import torch

from tangermeme.io import extract_loci


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
	"""

	def __init__(self, peak_sequences, peak_signals, negative_sequences,
		negative_signals, peak_controls=None, negative_controls=None,
		negative_ratio=0.1, in_window=2114, out_window=1000, max_jitter=0,
		reverse_complement=False, shuffle=True, random_state=None):
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

		if self._rc_flags[idx]:
			Xi = torch.flip(Xi, [0, 1])
			yi = torch.flip(yi, [0, 1])
			if Xi_ctl is not None:
				Xi_ctl = torch.flip(Xi_ctl, [0, 1])

		if Xi_ctl is not None:
			return Xi, Xi_ctl, yi, int(is_peak)
		return Xi, yi, int(is_peak)



def PeakGenerator(peaks, negatives, sequences, signals, controls=None,
	chroms=None, in_window=2114, out_window=1000, max_jitter=50,
	negative_ratio=0.1, reverse_complement=True, shuffle=True, min_counts=None,
	max_counts=None, summits=False, exclusion_lists=None, random_state=None,
	pin_memory=True, num_workers=1, batch_size=32, verbose=False):
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

	signals: list of strs or list of dictionaries
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps.

	controls: list of strs or list of dictionaries or None, optional
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps. If None, no control tensor is returned. Default is None.

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
		midpoints that are passed in. Default is 50.

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
		The number of data elements per batch. Default is 32.

	verbose: bool, optional
		Whether to display a progress bar while loading. Default is False.


	Returns
	-------
	X: torch.utils.data.DataLoader
		A PyTorch DataLoader wrapped DataGenerator object.
	"""

	X_peaks = extract_loci(loci=peaks, sequences=sequences,
		signals=signals, in_signals=controls, chroms=chroms, in_window=in_window,
		out_window=out_window, max_jitter=max_jitter, min_counts=min_counts,
		max_counts=max_counts, summits=summits, exclusion_lists=exclusion_lists,
		ignore=list('QWERYUIOPSDFHJKLZXVBNM'), return_mask=True, verbose=verbose)

	loci_counts = X_peaks[1].sum(dim=(1, 2))

	outlier_threshold = torch.quantile(X_peaks[1].sum(dim=(1, 2)), 0.99) * 1.2
	outlier_idxs = loci_counts > outlier_threshold

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
		random_state=random_state
	)

	X_gen = torch.utils.data.DataLoader(X_gen, pin_memory=pin_memory,
		num_workers=num_workers, batch_size=batch_size,
		persistent_workers=num_workers > 0)

	return X_gen
