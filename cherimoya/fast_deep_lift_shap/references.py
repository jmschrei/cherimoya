# references.py
# Author: Eugenio Mattei

"""
The DeepLIFT/SHAP references of the fast engine, drawn exactly as
tangermeme's `deep_lift_shap` draws them, in worker processes.

`tangermeme.deep_lift_shap.deep_lift_shap` gives the pair of sequence `e`
and reference `k` the reference ``dinucleotide_shuffle(X[e:e+1].float(),
n=1, random_state=random_state + k)[:, 0]``. It makes one seeded call per
pair, serially, between batches on the device. The fast engine needs exactly
those references, so this module calls that same public function on that
same input, in spawned worker processes that run ahead of the device.

* Processes rather than threads: tangermeme's numba shuffle kernel holds the
  GIL.
* Spawn rather than fork: the parent process may have initialized CUDA.
  Like any spawn-based pool, a script that runs it needs the usual
  ``if __name__ == "__main__":`` guard.

References travel between processes as uint8 tokens, one per position: 0-3
for the one-hot channel (A, C, G, T) and `N_TOKEN` for an all-zero column.
A token array is a quarter of the size of the int8 one-hot and a sixteenth
of the float32 one, and the encoding is lossless. Anything that is not a
one-hot with at most one 1 per position is refused, never rounded.
"""

import collections
import logging
import multiprocessing
import operator
import os
import time

import numpy
import torch

from concurrent.futures import ProcessPoolExecutor


logger = logging.getLogger(__name__)

N_TOKEN = 4
"""The token of an all-zero one-hot column, an N. Tokens 0-3 are the one-hot
channels."""

# Rows tokenized at a time, which bounds the temporaries for a large X.
_BLOCK_ROWS = 4096


def onehot_to_tokens(X):
	"""Encode a one-hot tensor of shape (..., 4, L) as uint8 tokens (..., L).

	Each position becomes the index of its hot channel, or `N_TOKEN` for an
	all-zero column. `tokens_to_onehot` inverts it exactly. The encoding
	sums and compares the four channels rather than taking an argmax, which
	on the CPU costs about half as much as a shuffle.


	Parameters
	----------
	X: torch.Tensor or numpy.ndarray, shape=(..., 4, L)
		A one-hot encoding with at most one 1 per position.


	Returns
	-------
	tokens: numpy.ndarray, dtype=uint8, shape=(..., L)
		The token of each position.


	Raises
	------
	ValueError
		If X does not have 4 channels on its second-to-last axis, holds a
		value other than 0 and 1, or has more than one 1 at a position.
	"""

	x = torch.as_tensor(X)
	if x.dim() < 2 or x.shape[-2] != 4:
		raise ValueError("expected a one-hot of shape (..., 4, L), got "
			"{}".format(tuple(x.shape)))

	if x.dtype == torch.bool:
		x = x.to(torch.uint8)

	lead, length = tuple(x.shape[:-2]), x.shape[-1]
	flat = x.reshape(-1, 4, length)
	out = numpy.empty((flat.shape[0], length), numpy.uint8)
	for a in range(0, flat.shape[0], _BLOCK_ROWS):
		block = flat[a:a + _BLOCK_ROWS]
		if not bool(((block == 0) | (block == 1)).all()):
			raise ValueError("not a one-hot encoding: it holds values other "
				"than 0 and 1")

		hot = block[:, 0] + block[:, 1] + block[:, 2] + block[:, 3]
		if bool((hot > 1).any()):
			raise ValueError("not a one-hot encoding: a position has more "
				"than one base")

		code = block[:, 1] + 2 * block[:, 2] + 3 * block[:, 3]
		out[a:a + block.shape[0]] = torch.where(hot == 0, N_TOKEN,
			code).to(torch.uint8).cpu().numpy()

	return out.reshape(*lead, length)


def tokens_to_onehot(tokens, dtype=torch.float32):
	"""Expand uint8 tokens (..., L) to a one-hot (..., 4, L) of `dtype`.

	`N_TOKEN` becomes an all-zero column. This is the inverse of
	`onehot_to_tokens`. The token values are checked on CPU tensors only:
	a check on a GPU tensor would synchronize the device, and tokens reach a
	GPU only from host arrays that were checked already.


	Parameters
	----------
	tokens: torch.Tensor or numpy.ndarray, dtype=uint8, shape=(..., L)
		The tokens.

	dtype: torch.dtype, optional
		The dtype of the one-hot. Default is torch.float32.


	Returns
	-------
	X: torch.Tensor, shape=(..., 4, L)
		The one-hot encoding, on the device of `tokens`.
	"""

	t = torch.as_tensor(tokens)
	if t.dtype != torch.uint8:
		raise TypeError("tokens must be uint8, got {}".format(t.dtype))

	if t.dim() < 1:
		raise ValueError("tokens must have a length axis")

	if t.device.type == "cpu" and t.numel() and int(t.max()) > N_TOKEN:
		raise ValueError("tokens must lie in 0-{}, got {}".format(N_TOKEN,
			int(t.max())))

	codes = torch.arange(4, dtype=torch.uint8, device=t.device)
	return (t.unsqueeze(-2) == codes[:, None]).to(dtype)


def shuffle_tokens(tokens, n_shuffles, random_state):
	"""Draw each sequence's references with tangermeme, exactly as
	`deep_lift_shap` does, and return them as tokens.

	Row ``[e, k]`` of the result re-expands to ``dinucleotide_shuffle(x_e,
	n=1, random_state=random_state + k)[:, 0]``, where `x_e` is sequence `e`
	as the float32 (1, 4, L) one-hot that `deep_lift_shap` passes. This is
	the body of `RefProducer`'s workers, and it also runs in-process.
	Whatever tangermeme raises propagates unchanged, e.g. its ValueError for
	an N.


	Parameters
	----------
	tokens: numpy.ndarray, dtype=uint8, shape=(n, L)
		The sequences, as tokens.

	n_shuffles: int
		The number of references per sequence, K.

	random_state: int
		The base seed. Reference `k` uses the seed ``random_state + k``.


	Returns
	-------
	references: numpy.ndarray, dtype=uint8, shape=(n, n_shuffles, L)
		The references, as tokens.
	"""

	from tangermeme.ersatz import dinucleotide_shuffle

	tokens = _check_tokens(tokens)
	n_shuffles = _positive(n_shuffles, "n_shuffles")
	random_state = _seed(random_state)

	out = numpy.empty((tokens.shape[0], n_shuffles, tokens.shape[1]),
		numpy.uint8)
	for e in range(tokens.shape[0]):
		x = tokens_to_onehot(tokens[e:e + 1])
		references = [dinucleotide_shuffle(x, n=1,
			random_state=random_state + k)[:, 0] for k in range(n_shuffles)]
		out[e] = onehot_to_tokens(torch.cat(references))

	return out


def default_workers():
	"""The default number of reference workers: min(8, usable CPUs - 2), and
	at least 1. The remaining CPUs are left to the main process."""

	try:
		n_cpus = len(os.sched_getaffinity(0))
	except AttributeError:
		n_cpus = os.cpu_count() or 1

	return max(1, min(8, n_cpus - 2))


class RefProducer:
	"""References for consecutive chunks of sequences, from a spawn process
	pool, in order and ahead of their use.

	Use it as a context manager, or call `start` and `close`. One producer
	serves any number of `chunks` calls. `stats` describes the latest one:
	the number of chunks and pairs, `stalls` (chunks that were not ready when
	asked for), `wait_s` (the time the consumer spent blocked on them),
	`first_wait_s` (the part of it spent on the first chunk, 0.0 when that
	chunk was ready) and `max_in_flight`.

	Before the pool starts, the parent process compiles, or loads from
	numba's cache, tangermeme's numba shuffle kernel, so that the workers
	load the cached kernel rather than each compiling it.


	Parameters
	----------
	workers: int or None, optional
		The number of worker processes. None means `default_workers()`, and 0
		draws the references in this process instead, for tests and small
		jobs. Default is None.

	ahead: int, optional
		The minimum number of chunks kept in flight. The pool keeps
		max(ahead, 2 * workers) chunks in flight, so that it keeps working
		while the consumer handles the earlier ones. Default is 3.
	"""

	def __init__(self, workers=None, ahead=3):
		if workers is None:
			self.workers = default_workers()
		else:
			self.workers = operator.index(workers)

		if self.workers < 0:
			raise ValueError("workers must be 0 or more, got {}".format(
				self.workers))

		self.ahead = _positive(ahead, "ahead")
		self.window = max(self.ahead, 2 * self.workers) if self.workers else 0
		self.numba_cache_path = None
		self.stats = {}
		self._pool = None
		self._warm = False

	def __enter__(self):
		return self.start()

	def __exit__(self, *exc_info):
		self.close()

	def start(self):
		"""Pre-warm tangermeme's numba kernel in this process, then start the
		pool. Calling it again does nothing."""

		if not self._warm:
			self.numba_cache_path = _prewarm_numba()
			self._warm = True

		if self.workers and self._pool is None:
			self._pool = ProcessPoolExecutor(self.workers,
				mp_context=multiprocessing.get_context("spawn"),
				initializer=_init_worker)
			logger.info("reference pool: %d spawned workers, numba cache %s",
				self.workers, self.numba_cache_path)

		return self

	def close(self):
		"""Stop the pool: cancel the chunks still queued and wait for the
		running ones."""

		if self._pool is not None:
			self._pool.shutdown(wait=True, cancel_futures=True)
			self._pool = None

	def chunks(self, tokens, n_shuffles, random_state, seqs_per_chunk):
		"""Yield ``(a, b, references)`` for the sequences [a, b), in order.

		The arguments are checked, and the first chunks are submitted to the
		pool, before this returns, so the pool starts working while the
		caller does other things, such as compiling. A chunk whose task
		failed raises its exception, with its original type, when the
		iteration reaches that chunk; the chunks still queued are then
		cancelled.


		Parameters
		----------
		tokens: numpy.ndarray, dtype=uint8, shape=(n, L)
			The sequences, as `onehot_to_tokens` returns them.

		n_shuffles: int
			The number of references per sequence, K.

		random_state: int
			The base seed. For `random_state=None` in tangermeme's sense,
			draw a base seed once and pass it.

		seqs_per_chunk: int
			The number of sequences per chunk. The last chunk may be shorter.


		Returns
		-------
		chunks: iterator of (int, int, numpy.ndarray)
			`a`, `b` and the references of the sequences [a, b), a uint8
			array of shape (b - a, n_shuffles, L), as `shuffle_tokens`
			returns them.
		"""

		tokens = _check_tokens(tokens)
		n_shuffles = _positive(n_shuffles, "n_shuffles")
		random_state = _seed(random_state)
		seqs_per_chunk = _positive(seqs_per_chunk, "seqs_per_chunk")

		bounds = collections.deque((a, min(a + seqs_per_chunk, len(tokens)))
			for a in range(0, len(tokens), seqs_per_chunk))

		self.start()
		stats = {
			"chunks": len(bounds),
			"pairs": len(tokens) * n_shuffles,
			"stalls": 0,
			"wait_s": 0.0,
			"first_wait_s": None,
			"max_in_flight": 0,
		}

		self.stats = stats
		task = (tokens, n_shuffles, random_state)
		if self._pool is None:
			return self._inline(task, bounds, stats)

		pending = collections.deque()
		self._fill(self._pool, pending, bounds, task, stats)
		return self._drain(self._pool, pending, bounds, task, stats)

	def _fill(self, pool, pending, bounds, task, stats):
		tokens, n_shuffles, random_state = task
		while bounds and len(pending) < self.window:
			a, b = bounds.popleft()
			future = pool.submit(shuffle_tokens, tokens[a:b], n_shuffles,
				random_state)
			pending.append((a, b, future))

		stats["max_in_flight"] = max(stats["max_in_flight"], len(pending))

	def _drain(self, pool, pending, bounds, task, stats):
		try:
			while pending:
				a, b, future = pending.popleft()
				waited = 0.0
				if not future.done():
					stats["stalls"] += 1
					start = time.perf_counter()
					# Waits without raising, so the wait is counted either way.
					future.exception()
					waited = time.perf_counter() - start
					stats["wait_s"] += waited

				if stats["first_wait_s"] is None:
					stats["first_wait_s"] = waited

				references = future.result()
				self._fill(pool, pending, bounds, task, stats)
				yield a, b, references
		finally:
			for _, _, future in pending:
				future.cancel()

	@staticmethod
	def _inline(task, bounds, stats):
		tokens, n_shuffles, random_state = task
		for a, b in bounds:
			start = time.perf_counter()
			references = shuffle_tokens(tokens[a:b], n_shuffles, random_state)
			waited = time.perf_counter() - start
			stats["stalls"] += 1
			stats["wait_s"] += waited
			if stats["first_wait_s"] is None:
				stats["first_wait_s"] = waited

			yield a, b, references


def _init_worker():
	"""The pool's initializer: one intra-op thread per worker, and
	tangermeme's numba kernel loaded from the cache the parent wrote."""

	torch.set_num_threads(1)
	import tangermeme.ersatz  # noqa: F401


def _prewarm_numba():
	"""Compile, or load from numba's cache, tangermeme's shuffle kernel in
	this process, before any worker starts. Returns the kernel's cache
	directory, or None if numba does not say.

	When NUMBA_CACHE_DIR is set but the kernel is cached elsewhere, because
	tangermeme was imported before the variable was set, each worker
	compiles the kernel itself; a warning says so.
	"""

	from tangermeme import ersatz

	x = tokens_to_onehot(numpy.array([[0, 1, 2, 3, 0, 2, 1, 3]], numpy.uint8))
	ersatz.dinucleotide_shuffle(x, n=1, random_state=0)

	# numba's private attributes, read defensively.
	kernel_cache = getattr(getattr(ersatz, "_fast_shuffle", None), "_cache",
		None)
	cache_path = getattr(kernel_cache, "cache_path", None)

	cache_dir = os.environ.get("NUMBA_CACHE_DIR")
	if cache_dir and cache_path is not None:
		root = os.path.realpath(cache_dir)
		path = os.path.realpath(cache_path)
		if os.path.commonpath([path, root]) != root:
			logger.warning("tangermeme's numba kernel is cached in %s, not "
				"under NUMBA_CACHE_DIR=%s, because tangermeme was imported "
				"before the variable was set; each reference worker compiles "
				"the kernel itself", cache_path, cache_dir)

	return cache_path


def _check_tokens(tokens):
	"""`tokens` as a C-contiguous (n, L) uint8 array with values in
	0..N_TOKEN, or raise."""

	if isinstance(tokens, torch.Tensor):
		tokens = tokens.cpu().numpy()

	t = numpy.ascontiguousarray(tokens)
	if t.dtype != numpy.uint8:
		raise TypeError("tokens must be uint8, got {}".format(t.dtype))

	if t.ndim != 2:
		raise ValueError("tokens must have shape (n_sequences, length), got "
			"{}".format(t.shape))

	if t.size and int(t.max()) > N_TOKEN:
		raise ValueError("tokens must lie in 0-{}, got {}".format(N_TOKEN,
			int(t.max())))

	return t


def _seed(random_state):
	try:
		return operator.index(random_state)
	except TypeError:
		raise TypeError("random_state must be an integer; for random_state "
			"None, draw a base seed once. Got {!r}".format(random_state)
			) from None


def _positive(value, name):
	v = operator.index(value)
	if v < 1:
		raise ValueError("{} must be at least 1, got {}".format(name, v))

	return v
