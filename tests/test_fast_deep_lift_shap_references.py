"""Tests that the fast engine's references are, bit for bit, the ones
tangermeme's `deep_lift_shap` draws.

`deep_lift_shap` gives the pair of sequence e and reference k the reference
``dinucleotide_shuffle(X[e:e+1].float(), n=1, random_state=rs + k)[:, 0]``.
`cherimoya.fast_deep_lift_shap.references` calls that same function in
spawned worker processes and returns each reference as uint8 tokens. These
tests hold it to tangermeme called in this process, on random sequences of
the default input length and on sequences that take the shuffle kernel's
corner paths, with two seeds and through pools of one and two workers.
"""

import os

import numpy
import pytest
import torch

from pathlib import Path

from tangermeme import ersatz

from cherimoya.fast_deep_lift_shap import references


L = 2114  # cherimoya attribute's default in_window
K = 20  # cherimoya attribute's default n_shuffles
RANDOM_STATES = (0, 1234)
N_RANDOM = 40
SEQS_PER_CHUNK = 7  # does not divide the number of sequences


def _onehot(tokens):
	"""(n, L) base indices -> (n, 4, L) int8 one-hot, built without the
	module under test."""

	return torch.from_numpy(numpy.ascontiguousarray(numpy.eye(4,
		dtype=numpy.int8)[tokens].transpose(0, 2, 1)))


def _edge_cases(rng):
	"""Sequences that take the shuffle kernel's corner paths. G is the rare
	base: the kernel permutes each base's successors with permutation(n - 1),
	so a base seen at most twice before the last position draws no random
	numbers."""

	def without_g():
		return rng.choice([0, 1, 3], L)

	cases = {"homopolymer A": numpy.zeros(L, numpy.int64),
		"homopolymer T": numpy.full(L, 3, numpy.int64)}
	cases["no T"] = rng.randint(0, 3, L)
	cases["no A"] = rng.randint(1, 4, L)
	cases["A and T only"] = rng.choice([0, 3], L)

	seq = without_g()
	seq[rng.choice(numpy.arange(1, L - 1), 2, replace=False)] = 2
	cases["G twice, inside"] = seq

	seq = without_g()
	seq[-1] = 2
	cases["G once, last base"] = seq

	seq = without_g()
	seq[[0, -1]] = 2
	cases["G first and last base"] = seq

	cases["alternating AC"] = numpy.tile([0, 1], L // 2)
	cases["alternating GT"] = numpy.tile([2, 3], L // 2)
	return cases


@pytest.fixture(scope="module")
def sequences():
	"""Random sequences, then the edge cases: names, base indices (n, L)
	uint8 and the int8 one-hot (n, 4, L)."""

	rng = numpy.random.RandomState(20261008)
	cases = _edge_cases(rng)
	tokens = numpy.concatenate([rng.randint(0, 4, (N_RANDOM, L)),
		numpy.stack(list(cases.values()))])
	names = ["random {}".format(i) for i in range(N_RANDOM)] + list(cases)
	return names, tokens.astype(numpy.uint8), _onehot(tokens)


@pytest.fixture(scope="module")
def truth(sequences):
	"""tangermeme's reference for every pair, from the call `deep_lift_shap`
	makes: random_state -> (n, K, 4, L) float32."""

	_, _, X = sequences
	out = {}
	for rs in RANDOM_STATES:
		out[rs] = torch.stack([torch.cat([ersatz.dinucleotide_shuffle(
			X[e:e + 1].float(), n=1, random_state=rs + k)[:, 0]
			for k in range(K)]) for e in range(len(X))])

	return out


def _mismatches(names, got, expected, rows):
	"""Pairs whose tokens, re-expanded to a float32 one-hot, are not
	torch.equal to tangermeme's reference."""

	bad = []
	for i, e in enumerate(rows):
		for k in range(K):
			ref = references.tokens_to_onehot(got[i, k])
			if ref.dtype != torch.float32 or not torch.equal(ref,
					expected[e, k]):
				bad.append((names[e], k))

	return bad


##


def test_tokens_round_trip(sequences, truth):
	"""The encoding is lossless, N columns included."""

	_, tokens, X = sequences
	got = references.onehot_to_tokens(X)
	assert got.dtype == numpy.uint8 and got.shape == tokens.shape
	numpy.testing.assert_array_equal(got, tokens)

	back = references.tokens_to_onehot(got, dtype=torch.int8)
	assert back.dtype == torch.int8 and torch.equal(back, X)

	refs = truth[RANDOM_STATES[0]]
	ref_tokens = references.onehot_to_tokens(refs)
	assert ref_tokens.shape == (len(X), K, L)
	again = references.tokens_to_onehot(ref_tokens)
	assert again.dtype == torch.float32 and torch.equal(again, refs)

	x = X[:2].clone()
	x[1, :, [0, 5, L - 1]] = 0  # N columns, as extract_loci writes them
	with_n = references.onehot_to_tokens(x)
	where = set(numpy.flatnonzero(with_n[1] == references.N_TOKEN))
	assert where == {0, 5, L - 1}
	assert torch.equal(references.tokens_to_onehot(with_n, dtype=torch.int8),
		x)


def test_non_onehot_inputs_and_bad_tokens_are_refused(sequences):
	_, tokens, X = sequences
	two_bases = X[:1].clone()
	two_bases[0, :, 9] = torch.tensor([1, 1, 0, 0], dtype=torch.int8)
	not_binary = X[:1].clone()
	not_binary[0, :, 9] = torch.tensor([0, 0, 2, 0], dtype=torch.int8)
	for bad in (two_bases, not_binary, X[:1].float() * 0.5, X[:1, :3]):
		with pytest.raises(ValueError):
			references.onehot_to_tokens(bad)

	out_of_range = tokens[:1].copy()
	out_of_range[0, 3] = references.N_TOKEN + 1
	with pytest.raises(ValueError):
		references.tokens_to_onehot(out_of_range)

	with pytest.raises(TypeError):
		references.tokens_to_onehot(tokens[:1].astype(numpy.int64))

	with pytest.raises(ValueError):
		references.shuffle_tokens(out_of_range, 1, 0)

	# The arguments are checked before any work starts.
	producer = references.RefProducer(workers=0)
	for args, error in (
		((tokens[:2], K, None, 1), TypeError),
		((tokens[:2], 0, 0, 1), ValueError),
		((tokens[:2], K, 0, 0), ValueError),
		((tokens[:2, None], K, 0, 1), ValueError),
		((tokens[:2].astype(numpy.int8), K, 0, 1), TypeError),
		((out_of_range, K, 0, 1), ValueError),
	):
		with pytest.raises(error):
			producer.chunks(*args)


def test_shuffle_tokens_in_process_matches_tangermeme(sequences, truth):
	"""The worker body run in this process, alone and as a producer without
	workers: the edge cases and three random sequences."""

	names, tokens, _ = sequences
	rows = list(range(N_RANDOM - 3, len(tokens)))
	for rs in RANDOM_STATES:
		got = references.shuffle_tokens(tokens[rows], K, rs)
		assert got.dtype == numpy.uint8 and got.shape == (len(rows), K, L)
		assert not _mismatches(names, got, truth[rs], rows)

	with references.RefProducer(workers=0) as producer:
		parts = list(producer.chunks(tokens[rows], K, RANDOM_STATES[1],
			SEQS_PER_CHUNK))
		stats = dict(producer.stats)

	assert [(a, b) for a, b, _ in parts] == [(a, min(a + SEQS_PER_CHUNK,
		len(rows))) for a in range(0, len(rows), SEQS_PER_CHUNK)]

	# In process, every chunk is drawn when it is asked for.
	assert stats["stalls"] == stats["chunks"]
	assert 0.0 < stats["first_wait_s"] <= stats["wait_s"]
	got = numpy.concatenate([refs for _, _, refs in parts])
	assert not _mismatches(names, got, truth[RANDOM_STATES[1]], rows)


@pytest.mark.parametrize("workers", [1, 2])
def test_pool_references_equal_tangermeme(sequences, truth, workers):
	"""Every pair, both seeds, through a spawned pool, in order."""

	names, _, X = sequences
	n = len(X)
	tokens = references.onehot_to_tokens(X)
	expected_bounds = [(a, min(a + SEQS_PER_CHUNK, n))
		for a in range(0, n, SEQS_PER_CHUNK)]

	with references.RefProducer(workers=workers) as producer:
		assert producer.window >= max(3, workers)
		for rs in RANDOM_STATES:
			# 255 is not a token, so a chunk never written would fail.
			got = numpy.full((n, K, L), 255, numpy.uint8)
			bounds = []
			for a, b, refs in producer.chunks(tokens, K, rs, SEQS_PER_CHUNK):
				bounds.append((a, b))
				got[a:b] = refs

			assert bounds == expected_bounds
			stats = dict(producer.stats)
			assert stats["max_in_flight"] == min(producer.window,
				len(expected_bounds))
			assert 0.0 <= stats["first_wait_s"] <= stats["wait_s"]

			bad = _mismatches(names, got, truth[rs], list(range(n)))
			assert not bad, "{} of {} pairs differ from tangermeme".format(
				len(bad), n * K)


def test_worker_exception_reaches_caller_with_same_type(sequences):
	"""What tangermeme raises in a worker, here for an N, reaches the caller
	with its type and message, after the chunks before it; the pool keeps
	working afterwards."""

	_, _, X = sequences
	x = X[:5].clone()
	x[3, :, 1000] = 0
	with pytest.raises(Exception) as direct:
		ersatz.dinucleotide_shuffle(x[3:4].float(), n=1, random_state=0)

	tokens = references.onehot_to_tokens(x)
	with pytest.raises(direct.type) as inline:
		references.shuffle_tokens(tokens[3:4], 2, 0)

	assert str(inline.value) == str(direct.value)

	with references.RefProducer(workers=2) as producer:
		delivered = []
		with pytest.raises(direct.type) as pooled:
			for a, _, _ in producer.chunks(tokens, 2, 0, 1):
				delivered.append(a)

		assert type(pooled.value) is direct.type
		assert str(pooled.value) == str(direct.value)
		assert delivered == [0, 1, 2]

		again = numpy.concatenate([refs for _, _, refs
			in producer.chunks(tokens[:3], 2, 0, 2)])
		numpy.testing.assert_array_equal(again,
			references.shuffle_tokens(tokens[:3], 2, 0))


def test_workers_are_single_threaded_and_load_the_cached_kernel():
	"""The parent compiles tangermeme's numba kernel into numba's cache
	before the pool starts; each worker runs one intra-op thread and sees
	the parent's NUMBA_CACHE_DIR."""

	with references.RefProducer(workers=1) as producer:
		cache_path = Path(producer.numba_cache_path)
		assert list(cache_path.glob("ersatz._fast_shuffle-*.nbi"))
		assert producer._pool.submit(torch.get_num_threads).result() == 1
		assert producer._pool.submit(os.getenv, "NUMBA_CACHE_DIR").result() \
			== os.environ.get("NUMBA_CACHE_DIR")


def test_default_workers_leave_two_cpus():
	n_cpus = len(os.sched_getaffinity(0)) if hasattr(os,
		"sched_getaffinity") else os.cpu_count()
	assert references.default_workers() == max(1, min(8, n_cpus - 2))
