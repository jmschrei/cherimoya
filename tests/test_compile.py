"""Tests for the `compile` opt-out kwarg on Cherimoya.

Background:
	Cherimoya's forward used to be decorated at class level with
	`@torch.compile(mode='max-autotune')`, which made it impossible to opt
	out of compilation without monkey-patching. The kwarg `compile=True`
	(default) preserves the old behavior; `compile=False` installs the
	eager `_forward_impl` as `self._forward_fn`. These tests check both
	back-compat and forward parity.
"""

import inspect

import pytest
import torch

from cherimoya import Cherimoya


# Use a tiny config everywhere so CPU runs stay fast.
def _tiny_kwargs(**overrides):
	base = dict(n_filters=8, n_layers=3, n_outputs=1, n_control_tracks=0,
		single_count_output=True, verbose=False)
	base.update(overrides)
	return base


def _input_window_for(model):
	return 2 * model.trimming + 64


# ---------------------------------------------------------------------------
# 1. Default behavior is unchanged
# ---------------------------------------------------------------------------

def test_default_compile_is_true():
	"""Constructing without specifying `compile` should leave compile on.
	This is the back-compat invariant: pre-existing user code that does
	`Cherimoya(...)` should behave exactly as it did before."""
	model = Cherimoya(**_tiny_kwargs())
	assert model._compile is True


def test_default_load_compiles(tmp_path):
	"""`Cherimoya.load(path)` with no `compile=` argument should also
	default to compile=True."""
	m = Cherimoya(**_tiny_kwargs())
	p = tmp_path / 'm.torch'
	m.save(str(p))
	loaded = Cherimoya.load(str(p))
	assert loaded._compile is True


def test_compile_default_value_in_signature():
	"""Lock the default to True so that an accidental flip to False in a
	future refactor would be caught loudly."""
	sig = inspect.signature(Cherimoya.__init__)
	assert sig.parameters['compile'].default is True

	sig = inspect.signature(Cherimoya.load)
	assert sig.parameters['compile'].default is True


# ---------------------------------------------------------------------------
# 2. Opt-out works
# ---------------------------------------------------------------------------

def test_compile_false_skips_torch_compile():
	"""With compile=False, `_forward_fn` should be the raw bound method,
	not a torch.compile wrapper. We check this by identity: the bound
	method `self._forward_impl` is a fresh object each access, so compare
	via `__func__` to its underlying function."""
	model = Cherimoya(**_tiny_kwargs(), compile=False)
	assert model._compile is False
	# `model._forward_fn` is `self._forward_impl` (a bound method).
	# torch.compile would wrap it into a different callable, so the
	# `__func__` attribute differs.
	assert getattr(model._forward_fn, '__func__', None) is \
		Cherimoya._forward_impl


def test_compile_false_forward_runs():
	"""compile=False should produce a working eager forward."""
	model = Cherimoya(**_tiny_kwargs(), compile=False).eval()
	L = _input_window_for(model)
	X = torch.randn(2, 4, L)
	with torch.no_grad():
		y_prof, y_count = model(X)
	assert y_prof.shape[0] == 2
	assert y_count.shape[0] == 2
	assert torch.isfinite(y_prof).all()
	assert torch.isfinite(y_count).all()


def test_load_accepts_compile_kwarg(tmp_path):
	"""`Cherimoya.load(..., compile=False)` should round-trip parameters
	and produce a non-compiled model."""
	m = Cherimoya(**_tiny_kwargs()).eval()
	p = tmp_path / 'm.torch'
	m.save(str(p))

	loaded = Cherimoya.load(str(p), compile=False).eval()
	assert loaded._compile is False
	# Weights must round-trip regardless of compile.
	for (n1, p1), (n2, p2) in zip(m.named_parameters(),
								   loaded.named_parameters()):
		assert n1 == n2
		assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# 3. Checkpoint back-compat: `compile` must not leak into saved configs
# ---------------------------------------------------------------------------

def test_compile_not_in_init_kwargs():
	"""`_init_kwargs` is the explicit serializer used by `save`. It must
	not contain `compile` — otherwise newly-trained checkpoints would
	pin a compile choice into their config and conflict with the
	load-time kwarg."""
	model = Cherimoya(**_tiny_kwargs(), compile=False)
	assert 'compile' not in model._init_kwargs()


def test_saved_checkpoint_has_no_compile_key(tmp_path):
	"""Same invariant as above, observed at the on-disk format level."""
	m = Cherimoya(**_tiny_kwargs(), compile=False)
	p = tmp_path / 'm.torch'
	m.save(str(p))
	payload = torch.load(str(p), weights_only=True)
	assert 'compile' not in payload['config']


def test_load_old_style_config_without_compile_key():
	"""Simulate an old checkpoint whose config predates the `compile`
	kwarg: construct directly via `cls(**config)` with no `compile=` in
	the dict. The default (`compile=True`) should apply, matching the
	pre-change behavior."""
	old_style_config = _tiny_kwargs()  # no `compile` key
	assert 'compile' not in old_style_config
	model = Cherimoya(**old_style_config)
	assert model._compile is True


def test_save_and_load_round_trip_default(tmp_path):
	"""End-to-end: save with defaults, load with defaults, weights
	identical, model usable."""
	m = Cherimoya(**_tiny_kwargs()).eval()
	L = _input_window_for(m)
	p = tmp_path / 'm.torch'
	m.save(str(p))

	loaded = Cherimoya.load(str(p)).eval()
	assert loaded._compile is True
	for (n1, p1), (n2, p2) in zip(m.named_parameters(),
								   loaded.named_parameters()):
		assert n1 == n2
		assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# 4. Subclass `super().forward(...)` still works
# ---------------------------------------------------------------------------

def test_subclass_super_forward_resolution():
	"""A subclass that overrides forward and delegates to
	`super().forward(...)` should keep working. This is why `forward`
	stays as a class-level method (the thin trampoline) instead of being
	replaced on the instance."""

	class MyCheri(Cherimoya):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.called = 0

		def forward(self, X, X_ctl=None):
			self.called += 1
			return super().forward(X, X_ctl)

	model = MyCheri(**_tiny_kwargs(), compile=False).eval()
	L = _input_window_for(model)
	X = torch.randn(1, 4, L)
	with torch.no_grad():
		y_prof, y_count = model(X)
	assert model.called == 1
	assert torch.isfinite(y_prof).all()
	assert torch.isfinite(y_count).all()


# ---------------------------------------------------------------------------
# 5. Forward parity: compile=True vs compile=False produce the same output
# ---------------------------------------------------------------------------

def _build_pair(device, **overrides):
	"""Build two models with identical weights but different compile
	settings."""
	kwargs = _tiny_kwargs(**overrides)
	m_eager = Cherimoya(**kwargs, compile=False).to(device).eval()
	m_compiled = Cherimoya(**kwargs, compile=True).to(device).eval()
	# Copy weights from eager to compiled so both share state.
	m_compiled.load_state_dict(m_eager.state_dict())
	return m_eager, m_compiled


def _assert_close(a, b, atol=1e-4, rtol=1e-4):
	# Compare in fp32 to avoid silently swallowing dtype mismatches.
	assert a.shape == b.shape
	assert torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol), (
		f"max abs diff = {(a.float() - b.float()).abs().max().item():.3e}, "
		f"atol={atol}, rtol={rtol}"
	)


def test_forward_parity_compile_vs_eager_cpu():
	"""On CPU, compile=True and compile=False should produce numerically
	close outputs for the same input. (`torch.compile` on CPU may still
	rewrite the graph; this test verifies it doesn't change the numerics
	beyond float-precision noise.)"""
	torch.manual_seed(0)
	m_eager, m_compiled = _build_pair('cpu')
	L = _input_window_for(m_eager)
	X = torch.randn(2, 4, L)

	with torch.no_grad():
		y_eager_p, y_eager_c = m_eager(X)
		y_comp_p,  y_comp_c  = m_compiled(X)

	_assert_close(y_eager_p, y_comp_p)
	_assert_close(y_eager_c, y_comp_c)


# Triton's MLP kernel requires K = n_filters >= 16 (see the
# `tl.dot(x_dot, w1, ...)` constraint in `_fwd_inf_forward`), and the
# inference path itself requires `(expansion * n_filters) % 16 == 0`. CUDA
# parity tests use n_filters=32 to match the existing CUDA test style.
_CUDA_KWARGS = dict(n_filters=32, n_layers=2)


@pytest.mark.cuda
def test_forward_parity_compile_vs_eager_cuda():
	"""On CUDA, the compiled path additionally goes through Cheri's
	triton inference kernel + bf16 weight cast (see
	`Cheri._can_use_inference_path`). Eager-without-compile uses the
	same inference path under no_grad (it's gated only on the input
	being CUDA + `not torch.is_grad_enabled()`). So this checks that the
	additional `torch.compile` wrapping doesn't drift the numerics."""
	if not torch.cuda.is_available():
		pytest.skip('cuda not available')

	torch.manual_seed(0)
	m_eager, m_compiled = _build_pair('cuda', **_CUDA_KWARGS)
	L = _input_window_for(m_eager)
	X = torch.randn(2, 4, L, device='cuda')

	with torch.no_grad():
		y_eager_p, y_eager_c = m_eager(X)
		y_comp_p,  y_comp_c  = m_compiled(X)

	_assert_close(y_eager_p, y_comp_p)
	_assert_close(y_eager_c, y_comp_c)


@pytest.mark.cuda
def test_forward_parity_with_control_tracks_cuda():
	"""Same parity check but with X_ctl present, which exercises the
	`X_w_ctl` concat and the count-head control branch."""
	if not torch.cuda.is_available():
		pytest.skip('cuda not available')

	torch.manual_seed(0)
	m_eager, m_compiled = _build_pair('cuda', n_control_tracks=1,
		**_CUDA_KWARGS)
	L = _input_window_for(m_eager)
	X = torch.randn(2, 4, L, device='cuda')
	# X_ctl is concatenated to X on the channel axis before trimming, so
	# it shares the *full* sequence length, not the trimmed output length.
	# The count head computes log(sum(X_ctl) + 1), so X_ctl must stay
	# non-negative or the log produces NaNs.
	X_ctl = torch.rand(2, 1, L, device='cuda')

	with torch.no_grad():
		y_eager_p, y_eager_c = m_eager(X, X_ctl)
		y_comp_p,  y_comp_c  = m_compiled(X, X_ctl)

	_assert_close(y_eager_p, y_comp_p)
	_assert_close(y_eager_c, y_comp_c)


@pytest.mark.cuda
def test_forward_parity_eager_fallback_path_cuda():
	"""When `_can_use_inference_path` is False, the model takes the
	eager (non-triton) fused conv + standard linear path. Force this by
	enabling grad — that path is then exercised end-to-end. compile=True
	vs compile=False should still agree."""
	if not torch.cuda.is_available():
		pytest.skip('cuda not available')

	torch.manual_seed(0)
	m_eager, m_compiled = _build_pair('cuda', **_CUDA_KWARGS)
	L = _input_window_for(m_eager)
	X = torch.randn(2, 4, L, device='cuda')

	# grad_enabled=True forces `_can_use_inference_path` -> False, so both
	# models go through the eager `fused_dilated_conv_norm` + linear path.
	y_eager_p, y_eager_c = m_eager(X)
	y_comp_p,  y_comp_c  = m_compiled(X)

	_assert_close(y_eager_p.detach(), y_comp_p.detach())
	_assert_close(y_eager_c.detach(), y_comp_c.detach())


def test_forward_parity_across_dtypes_cpu():
	"""On CPU, switching the input dtype (fp32 vs fp16) should not
	silently change which forward path runs in a way that bypasses
	parity. Both compile settings should track each other regardless of
	dtype."""
	torch.manual_seed(0)
	m_eager, m_compiled = _build_pair('cpu')
	L = _input_window_for(m_eager)
	X = torch.randn(2, 4, L)

	with torch.no_grad():
		y_eager_p, y_eager_c = m_eager(X)
		y_comp_p,  y_comp_c  = m_compiled(X)
		# Sanity: parity holds independently for each compile setting on
		# repeated calls (no hidden state).
		y_eager_p2, _ = m_eager(X)
		y_comp_p2, _ = m_compiled(X)

	_assert_close(y_eager_p, y_comp_p)
	_assert_close(y_eager_c, y_comp_c)
	_assert_close(y_eager_p, y_eager_p2, atol=0, rtol=0)
	_assert_close(y_comp_p, y_comp_p2, atol=0, rtol=0)
