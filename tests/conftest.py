# Shared pytest fixtures for the cherimoya test suite.

import os

# Disable torch.compile globally for the test suite. Tests need to run
# quickly, deterministically, and on CPU; the compiled forward path adds
# minutes of warm-up and isn't what we're trying to verify here.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import pytest
import torch


def _has_cuda():
	return torch.cuda.is_available()


def _has_triton():
	try:
		import triton  # noqa: F401
	except ImportError:
		return False
	return _has_cuda()


@pytest.fixture(scope="session")
def cuda_available():
	"""True if a CUDA device is visible to PyTorch."""
	return _has_cuda()


@pytest.fixture(scope="session")
def device():
	"""The default device for tests — GPU if available, else CPU."""
	return "cuda" if _has_cuda() else "cpu"


def pytest_collection_modifyitems(config, items):
	# Skip tests marked `cuda` when no GPU is available, and `triton` when
	# Triton isn't usable (no GPU or no install).
	skip_cuda = pytest.mark.skip(reason="No CUDA device available")
	skip_triton = pytest.mark.skip(reason="Triton requires CUDA and an install")
	for item in items:
		if "cuda" in item.keywords and not _has_cuda():
			item.add_marker(skip_cuda)
		if "triton" in item.keywords and not _has_triton():
			item.add_marker(skip_triton)


@pytest.fixture
def forced(monkeypatch):
	"""`cherimoya.deep_lift_shap.deep_lift_shap` takes the fast path for any
	job, however quick, where it applies."""

	monkeypatch.setattr("cherimoya.deep_lift_shap._FAST_MIN_SECONDS", 0.0)


@pytest.fixture
def engines(monkeypatch):
	"""The fast-path engines `cherimoya.deep_lift_shap.deep_lift_shap`
	builds, in order."""

	from cherimoya._deep_lift_shap_engine import Engine

	built = []

	def counting(*args, **kwargs):
		built.append(Engine(*args, **kwargs))
		return built[-1]

	monkeypatch.setattr("cherimoya.deep_lift_shap.Engine", counting)
	return built
