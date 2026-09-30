# cherimoya
# Author: Jacob Schreiber

from .cherimoya import Cherimoya
from .cherimoya import EMA
from .cheri import CheriBlock
from .wrappers import ControlWrapper
from .wrappers import ProfileWrapper
from .wrappers import LogCountWrapper
from .wrappers import ExpectedCountsWrapper

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

# An import from a source tree that was never installed, such as the Read
# the Docs build, has no package metadata.
try:
	__version__ = version("cherimoya")
except PackageNotFoundError:
	__version__ = "unknown"
