"""Fast, batched, multi-angle 2D fractional Fourier transform for PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from ._frft2d import frft2d_multi

try:
    __version__ = version("torch-frft2d")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__", "frft2d_multi"]
