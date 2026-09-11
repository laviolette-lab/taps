"""TAPS prostate-segmentation package."""

import warnings

# Suppress upstream MONAI/PyTorch deprecation warning triggered on MONAI package scan
warnings.filterwarnings(
    "ignore",
    message=r".*`torch\.jit\.interface` is deprecated.*",
    category=DeprecationWarning,
)

from taps.__about__ import __version__
from taps.inference import segment

__all__ = ["__version__", "segment"]
