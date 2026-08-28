"""
NanoLocz Python - AFM Image Analysis Platform

A Python port of NanoLocz with GPU acceleration for Linux systems.
No MATLAB dependency required.

Original MATLAB version: https://github.com/george-r-heath/NanoLocz
"""

__version__ = "0.1.0-dev"
__author__ = "NanoLocz Python Team"
__license__ = "GPL-3.0-or-later"

# Import core modules
from nanolocz.core import detection
from nanolocz.core.parity import (
    ParityResult,
    compare_arrays,
    run_parity_test,
    generate_parity_report,
)
from nanolocz.gpu import utils as gpu_utils
from nanolocz.formats import tiff_reader, hdf5_reader

# Expose key APIs at package level
from nanolocz.core.detection import fast_peaks2d, detect_particles

__all__ = [
    "detection",
    "gpu_utils",
    "tiff_reader",
    "hdf5_reader",
    "ParityResult",
    "compare_arrays",
    "run_parity_test",
    "generate_parity_report",
    "fast_peaks2d",
    "detect_particles",
]
