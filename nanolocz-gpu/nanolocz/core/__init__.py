"""
Core image processing algorithms for NanoLocz.

This module contains the main particle detection, localization, 
and tracking algorithms ported from MATLAB.
"""

from nanolocz.core.detection import fast_peaks2d, detect_particles
from nanolocz.core.parity import (
    ParityResult,
    compare_arrays,
    run_parity_test,
    generate_parity_report,
)

__all__ = [
    'fast_peaks2d',
    'detect_particles',
    'ParityResult',
    'compare_arrays',
    'run_parity_test',
    'generate_parity_report',
]
