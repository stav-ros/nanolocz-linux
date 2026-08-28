"""
Core image processing algorithms and parity testing utilities for NanoLocz.

This module contains the main particle detection, localization, 
and tracking algorithms ported from MATLAB, along with NL-02 parity
infrastructure for validating numerical equivalence.
"""

from nanolocz.core.detection import fast_peaks2d, detect_particles
from nanolocz.core.types import (
    Frame,
    Localizations,
    Meta,
    ParticleStack,
    DetectionResult,
    LocalizedParticle,
    ParticleTrack,
    ImageMetadata,
)
from nanolocz.core.parity import (
    ParityResult,
    compare_arrays,
    run_parity_test,
    generate_parity_report,
    # NL-02 additions
    TolerancePolicy,
    CPU_TOLERANCE,
    GPU_TOLERANCE,
    ParityError,
    FixtureError,
    ToleranceError,
    load_npy_fixture,
    assert_close,
)

__all__ = [
    'fast_peaks2d',
    'detect_particles',
    # Core data contracts (NL-03)
    'Frame',
    'Localizations',
    'Meta',
    'ParticleStack',
    'DetectionResult',
    'LocalizedParticle',
    'ParticleTrack',
    'ImageMetadata',
    # Parity testing
    'ParityResult',
    'compare_arrays',
    'run_parity_test',
    'generate_parity_report',
    # NL-02 tolerance policies
    'TolerancePolicy',
    'CPU_TOLERANCE',
    'GPU_TOLERANCE',
    # NL-02 exceptions
    'ParityError',
    'FixtureError',
    'ToleranceError',
    # NL-02 utilities
    'load_npy_fixture',
    'assert_close',
]
