"""GPU acceleration module for NanoLocz.

Provides CUDA-accelerated operations using CuPy with automatic
fallback to CPU when GPU is unavailable.
"""

from nanolocz.gpu.backend import (
    Backend,
    BackendConfig,
    BackendContext,
    CUPY_AVAILABLE,
    CPU_REFERENCE_TOLERANCE,
    CPU_STANDARD_TOLERANCE,
    CROSS_BACKEND_TOLERANCE,
    GPU_FLOAT32_TOLERANCE,
    GPU_FLOAT64_TOLERANCE,
    PrecisionMode,
    TolerancePolicy,
    assert_close,
    create_gpu_context,
    create_reference_context,
    get_backend_context,
    get_tolerance,
    validate_backend_result,
)

from nanolocz.gpu.detection import (
    detect_particles_gpu,
    local_maxima_gpu,
    min_distance_suppression_gpu,
    prominence_gpu,
    statistics_gpu,
)

from nanolocz.gpu.lafm import (
    batch_splat_gpu,
    compute_frc_gpu,
    frc_resolution,
    splat_gaussian_gpu,
    splat_localizations_gpu,
)

from nanolocz.gpu.simafm import (
    add_scan_artifacts_gpu,
    add_shot_noise_gpu,
    add_thermal_noise_gpu,
    compute_height_field_gpu,
    convolve_tip_gpu,
    simulate_afm_image_gpu,
)

from nanolocz.gpu.utils import (
    from_gpu,
    get_array_module,
    GPUArrayModule,
    to_gpu,
)

try:
    from nanolocz.gpu.classification import (
        reduce_dimensions_pca_gpu,
        classify_particles_gpu,
    )
    _HAS_CLASSIFICATION_GPU = True
except ImportError:
    _HAS_CLASSIFICATION_GPU = False

__all__ = [
    # Backend management
    'Backend',
    'BackendConfig',
    'BackendContext',
    'PrecisionMode',
    'TolerancePolicy',
    # Tolerance constants
    'CPU_REFERENCE_TOLERANCE',
    'CPU_STANDARD_TOLERANCE',
    'GPU_FLOAT32_TOLERANCE',
    'GPU_FLOAT64_TOLERANCE',
    'CROSS_BACKEND_TOLERANCE',
    # Backend functions
    'get_backend_context',
    'create_reference_context',
    'create_gpu_context',
    'get_tolerance',
    'assert_close',
    'validate_backend_result',
    # Detection kernels
    'local_maxima_gpu',
    'prominence_gpu',
    'min_distance_suppression_gpu',
    'detect_particles_gpu',
    'statistics_gpu',
    # LAFM kernels
    'splat_gaussian_gpu',
    'splat_localizations_gpu',
    'compute_frc_gpu',
    'frc_resolution',
    'batch_splat_gpu',
    # SimAFM kernels
    'compute_height_field_gpu',
    'convolve_tip_gpu',
    'add_thermal_noise_gpu',
    'add_shot_noise_gpu',
    'add_scan_artifacts_gpu',
    'simulate_afm_image_gpu',
    # Legacy utils (deprecated - use backend module instead)
    'get_array_module',
    'to_gpu',
    'from_gpu',
    'GPUArrayModule',
    # Availability flag
    'CUPY_AVAILABLE',
    # Classification (GPU-accelerated PCA)
    'reduce_dimensions_pca_gpu',
    'classify_particles_gpu',
    '_HAS_CLASSIFICATION_GPU',
]
