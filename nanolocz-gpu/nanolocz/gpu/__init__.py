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

from nanolocz.gpu.utils import (
    from_gpu,
    get_array_module,
    GPUArrayModule,
    to_gpu,
)

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
    # Legacy utils (deprecated - use backend module instead)
    'get_array_module',
    'to_gpu',
    'from_gpu',
    'GPUArrayModule',
    # Availability flag
    'CUPY_AVAILABLE',
]
