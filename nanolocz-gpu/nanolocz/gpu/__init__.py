"""
GPU acceleration module for NanoLocz.

Provides CUDA-accelerated operations using CuPy with automatic
fallback to CPU when GPU is unavailable.
"""

from nanolocz.gpu.utils import (
    get_array_module,
    to_gpu,
    from_gpu,
    GPUArrayModule,
    CUPY_AVAILABLE,
)

__all__ = [
    'get_array_module',
    'to_gpu',
    'from_gpu',
    'GPUArrayModule',
    'CUPY_AVAILABLE',
]
