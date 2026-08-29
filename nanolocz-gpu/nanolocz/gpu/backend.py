"""NumPy/CuPy backend management and precision policy.

This module provides the foundation for CPU/GPU execution with consistent
numerical behavior across backends. It defines:

- Backend selection and switching policies
- Float64 CPU reference behavior
- GPU precision and tolerance rules
- Array conversion utilities between backends
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

# Try to import CuPy for GPU support
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class Backend(Enum):
    """Array computation backend."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class PrecisionMode(Enum):
    """Numerical precision mode for computations."""
    # Reference precision: always float64 on CPU
    REFERENCE = "reference"
    # Mixed precision: float32 on GPU, float64 on CPU
    MIXED = "mixed"
    # High precision: float64 everywhere (GPU if supported)
    HIGH = "high"


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for backend execution.
    
    Attributes:
        backend: Target backend (CPU, CUDA, or AUTO)
        precision: Numerical precision mode
        device_id: CUDA device ID (used when backend=CUDA)
        memory_limit: Maximum GPU memory to use in bytes (None for no limit)
        allow_downcast: Allow automatic downcasting to float32 on GPU
    """
    backend: Backend = Backend.AUTO
    precision: PrecisionMode = PrecisionMode.MIXED
    device_id: int | None = None
    memory_limit: int | None = None
    allow_downcast: bool = True
    
    def resolve_backend(self) -> Backend:
        """Resolve actual backend based on availability."""
        if self.backend != Backend.AUTO:
            return self.backend
        
        # Auto-detect: prefer CUDA if available
        if CUPY_AVAILABLE:
            try:
                # Check if any CUDA devices are available
                if cp.cuda.runtime.getDeviceCount() > 0:
                    return Backend.CUDA
            except Exception:
                pass
        
        return Backend.CPU
    
    def get_dtype(self, is_gpu: bool = False) -> np.dtype:
        """Get default dtype based on precision mode and backend.
        
        Args:
            is_gpu: Whether the target is GPU
            
        Returns:
            NumPy dtype for array operations
        """
        if self.precision == PrecisionMode.REFERENCE:
            # Always float64 for reference computations
            return np.float64
        elif self.precision == PrecisionMode.HIGH:
            # Float64 everywhere
            return np.float64
        else:
            # MIXED mode: float32 on GPU, float64 on CPU
            if is_gpu and self.allow_downcast:
                return np.float32
            return np.float64


@dataclass(frozen=True)
class TolerancePolicy:
    """Precision tolerance policy for numerical comparisons.
    
    Defines acceptable tolerances for comparing results across different
    backends and precision modes.
    
    Attributes:
        rtol: Relative tolerance
        atol: Absolute tolerance  
        name: Human-readable identifier
        description: Explanation of when this tolerance applies
    """
    rtol: float
    atol: float
    name: str
    description: str = ""


# ============================================================================
# Tolerance definitions
# ============================================================================

# CPU reference: strictest tolerance for float64 computations
CPU_REFERENCE_TOLERANCE = TolerancePolicy(
    rtol=1e-10,
    atol=1e-12,
    name="cpu-reference-float64",
    description="Reference tolerance for CPU float64 computations"
)

# CPU standard: standard float64 tolerance
CPU_STANDARD_TOLERANCE = TolerancePolicy(
    rtol=1e-5,
    atol=1e-8,
    name="cpu-standard-float64", 
    description="Standard tolerance for CPU float64 computations"
)

# GPU float32: relaxed tolerance for single-precision GPU computations
GPU_FLOAT32_TOLERANCE = TolerancePolicy(
    rtol=1e-3,
    atol=1e-5,
    name="gpu-float32",
    description="Tolerance for GPU float32 computations"
)

# GPU float64: tighter tolerance for double-precision GPU computations
GPU_FLOAT64_TOLERANCE = TolerancePolicy(
    rtol=1e-7,
    atol=1e-10,
    name="gpu-float64",
    description="Tolerance for GPU float64 computations"
)

# Cross-backend: tolerance when comparing CPU vs GPU results
CROSS_BACKEND_TOLERANCE = TolerancePolicy(
    rtol=1e-4,
    atol=1e-6,
    name="cross-backend",
    description="Tolerance for cross-backend comparison (CPU vs GPU)"
)


def get_tolerance(
    backend: Backend,
    precision: PrecisionMode,
    compare_mode: Literal["self", "cross"] = "self"
) -> TolerancePolicy:
    """Get appropriate tolerance for given backend and precision configuration.
    
    Args:
        backend: Target backend
        precision: Precision mode
        compare_mode: Whether comparing within same backend or across backends
        
    Returns:
        Appropriate TolerancePolicy for the configuration
    """
    if compare_mode == "cross":
        return CROSS_BACKEND_TOLERANCE
    
    if backend == Backend.CPU:
        if precision == PrecisionMode.REFERENCE:
            return CPU_REFERENCE_TOLERANCE
        return CPU_STANDARD_TOLERANCE
    
    # GPU backend
    if precision == PrecisionMode.HIGH:
        return GPU_FLOAT64_TOLERANCE
    return GPU_FLOAT32_TOLERANCE


class BackendArray(Protocol):
    """Protocol for array-like objects from any backend."""
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Return array shape."""
        ...
    
    @property
    def dtype(self) -> Any:
        """Return array dtype."""
        ...
    
    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        ...
    
    def __getitem__(self, key: Any) -> Any:
        """Support indexing."""
        ...


@runtime_checkable
class GPUArray(BackendArray, Protocol):
    """Protocol for GPU array objects (CuPy)."""
    
    def get(self) -> np.ndarray:
        """Transfer array from GPU to CPU."""
        ...


@dataclass
class BackendContext:
    """Execution context for backend operations.
    
    Manages array creation, transfer, and computation across CPU/GPU backends.
    
    Examples:
        >>> ctx = BackendContext(backend=Backend.CUDA)
        >>> arr = ctx.ones((100, 100))
        >>> result = ctx.xp.sum(arr)
        >>> cpu_result = ctx.to_cpu(result)
    """
    config: BackendConfig = field(default_factory=BackendConfig)
    _xp: Any = field(init=False, default=None)
    _backend: Backend = field(init=False, default=None)
    
    def __post_init__(self):
        """Initialize backend and array module."""
        resolved = self.config.resolve_backend()
        object.__setattr__(self, '_backend', resolved)
        
        if resolved == Backend.CUDA and CUPY_AVAILABLE:
            object.__setattr__(self, '_xp', cp)
            if self.config.device_id is not None:
                cp.cuda.Device(self.config.device_id).use()
        else:
            object.__setattr__(self, '_xp', np)
    
    @property
    def xp(self) -> Any:
        """Get array module (numpy or cupy)."""
        return self._xp
    
    @property
    def backend(self) -> Backend:
        """Get resolved backend."""
        return self._backend
    
    @property
    def is_gpu(self) -> bool:
        """Check if using GPU backend."""
        return self._backend == Backend.CUDA
    
    @property
    def dtype(self) -> np.dtype:
        """Get default dtype for this context."""
        return self.config.get_dtype(is_gpu=self.is_gpu)
    
    def allocate(self, shape: tuple[int, ...], dtype: np.dtype | None = None) -> Any:
        """Allocate zero-initialized array on current backend.
        
        Args:
            shape: Array shape
            dtype: Data type (uses context default if None)
            
        Returns:
            Zero-filled array on current backend
        """
        if dtype is None:
            dtype = self.dtype
        return self._xp.zeros(shape, dtype=dtype)
    
    def array(self, data: Any, dtype: np.dtype | None = None) -> Any:
        """Create array on current backend from input data.
        
        Args:
            data: Input data (list, numpy array, etc.)
            dtype: Target data type (uses context default if None)
            
        Returns:
            Array on current backend
        """
        if dtype is None:
            dtype = self.dtype
        return self._xp.asarray(data, dtype=dtype)
    
    def to_cpu(self, arr: Any) -> np.ndarray:
        """Transfer array to CPU as numpy array.
        
        Args:
            arr: Array from any backend
            
        Returns:
            NumPy array on CPU
        """
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)
    
    def to_backend(self, arr: np.ndarray, dtype: np.dtype | None = None) -> Any:
        """Transfer numpy array to current backend.
        
        Args:
            arr: NumPy array on CPU
            dtype: Target data type
            
        Returns:
            Array on current backend
        """
        if dtype is None:
            dtype = self.dtype
        if self.is_gpu and CUPY_AVAILABLE:
            return cp.asarray(arr, dtype=dtype)
        return np.asarray(arr, dtype=dtype)
    
    def ones(self, shape: tuple[int, ...], dtype: np.dtype | None = None) -> Any:
        """Create array of ones on current backend."""
        if dtype is None:
            dtype = self.dtype
        return self._xp.ones(shape, dtype=dtype)
    
    def zeros(self, shape: tuple[int, ...], dtype: np.dtype | None = None) -> Any:
        """Create array of zeros on current backend."""
        if dtype is None:
            dtype = self.dtype
        return self._xp.zeros(shape, dtype=dtype)
    
    def empty(self, shape: tuple[int, ...], dtype: np.dtype | None = None) -> Any:
        """Create uninitialized array on current backend."""
        if dtype is None:
            dtype = self.dtype
        return self._xp.empty(shape, dtype=dtype)
    
    def copy(self, arr: Any) -> Any:
        """Create a copy of array on same backend."""
        return self._xp.array(arr, copy=True)
    
    def astype(self, arr: Any, dtype: np.dtype) -> Any:
        """Cast array to different dtype."""
        return arr.astype(dtype)
    
    def get_stream(self) -> Any | None:
        """Get CUDA stream if available, None otherwise."""
        if self.is_gpu and CUPY_AVAILABLE:
            return cp.cuda.Stream()
        return None


def assert_close(
    actual: Any,
    expected: Any,
    *,
    tolerance: TolerancePolicy = CPU_STANDARD_TOLERANCE,
    label: str = "array",
) -> None:
    """Assert numerical closeness with shape checks and NaN equality.
    
    This is the primary function for verifying parity between:
    - CPU and GPU implementations
    - Different precision modes
    - Computed results vs golden fixtures
    
    Args:
        actual: Computed array (can be numpy or cupy)
        expected: Expected/reference array (can be numpy or cupy)
        tolerance: Tolerance policy to use
        label: Label for error messages
        
    Raises:
        AssertionError: If arrays differ beyond tolerance
    """
    # Convert to numpy for comparison
    actual_array = np.asarray(actual) if not hasattr(actual, 'get') else actual.get()
    expected_array = np.asarray(expected) if not hasattr(expected, 'get') else expected.get()
    
    # Check shapes match
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{label} shape mismatch: {actual_array.shape} != {expected_array.shape}"
        )
    
    # Integer types: require exact equality
    if actual_array.dtype.kind in "biu" and expected_array.dtype.kind in "biu":
        np.testing.assert_array_equal(
            actual_array, expected_array, 
            err_msg=f"{label} (integer equality check)"
        )
        return
    
    # Floating point: use tolerance-based comparison
    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
        equal_nan=True,
        err_msg=f"{label} ({tolerance.name})",
    )


def validate_backend_result(
    result: Any,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype | None = None,
    finite: bool = True,
    label: str = "result",
) -> None:
    """Validate a backend computation result meets expectations.
    
    Args:
        result: Computed result array
        expected_shape: Expected shape
        expected_dtype: Expected dtype (None to skip dtype check)
        finite: Require all values to be finite (no NaN/Inf)
        label: Label for error messages
        
    Raises:
        AssertionError: If validation fails
    """
    # Get as numpy for validation
    if hasattr(result, 'get'):
        result_np = result.get()
    else:
        result_np = np.asarray(result)
    
    # Shape check
    if result_np.shape != expected_shape:
        raise AssertionError(
            f"{label} shape mismatch: expected {expected_shape}, got {result_np.shape}"
        )
    
    # Dtype check
    if expected_dtype is not None:
        if result_np.dtype != expected_dtype:
            raise AssertionError(
                f"{label} dtype mismatch: expected {expected_dtype}, got {result_np.dtype}"
            )
    
    # Finite check
    if finite:
        if not np.all(np.isfinite(result_np)):
            n_nonfinite = np.sum(~np.isfinite(result_np))
            raise AssertionError(
                f"{label} contains {n_nonfinite} non-finite values"
            )


# ============================================================================
# Convenience functions for quick backend access
# ============================================================================

def get_backend_context(
    backend: Backend | str = Backend.AUTO,
    precision: PrecisionMode | str = PrecisionMode.MIXED,
    device_id: int | None = None,
) -> BackendContext:
    """Create a backend context with specified configuration.
    
    Args:
        backend: Target backend ('cpu', 'cuda', or 'auto')
        precision: Precision mode ('reference', 'mixed', or 'high')
        device_id: CUDA device ID (for multi-GPU systems)
        
    Returns:
        Configured BackendContext
    """
    if isinstance(backend, str):
        backend = Backend(backend.lower())
    if isinstance(precision, str):
        precision = PrecisionMode(precision.lower())
    
    config = BackendConfig(
        backend=backend,
        precision=precision,
        device_id=device_id,
    )
    return BackendContext(config=config)


def create_reference_context() -> BackendContext:
    """Create a CPU context with reference (float64) precision.
    
    This is the recommended context for:
    - Golden fixture generation
    - Reference implementation validation
    - High-accuracy scientific computations
    
    Returns:
        BackendContext configured for CPU float64
    """
    return BackendContext(
        config=BackendConfig(
            backend=Backend.CPU,
            precision=PrecisionMode.REFERENCE,
        )
    )


def create_gpu_context(
    device_id: int | None = None,
    high_precision: bool = False,
) -> BackendContext:
    """Create a GPU context with specified precision.
    
    Args:
        device_id: CUDA device ID
        high_precision: Use float64 instead of float32
        
    Returns:
        BackendContext configured for GPU
        
    Raises:
        RuntimeError: If CuPy is not available or no CUDA devices
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available - cannot create GPU context")
    
    precision = PrecisionMode.HIGH if high_precision else PrecisionMode.MIXED
    return BackendContext(
        config=BackendConfig(
            backend=Backend.CUDA,
            precision=precision,
            device_id=device_id,
        )
    )


# Export public API
__all__ = [
    # Enums
    'Backend',
    'PrecisionMode',
    # Configuration
    'BackendConfig',
    'BackendContext',
    'TolerancePolicy',
    # Tolerance constants
    'CPU_REFERENCE_TOLERANCE',
    'CPU_STANDARD_TOLERANCE',
    'GPU_FLOAT32_TOLERANCE',
    'GPU_FLOAT64_TOLERANCE',
    'CROSS_BACKEND_TOLERANCE',
    # Functions
    'get_tolerance',
    'assert_close',
    'validate_backend_result',
    'get_backend_context',
    'create_reference_context',
    'create_gpu_context',
    # Protocols
    'BackendArray',
    'GPUArray',
    # Availability flag
    'CUPY_AVAILABLE',
]
