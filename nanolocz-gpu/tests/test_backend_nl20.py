"""Backend consistency tests for NL-20.

Tests for NumPy/CuPy backend switch and precision policy:
- CPU/GPU array selection
- Float64 CPU reference behavior  
- GPU precision/tolerance rules
- Backend consistency across operations
"""

import numpy as np
import pytest

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


# ============================================================================
# Backend Configuration Tests
# ============================================================================

class TestBackendConfig:
    """Test backend configuration and resolution."""
    
    def test_default_config_auto_resolves_to_cpu_when_no_gpu(self):
        """Default config should resolve to CPU when GPU unavailable."""
        config = BackendConfig()
        assert config.backend == Backend.AUTO
        
        resolved = config.resolve_backend()
        if not CUPY_AVAILABLE:
            assert resolved == Backend.CPU
    
    def test_explicit_cpu_backend(self):
        """Explicit CPU backend should always resolve to CPU."""
        config = BackendConfig(backend=Backend.CPU)
        assert config.resolve_backend() == Backend.CPU
    
    def test_explicit_cuda_backend(self):
        """Explicit CUDA backend should return CUDA even if unavailable."""
        config = BackendConfig(backend=Backend.CUDA)
        # Note: resolve_backend returns what was requested, actual availability
        # is checked in BackendContext
        assert config.resolve_backend() == Backend.CUDA
    
    def test_precision_mode_reference_always_float64(self):
        """Reference precision mode should always use float64."""
        config_ref = BackendConfig(precision=PrecisionMode.REFERENCE)
        assert config_ref.get_dtype(is_gpu=False) == np.float64
        assert config_ref.get_dtype(is_gpu=True) == np.float64
    
    def test_precision_mode_high_always_float64(self):
        """High precision mode should always use float64."""
        config_high = BackendConfig(precision=PrecisionMode.HIGH)
        assert config_high.get_dtype(is_gpu=False) == np.float64
        assert config_high.get_dtype(is_gpu=True) == np.float64
    
    def test_precision_mode_mixed_gpu_float32_cpu_float64(self):
        """Mixed precision: float32 on GPU, float64 on CPU."""
        config_mixed = BackendConfig(precision=PrecisionMode.MIXED)
        assert config_mixed.get_dtype(is_gpu=False) == np.float64
        assert config_mixed.get_dtype(is_gpu=True) == np.float32
    
    def test_allow_downcast_false_keeps_float64_on_gpu(self):
        """Disabling downcast should keep float64 even in mixed mode."""
        config = BackendConfig(
            precision=PrecisionMode.MIXED,
            allow_downcast=False
        )
        assert config.get_dtype(is_gpu=True) == np.float64


# ============================================================================
# Tolerance Policy Tests
# ============================================================================

class TestTolerancePolicy:
    """Test tolerance policy definitions and selection."""
    
    def test_cpu_reference_tolerance_values(self):
        """CPU reference tolerance should be strictest."""
        assert CPU_REFERENCE_TOLERANCE.rtol == 1e-10
        assert CPU_REFERENCE_TOLERANCE.atol == 1e-12
    
    def test_cpu_standard_tolerance_values(self):
        """CPU standard tolerance values."""
        assert CPU_STANDARD_TOLERANCE.rtol == 1e-5
        assert CPU_STANDARD_TOLERANCE.atol == 1e-8
    
    def test_gpu_float32_tolerance_values(self):
        """GPU float32 tolerance should be relaxed."""
        assert GPU_FLOAT32_TOLERANCE.rtol == 1e-3
        assert GPU_FLOAT32_TOLERANCE.atol == 1e-5
    
    def test_gpu_float64_tolerance_values(self):
        """GPU float64 tolerance should be tighter than float32."""
        assert GPU_FLOAT64_TOLERANCE.rtol == 1e-7
        assert GPU_FLOAT64_TOLERANCE.atol == 1e-10
    
    def test_cross_backend_tolerance_values(self):
        """Cross-backend tolerance for CPU vs GPU comparison."""
        assert CROSS_BACKEND_TOLERANCE.rtol == 1e-4
        assert CROSS_BACKEND_TOLERANCE.atol == 1e-6
    
    def test_get_tolerance_cpu_reference(self):
        """Get tolerance for CPU reference mode."""
        tol = get_tolerance(Backend.CPU, PrecisionMode.REFERENCE)
        assert tol == CPU_REFERENCE_TOLERANCE
    
    def test_get_tolerance_cpu_standard(self):
        """Get tolerance for CPU standard mode."""
        tol = get_tolerance(Backend.CPU, PrecisionMode.MIXED)
        assert tol == CPU_STANDARD_TOLERANCE
    
    def test_get_tolerance_gpu_float32(self):
        """Get tolerance for GPU float32 mode."""
        tol = get_tolerance(Backend.CUDA, PrecisionMode.MIXED)
        assert tol == GPU_FLOAT32_TOLERANCE
    
    def test_get_tolerance_gpu_float64(self):
        """Get tolerance for GPU float64 mode."""
        tol = get_tolerance(Backend.CUDA, PrecisionMode.HIGH)
        assert tol == GPU_FLOAT64_TOLERANCE
    
    def test_get_tolerance_cross_backend(self):
        """Get cross-backend tolerance."""
        tol = get_tolerance(Backend.CPU, PrecisionMode.REFERENCE, compare_mode="cross")
        assert tol == CROSS_BACKEND_TOLERANCE


# ============================================================================
# Backend Context Tests
# ============================================================================

class TestBackendContext:
    """Test backend context creation and array operations."""
    
    def test_cpu_context_uses_numpy(self):
        """CPU context should use numpy."""
        ctx = get_backend_context(backend=Backend.CPU)
        assert ctx.xp is np
        assert ctx.backend == Backend.CPU
        assert ctx.is_gpu is False
    
    def test_cpu_context_dtype_is_float64(self):
        """CPU context default dtype should be float64."""
        ctx = get_backend_context(backend=Backend.CPU)
        assert ctx.dtype == np.float64
    
    def test_reference_context_is_cpu_float64(self):
        """Reference context should be CPU with float64."""
        ctx = create_reference_context()
        assert ctx.backend == Backend.CPU
        assert ctx.dtype == np.float64
        assert ctx.xp is np
    
    def test_context_array_creation_zeros(self):
        """Test zeros array creation."""
        ctx = get_backend_context(backend=Backend.CPU)
        arr = ctx.zeros((3, 4))
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float64
        assert np.all(arr == 0)
    
    def test_context_array_creation_ones(self):
        """Test ones array creation."""
        ctx = get_backend_context(backend=Backend.CPU)
        arr = ctx.ones((2, 3))
        assert arr.shape == (2, 3)
        assert arr.dtype == np.float64
        assert np.all(arr == 1)
    
    def test_context_array_creation_from_data(self):
        """Test array creation from data."""
        ctx = get_backend_context(backend=Backend.CPU)
        data = [[1.0, 2.0], [3.0, 4.0]]
        arr = ctx.array(data)
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr, data)
    
    def test_context_to_cpu_with_numpy(self):
        """Test to_cpu with numpy array (no-op)."""
        ctx = get_backend_context(backend=Backend.CPU)
        arr = np.array([1.0, 2.0, 3.0])
        result = ctx.to_cpu(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
    
    def test_context_copy(self):
        """Test array copy."""
        ctx = get_backend_context(backend=Backend.CPU)
        arr = ctx.array([1.0, 2.0, 3.0])
        copied = ctx.copy(arr)
        np.testing.assert_array_equal(copied, arr)
        assert copied is not arr
    
    def test_context_astype(self):
        """Test dtype conversion."""
        ctx = get_backend_context(backend=Backend.CPU)
        arr = ctx.array([1.0, 2.0, 3.0], dtype=np.float64)
        converted = ctx.astype(arr, np.float32)
        assert converted.dtype == np.float32
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_context_uses_cupy(self):
        """GPU context should use cupy when available."""
        ctx = get_backend_context(backend=Backend.CUDA)
        assert ctx.xp is cp
        assert ctx.backend == Backend.CUDA
        assert ctx.is_gpu is True
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_context_dtype_is_float32_in_mixed_mode(self):
        """GPU context in mixed mode should use float32."""
        ctx = get_backend_context(
            backend=Backend.CUDA,
            precision=PrecisionMode.MIXED
        )
        assert ctx.dtype == np.float32
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_context_dtype_is_float64_in_high_mode(self):
        """GPU context in high precision mode should use float64."""
        ctx = get_backend_context(
            backend=Backend.CUDA,
            precision=PrecisionMode.HIGH
        )
        assert ctx.dtype == np.float64
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_context_array_operations(self):
        """Test GPU array operations."""
        ctx = get_backend_context(backend=Backend.CUDA)
        arr = ctx.ones((3, 4))
        assert arr.shape == (3, 4)
        # Transfer back to CPU for verification
        cpu_arr = ctx.to_cpu(arr)
        assert isinstance(cpu_arr, np.ndarray)
        assert np.all(cpu_arr == 1)
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_context_to_backend_transfer(self):
        """Test transferring numpy array to GPU."""
        ctx = get_backend_context(backend=Backend.CUDA)
        cpu_arr = np.array([1.0, 2.0, 3.0])
        gpu_arr = ctx.to_backend(cpu_arr)
        # Should be a cupy array
        assert hasattr(gpu_arr, 'get')
        # Verify data
        retrieved = ctx.to_cpu(gpu_arr)
        np.testing.assert_array_equal(retrieved, cpu_arr)
    
    def test_create_gpu_context_raises_without_cupy(self):
        """Creating GPU context should raise if CuPy unavailable."""
        if not CUPY_AVAILABLE:
            with pytest.raises(RuntimeError, match="CuPy not available"):
                create_gpu_context()


# ============================================================================
# Assert Close Tests
# ============================================================================

class TestAssertClose:
    """Test numerical comparison with tolerances."""
    
    def test_assert_close_exact_match(self):
        """Exact match should pass."""
        arr = np.array([1.0, 2.0, 3.0])
        assert_close(arr, arr, tolerance=CPU_STANDARD_TOLERANCE)
    
    def test_assert_close_within_tolerance(self):
        """Values within tolerance should pass."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = arr1 + 1e-6  # Within standard tolerance
        assert_close(arr1, arr2, tolerance=CPU_STANDARD_TOLERANCE)
    
    def test_assert_close_exceeds_tolerance(self):
        """Values exceeding tolerance should fail."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = arr1 + 1e-2  # Exceeds standard tolerance
        with pytest.raises(AssertionError):
            assert_close(arr1, arr2, tolerance=CPU_STANDARD_TOLERANCE)
    
    def test_assert_close_shape_mismatch(self):
        """Shape mismatch should raise AssertionError."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0])
        with pytest.raises(AssertionError, match="shape mismatch"):
            assert_close(arr1, arr2, tolerance=CPU_STANDARD_TOLERANCE)
    
    def test_assert_close_nan_equality(self):
        """NaN values should be considered equal."""
        arr1 = np.array([1.0, np.nan, 3.0])
        arr2 = np.array([1.0, np.nan, 3.0])
        assert_close(arr1, arr2, tolerance=CPU_STANDARD_TOLERANCE)
    
    def test_assert_close_integer_exact_equality(self):
        """Integer arrays require exact equality."""
        arr1 = np.array([1, 2, 3], dtype=np.int64)
        arr2 = np.array([1, 2, 3], dtype=np.int64)
        assert_close(arr1, arr2, tolerance=CPU_STANDARD_TOLERANCE)
        
        arr3 = np.array([1, 2, 4], dtype=np.int64)
        with pytest.raises(AssertionError):
            assert_close(arr1, arr3, tolerance=CPU_STANDARD_TOLERANCE)
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_assert_close_gpu_arrays(self):
        """Test assert_close with GPU arrays."""
        import cupy as cp
        arr1 = cp.array([1.0, 2.0, 3.0])
        arr2 = cp.array([1.0, 2.0, 3.0])
        assert_close(arr1, arr2, tolerance=GPU_FLOAT32_TOLERANCE)
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_assert_close_cross_backend(self):
        """Test comparing CPU and GPU arrays."""
        import cupy as cp
        cpu_arr = np.array([1.0, 2.0, 3.0])
        gpu_arr = cp.array([1.0, 2.0, 3.0])
        assert_close(cpu_arr, gpu_arr, tolerance=CROSS_BACKEND_TOLERANCE)


# ============================================================================
# Validate Backend Result Tests
# ============================================================================

class TestValidateBackendResult:
    """Test backend result validation."""
    
    def test_validate_valid_result(self):
        """Valid result should pass validation."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        validate_backend_result(
            arr,
            expected_shape=(2, 2),
            expected_dtype=np.float64,
            finite=True,
            label="test"
        )
    
    def test_validate_shape_mismatch(self):
        """Shape mismatch should raise AssertionError."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(AssertionError, match="shape mismatch"):
            validate_backend_result(
                arr,
                expected_shape=(3, 3),
                label="test"
            )
    
    def test_validate_dtype_mismatch(self):
        """Dtype mismatch should raise AssertionError."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with pytest.raises(AssertionError, match="dtype mismatch"):
            validate_backend_result(
                arr,
                expected_shape=(2, 2),
                expected_dtype=np.float64,
                label="test"
            )
    
    def test_validate_non_finite(self):
        """Non-finite values should raise AssertionError."""
        arr = np.array([[1.0, np.inf], [3.0, 4.0]])
        with pytest.raises(AssertionError, match="non-finite"):
            validate_backend_result(
                arr,
                expected_shape=(2, 2),
                finite=True,
                label="test"
            )
    
    def test_validate_skip_dtype_check(self):
        """Should skip dtype check when expected_dtype is None."""
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        validate_backend_result(
            arr,
            expected_shape=(1, 2),
            expected_dtype=None,
            label="test"
        )
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_validate_gpu_result(self):
        """Test validation of GPU result."""
        import cupy as cp
        gpu_arr = cp.array([[1.0, 2.0], [3.0, 4.0]])
        validate_backend_result(
            gpu_arr,
            expected_shape=(2, 2),
            finite=True,
            label="gpu_test"
        )


# ============================================================================
# Integration Tests
# ============================================================================

class TestBackendIntegration:
    """Integration tests for backend switching and consistency."""
    
    def test_reference_context_for_golden_fixtures(self):
        """Reference context should be suitable for golden fixture generation."""
        ctx = create_reference_context()
        
        # Generate some test data
        data = ctx.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Should be float64
        assert data.dtype == np.float64
        
        # Operations should maintain precision
        result = ctx.xp.sum(data)
        result_cpu = ctx.to_cpu(result)
        
        # Verify with reference tolerance
        expected = 10.0
        assert_close(result_cpu, np.array(expected), 
                    tolerance=CPU_REFERENCE_TOLERANCE)
    
    def test_mixed_precision_workflow(self):
        """Test mixed precision workflow on CPU."""
        # Start with reference precision
        ref_ctx = create_reference_context()
        ref_data = ref_ctx.array([1.0, 2.0, 3.0])
        
        # Switch to mixed precision (still CPU in test environment)
        mixed_ctx = get_backend_context(precision=PrecisionMode.MIXED)
        mixed_data = mixed_ctx.array(ref_data)
        
        # Results should be comparable with appropriate tolerance
        assert_close(
            ref_data, mixed_data,
            tolerance=CPU_STANDARD_TOLERANCE
        )
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_cpu_gpu_parity_with_tolerance(self):
        """Test CPU/GPU results match within cross-backend tolerance."""
        import cupy as cp
        
        # Create test data
        cpu_ctx = create_reference_context()
        gpu_ctx = create_gpu_context(high_precision=True)
        
        # Same computation on both backends
        data_cpu = cpu_ctx.array([[1.0, 2.0], [3.0, 4.0]])
        data_gpu = gpu_ctx.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Compute sum
        sum_cpu = cpu_ctx.to_cpu(cpu_ctx.xp.sum(data_cpu))
        sum_gpu = gpu_ctx.to_cpu(gpu_ctx.xp.sum(data_gpu))
        
        # Should match within cross-backend tolerance
        assert_close(sum_cpu, sum_gpu, tolerance=CROSS_BACKEND_TOLERANCE)
    
    def test_backend_switch_preserves_data(self):
        """Test that switching backends preserves array data."""
        # Create data on CPU
        cpu_ctx = get_backend_context(backend=Backend.CPU)
        original = cpu_ctx.array([[1.0, 2.0], [3.0, 4.0]])
        
        # In a real GPU scenario, would transfer to GPU and back
        # For CPU-only test, just verify round-trip through to_cpu
        transferred = cpu_ctx.to_cpu(original)
        
        assert_close(original, transferred, tolerance=CPU_REFERENCE_TOLERANCE)
    
    def test_multiple_contexts_independent(self):
        """Test that multiple contexts operate independently."""
        ctx1 = get_backend_context(backend=Backend.CPU, precision=PrecisionMode.REFERENCE)
        ctx2 = get_backend_context(backend=Backend.CPU, precision=PrecisionMode.MIXED)
        
        # Different dtypes
        assert ctx1.dtype == np.float64
        assert ctx2.dtype == np.float64  # CPU always float64 in mixed mode
        
        # Arrays from different contexts should be compatible
        arr1 = ctx1.ones((2, 2))
        arr2 = ctx2.ones((2, 2))
        
        assert_close(arr1, arr2, tolerance=CPU_REFERENCE_TOLERANCE)


# ============================================================================
# Edge Cases
# ============================================================================

class TestBackendEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_array_handling(self):
        """Test handling of empty arrays."""
        ctx = get_backend_context(backend=Backend.CPU)
        arr = ctx.zeros((0, 0))
        assert arr.shape == (0, 0)
        assert arr.size == 0
    
    def test_large_array_allocation(self):
        """Test allocation of larger arrays."""
        ctx = get_backend_context(backend=Backend.CPU)
        # Allocate a moderate-sized array
        arr = ctx.zeros((1000, 1000))
        assert arr.shape == (1000, 1000)
        assert arr.dtype == np.float64
    
    def test_complex_dtype(self):
        """Test complex number support."""
        ctx = get_backend_context(backend=Backend.CPU)
        # Explicitly specify complex dtype since default is float64
        arr = ctx.array([1+2j, 3+4j], dtype=np.complex128)
        assert arr.dtype.kind == 'c'
    
    def test_string_conversion_to_float(self):
        """Test that string inputs are properly handled."""
        ctx = get_backend_context(backend=Backend.CPU)
        # NumPy will convert numeric strings
        arr = ctx.array(["1.0", "2.0", "3.0"])
        assert arr.dtype.kind in 'fc'  # float or complex
    
    def test_assert_close_different_dtypes(self):
        """Test assert_close with different but compatible dtypes."""
        arr1 = np.array([1.0, 2.0], dtype=np.float64)
        arr2 = np.array([1.0, 2.0], dtype=np.float32)
        # Should pass with appropriate tolerance
        assert_close(arr1, arr2, tolerance=CPU_STANDARD_TOLERANCE)
