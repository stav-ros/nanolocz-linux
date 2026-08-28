"""
Unit tests for GPU acceleration utilities.
"""

import numpy as np
import pytest
from nanolocz.gpu.utils import (
    get_array_module, 
    to_gpu, 
    from_gpu, 
    GPUArrayModule,
    CUPY_AVAILABLE
)


def test_get_array_module_cpu():
    """Test getting NumPy module for CPU operations."""
    xp = get_array_module(use_gpu=False)
    assert xp is np


def test_get_array_module_gpu_fallback():
    """Test GPU module selection with fallback."""
    # If CuPy is available, should return cupy when use_gpu=True
    # Otherwise should return numpy
    xp = get_array_module(use_gpu=True)
    
    if CUPY_AVAILABLE:
        import cupy as cp
        assert xp is cp
    else:
        assert xp is np


def test_to_from_gpu_cpu():
    """Test GPU transfer functions on CPU path."""
    data = np.random.rand(10, 10)
    
    # Should return array unchanged when GPU not used
    result = to_gpu(data, use_gpu=False)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(data, result)
    
    # from_gpu should also work on CPU arrays
    back = from_gpu(result)
    assert np.array_equal(data, back)


def test_gpu_array_module_basic():
    """Test basic GPUArrayModule functionality."""
    gpu_mod = GPUArrayModule(use_gpu=False)
    
    data = np.random.rand(20, 20)
    
    # Test to_device
    dev_data = gpu_mod.to_device(data)
    assert isinstance(dev_data, np.ndarray)
    
    # Test from_device
    cpu_data = gpu_mod.from_device(dev_data)
    assert isinstance(cpu_data, np.ndarray)
    assert np.array_equal(data, cpu_data)


def test_gpu_array_module_operations():
    """Test GPUArrayModule mathematical operations."""
    gpu_mod = GPUArrayModule(use_gpu=False)
    
    data = np.random.rand(32, 32)
    
    # Test Gaussian filter
    filtered = gpu_mod.gaussian_filter(data, sigma=1.5)
    assert filtered.shape == data.shape
    assert not np.array_equal(data, filtered)  # Should be different
    
    # Test FFT
    fft_result = gpu_mod.fft2(data)
    assert fft_result.shape == data.shape
    
    # Test inverse FFT
    ifft_result = gpu_mod.ifft2(fft_result)
    # Should recover original (within numerical precision)
    assert np.allclose(data, np.real(ifft_result), rtol=1e-5)


def test_cupy_availability_flag():
    """Test that CUPY_AVAILABLE flag works correctly."""
    # Just verify the flag exists and is boolean
    assert isinstance(CUPY_AVAILABLE, bool)
    
    # Print status for debugging
    print(f"\nCuPy available: {CUPY_AVAILABLE}")


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
def test_gpu_actual_transfer():
    """Test actual GPU transfer (only runs if CuPy available)."""
    import cupy as cp
    
    gpu_mod = GPUArrayModule(use_gpu=True)
    data = np.random.rand(100, 100)
    
    # Transfer to GPU
    dev_data = gpu_mod.to_device(data)
    assert isinstance(dev_data, cp.ndarray)
    
    # Transfer back
    cpu_data = gpu_mod.from_device(dev_data)
    assert isinstance(cpu_data, np.ndarray)
    assert np.allclose(data, cpu_data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
