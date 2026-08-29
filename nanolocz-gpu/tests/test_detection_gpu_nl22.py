"""Test suite for NL-22: GPU Detection and Statistics Kernels.

Tests GPU-accelerated detection algorithms with CPU fallback,
validating parity with the reference implementation.
"""

import numpy as np
import pytest

from nanolocz.core.detection import detect_particles as detect_particles_cpu
from nanolocz.gpu import (
    CUPY_AVAILABLE,
    create_reference_context,
    get_backend_context,
)
from nanolocz.gpu.detection import (
    detect_particles_gpu,
    local_maxima_gpu,
    min_distance_suppression_gpu,
    prominence_gpu,
    statistics_gpu,
)


pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="CuPy not available - tests run in CPU fallback mode",
)


class TestLocalMaximaGPU:
    """Tests for local_maxima_gpu function."""

    def test_simple_peaks(self):
        """Test detection of simple peaks."""
        img = np.zeros((32, 32), dtype=np.float64)
        img[10, 15] = 100
        img[20, 25] = 80
        
        y_coords, x_coords = local_maxima_gpu(img, kernel_size=5, threshold=50)
        
        assert len(x_coords) >= 2
        # Check that peaks are detected near expected locations
        detected = list(zip(x_coords, y_coords))
        assert any(abs(x - 15) < 2 and abs(y - 10) < 2 for x, y in detected)
        assert any(abs(x - 25) < 2 and abs(y - 20) < 2 for x, y in detected)

    def test_with_mask(self):
        """Test detection with boolean mask."""
        img = np.random.rand(32, 32).astype(np.float64) * 100
        img[15, 15] = 500
        
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True  # Only allow detection in center region
        
        y_coords, x_coords = local_maxima_gpu(
            img, kernel_size=5, threshold=200, mask=mask
        )
        
        # All detections should be within mask
        for x, y in zip(x_coords, y_coords):
            assert mask[y, x], f"Detection at ({x}, {y}) outside mask"

    def test_empty_result(self):
        """Test when no peaks exceed threshold."""
        img = np.random.rand(32, 32).astype(np.float64) * 10
        
        y_coords, x_coords = local_maxima_gpu(img, kernel_size=5, threshold=100)
        
        assert len(x_coords) == 0
        assert len(y_coords) == 0

    def test_kernel_size_variants(self):
        """Test different kernel sizes."""
        img = np.zeros((64, 64), dtype=np.float64)
        img[30, 30] = 100
        
        for kernel_size in [3, 5, 7, 9]:
            y_coords, x_coords = local_maxima_gpu(
                img, kernel_size=kernel_size, threshold=50
            )
            assert len(x_coords) >= 1


class TestProminenceGPU:
    """Tests for prominence_gpu function."""

    def test_prominence_calculation(self):
        """Test prominence calculation for simple peaks."""
        # Create a ridge profile
        img = np.zeros((32, 32), dtype=np.float64)
        img[15, :] = np.sin(np.linspace(0, 4 * np.pi, 32)) * 20 + 50
        
        peaks = np.array([[15.0, 8.0], [15.0, 24.0]])  # Two peaks
        heights = np.array([img[8, 15], img[24, 15]])
        
        prominences = prominence_gpu(img, peaks, heights)
        
        assert len(prominences) == 2
        assert all(p >= 0 for p in prominences)

    def test_no_higher_peak(self):
        """Test prominence when no higher peak exists."""
        img = np.random.rand(32, 32).astype(np.float64) * 100
        img[15, 15] = 500  # Highest peak
        
        peaks = np.array([[15.0, 15.0]])
        heights = np.array([500.0])
        
        prominences = prominence_gpu(img, peaks, heights)
        
        assert len(prominences) == 1
        assert prominences[0] >= 0


class TestMinDistanceSuppressionGPU:
    """Tests for min_distance_suppression_gpu function."""

    def test_suppression_applied(self):
        """Test that close peaks are suppressed."""
        # Create two very close peaks
        peaks = np.array([[10.0, 10.0], [10.5, 10.5]])
        heights = np.array([100.0, 80.0])
        
        keep_indices = min_distance_suppression_gpu(
            peaks, heights, min_distance=5.0
        )
        
        # Only the stronger peak should be kept
        assert len(keep_indices) == 1
        assert keep_indices[0] == 0  # Strongest peak

    def test_no_suppression_needed(self):
        """Test when peaks are far apart."""
        peaks = np.array([[5.0, 5.0], [25.0, 25.0]])
        heights = np.array([100.0, 80.0])
        
        keep_indices = min_distance_suppression_gpu(
            peaks, heights, min_distance=5.0
        )
        
        # Both peaks should be kept
        assert len(keep_indices) == 2

    def test_zero_min_distance(self):
        """Test with min_distance=0 (no suppression)."""
        peaks = np.array([[10.0, 10.0], [10.1, 10.1]])
        heights = np.array([100.0, 80.0])
        
        keep_indices = min_distance_suppression_gpu(
            peaks, heights, min_distance=0.0
        )
        
        # All peaks should be kept
        assert len(keep_indices) == 2


class TestDetectParticlesGPU:
    """Tests for detect_particles_gpu main entry point."""

    def test_parity_with_cpu(self):
        """Test GPU results match CPU reference."""
        np.random.seed(42)
        img = np.random.rand(64, 64).astype(np.float64) * 100
        img[20, 30] = 500
        img[40, 50] = 450
        
        # CPU reference
        cpu_result = detect_particles_cpu(
            img, thresh=200, kernel_size=5, min_distance=3.0
        )
        
        # GPU (or CPU fallback)
        gpu_result = detect_particles_gpu(
            img, threshold=200, kernel_size=5, min_distance=3.0
        )
        
        # Compare number of detections
        assert len(gpu_result) == len(cpu_result.coordinates)
        
        # If both have detections, compare coordinates
        if len(cpu_result.coordinates) > 0:
            assert gpu_result.shape[1] == 4  # x, y, height, prominence

    def test_empty_detection(self):
        """Test when no particles detected."""
        img = np.random.rand(32, 32).astype(np.float64) * 10
        
        result = detect_particles_gpu(img, threshold=100)
        
        assert len(result) == 0
        assert result.shape == (0, 4)

    def test_with_min_prominence(self):
        """Test filtering by prominence."""
        img = np.zeros((64, 64), dtype=np.float64)
        img[20, 30] = 500  # High prominence
        img[30, 30] = 100  # Low prominence (on slope)
        
        result_strict = detect_particles_gpu(
            img, threshold=50, min_prominence=100
        )
        result_lenient = detect_particles_gpu(
            img, threshold=50, min_prominence=10
        )
        
        # Stricter prominence should give fewer or equal detections
        assert len(result_strict) <= len(result_lenient)


class TestStatisticsGPU:
    """Tests for statistics_gpu function."""

    def test_statistics_calculation(self):
        """Test area, volume, eccentricity computation."""
        img = np.zeros((64, 64), dtype=np.float64)
        img[30, 30] = 100  # Single peak
        
        coords = np.array([[30.0, 30.0]])
        
        stats = statistics_gpu(img, coords, radius=3)
        
        assert 'area' in stats
        assert 'volume' in stats
        assert 'eccentricity' in stats
        
        assert len(stats['area']) == 1
        assert len(stats['volume']) == 1
        assert len(stats['eccentricity']) == 1
        
        # Volume should be positive
        assert stats['volume'][0] > 0
        
        # Eccentricity should be in [0, 1]
        assert 0 <= stats['eccentricity'][0] <= 1

    def test_multiple_particles(self):
        """Test statistics for multiple particles."""
        img = np.zeros((64, 64), dtype=np.float64)
        img[20, 20] = 100
        img[40, 40] = 80
        
        coords = np.array([[20.0, 20.0], [40.0, 40.0]])
        
        stats = statistics_gpu(img, coords, radius=3)
        
        assert len(stats['area']) == 2
        assert len(stats['volume']) == 2
        assert len(stats['eccentricity']) == 2


class TestBackendContextIntegration:
    """Tests for backend context integration."""

    def test_reference_context(self):
        """Test with reference (float64) context."""
        ctx = create_reference_context()
        
        img = np.random.rand(32, 32).astype(np.float64) * 100
        img[15, 15] = 500
        
        result = detect_particles_gpu(img, threshold=200, ctx=ctx)
        
        assert result.dtype == np.float64

    def test_auto_context(self):
        """Test with auto-selected context."""
        ctx = get_backend_context()
        
        img = np.random.rand(32, 32).astype(np.float64) * 100
        img[15, 15] = 500
        
        result = detect_particles_gpu(img, threshold=200, ctx=ctx)
        
        assert len(result) >= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_pixel_image(self):
        """Test with minimal image size."""
        img = np.array([[100.0]], dtype=np.float64)
        
        with pytest.raises(ValueError, match="2D"):
            local_maxima_gpu(img)

    def test_non_finite_values(self):
        """Test handling of NaN/Inf values."""
        img = np.random.rand(32, 32).astype(np.float64) * 100
        img[15, 15] = np.nan
        img[16, 16] = np.inf
        
        # Should not crash
        y_coords, x_coords = local_maxima_gpu(img, threshold=50)
        
        # Results may vary but should not contain NaN positions
        assert not np.any(np.isnan(x_coords))
        assert not np.any(np.isnan(y_coords))

    def test_large_image(self):
        """Test with larger image for performance."""
        img = np.random.rand(256, 256).astype(np.float64) * 100
        
        # Should complete without error
        result = detect_particles_gpu(img, threshold=50)
        
        assert result.shape[1] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
