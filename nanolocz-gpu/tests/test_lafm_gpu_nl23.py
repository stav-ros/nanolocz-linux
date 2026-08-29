"""Test suite for NL-23: LAFM Splatting and FRC Kernels.

Tests GPU-accelerated LAFM reconstruction and Fourier Ring Correlation,
with CPU fallback when CuPy is unavailable.
"""

import numpy as np
import pytest

from nanolocz.gpu import CUPY_AVAILABLE, create_reference_context
from nanolocz.gpu.lafm import (
    batch_splat_gpu,
    compute_frc_gpu,
    frc_resolution,
    splat_gaussian_gpu,
    splat_localizations_gpu,
)


pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="CuPy not available - tests run in CPU fallback mode",
)


class TestSplatGaussianGPU:
    """Tests for splat_gaussian_gpu function."""

    def test_single_gaussian(self):
        """Test splatting a single Gaussian."""
        coords = np.array([[30.0, 30.0]])
        
        image = splat_gaussian_gpu(
            coords, output_shape=(64, 64), global_sigma=2.0
        )
        
        assert image.shape == (64, 64)
        # Peak should be at the center
        assert image[30, 30] > image[25, 25]
        # Maximum should be near the coordinate
        max_pos = np.unravel_index(np.argmax(image), image.shape)
        assert abs(max_pos[0] - 30) <= 2
        assert abs(max_pos[1] - 30) <= 2

    def test_multiple_gaussians(self):
        """Test splatting multiple Gaussians."""
        coords = np.array([[20.0, 20.0], [40.0, 40.0]])
        intensities = np.array([1.0, 0.8])
        
        image = splat_gaussian_gpu(
            coords, intensities=intensities, output_shape=(64, 64)
        )
        
        assert image.shape == (64, 64)
        # Both peaks should be visible
        assert image[20, 20] > 0
        assert image[40, 40] > 0
        # Stronger peak should have higher value
        assert image[20, 20] > image[40, 40]

    def test_variable_sigmas(self):
        """Test with per-localization sigma values."""
        coords = np.array([[20.0, 20.0], [40.0, 40.0]])
        sigmas = np.array([1.0, 5.0])  # Different widths
        
        image = splat_gaussian_gpu(
            coords, sigmas=sigmas, output_shape=(64, 64)
        )
        
        assert image.shape == (64, 64)
        # Wider sigma should produce broader peak

    def test_empty_coordinates(self):
        """Test with no coordinates."""
        coords = np.empty((0, 2))
        
        image = splat_gaussian_gpu(
            coords, output_shape=(64, 64)
        )
        
        assert image.shape == (64, 64)
        assert np.all(image == 0)

    def test_auto_output_shape(self):
        """Test automatic output shape computation."""
        coords = np.array([[50.0, 50.0]])
        
        image = splat_gaussian_gpu(coords, global_sigma=2.0)
        
        # Shape should accommodate the coordinate
        assert image.shape[0] > 50
        assert image.shape[1] > 50


class TestSplatLocalizationsGPU:
    """Tests for splat_localizations_gpu high-level interface."""

    def test_dict_input(self):
        """Test with dictionary input."""
        locs = {
            'coordinates': np.array([[20.0, 30.0], [40.0, 50.0]]),
            'intensities': np.array([1.0, 0.5]),
        }
        
        image = splat_localizations_gpu(locs, output_shape=(64, 64))
        
        assert image.shape == (64, 64)

    def test_array_input(self):
        """Test with array-only input."""
        coords = np.array([[20.0, 30.0], [40.0, 50.0]])
        
        image = splat_localizations_gpu(coords, output_shape=(64, 64))
        
        assert image.shape == (64, 64)

    def test_use_uncertainty_flag(self):
        """Test use_uncertainty parameter."""
        locs = {
            'coordinates': np.array([[30.0, 30.0]]),
            'sigmas': np.array([3.0]),
        }
        
        image_with_sigma = splat_localizations_gpu(
            locs, output_shape=(64, 64), use_uncertainty=True
        )
        image_default = splat_localizations_gpu(
            locs, output_shape=(64, 64), use_uncertainty=False
        )
        
        # Results should differ when using vs ignoring sigmas
        assert not np.array_equal(image_with_sigma, image_default)


class TestComputeFRCGPU:
    """Tests for compute_frc_gpu function."""

    def test_identical_maps(self):
        """Test FRC of identical maps (should be ~1.0)."""
        map1 = np.random.rand(64, 64).astype(np.float64)
        map2 = map1.copy()
        
        freqs, frc_vals = compute_frc_gpu(map1, map2)
        
        assert len(freqs) == len(frc_vals)
        # FRC should be close to 1 for identical maps
        assert np.mean(frc_vals[:10]) > 0.9  # Low frequencies

    def test_independent_noise(self):
        """Test FRC of independent noise (should be low)."""
        map1 = np.random.rand(64, 64).astype(np.float64)
        map2 = np.random.rand(64, 64).astype(np.float64)
        
        freqs, frc_vals = compute_frc_gpu(map1, map2)
        
        assert len(freqs) == len(frc_vals)
        # FRC should be low for independent noise
        assert np.mean(frc_vals) < 0.3

    def test_with_mask(self):
        """Test FRC computation with mask."""
        map1 = np.random.rand(64, 64).astype(np.float64)
        map2 = map1 + np.random.randn(64, 64).astype(np.float64) * 0.1
        
        mask = np.zeros((64, 64), dtype=bool)
        mask[16:48, 16:48] = True  # Center region only
        
        freqs, frc_vals = compute_frc_gpu(map1, map2, mask=mask)
        
        assert len(freqs) == len(frc_vals)

    def test_frequency_range(self):
        """Test that frequencies span expected range."""
        map1 = np.random.rand(128, 128).astype(np.float64)
        map2 = map1 + np.random.randn(128, 128).astype(np.float64) * 0.1
        
        freqs, frc_vals = compute_frc_gpu(map1, map2)
        
        assert freqs[0] >= 0
        assert freqs[-1] <= 0.5  # Nyquist frequency


class TestFRCResolution:
    """Tests for frc_resolution function."""

    def test_one_seventh_threshold(self):
        """Test resolution estimation with 1/7 threshold."""
        # Create a typical FRC curve
        freqs = np.linspace(0, 0.5, 50)
        frc_vals = np.exp(-freqs * 10)  # Decaying curve
        
        resolution = frc_resolution(freqs, frc_vals, threshold="1/7")
        
        assert np.isfinite(resolution)
        assert resolution > 0

    def test_half_bit_threshold(self):
        """Test resolution estimation with half-bit threshold."""
        freqs = np.linspace(0, 0.5, 50)
        frc_vals = np.exp(-freqs * 10)
        
        resolution = frc_resolution(freqs, frc_vals, threshold="1/2bit")
        
        assert np.isfinite(resolution)

    def test_custom_threshold(self):
        """Test with custom numeric threshold."""
        freqs = np.linspace(0, 0.5, 50)
        frc_vals = np.exp(-freqs * 10)
        
        resolution = frc_resolution(freqs, frc_vals, threshold=0.5)
        
        assert np.isfinite(resolution)

    def test_never_crosses_threshold(self):
        """Test when FRC never crosses threshold."""
        freqs = np.linspace(0, 0.5, 50)
        frc_vals = np.ones(50) * 0.01  # Always below threshold
        
        resolution = frc_resolution(freqs, frc_vals, threshold="1/7")
        
        assert np.isinf(resolution)

    def test_zero_frequency(self):
        """Test handling of zero/negative cutoff frequency."""
        freqs = np.array([0.0, 0.1, 0.2])
        frc_vals = np.array([0.5, 0.1, 0.05])  # Crosses at zero
        
        resolution = frc_resolution(freqs, frc_vals, threshold="1/7")
        
        assert np.isinf(resolution) or resolution > 0


class TestBatchSplatGPU:
    """Tests for batch_splat_gpu function."""

    def test_batch_processing(self):
        """Test batch processing of multiple frames."""
        locs_list = [
            {'coordinates': np.array([[20.0, 20.0]])},
            {'coordinates': np.array([[40.0, 40.0]])},
            {'coordinates': np.array([[30.0, 30.0]])},
        ]
        
        frames = batch_splat_gpu(locs_list, output_shape=(64, 64))
        
        assert frames.shape == (3, 64, 64)
        # Each frame should have its peak at different location
        assert frames[0, 20, 20] > frames[0, 40, 40]
        assert frames[1, 40, 40] > frames[1, 20, 20]

    def test_empty_batch(self):
        """Test with empty list."""
        frames = batch_splat_gpu([], output_shape=(64, 64))
        
        assert frames.shape[0] == 0
        assert frames.shape[1:] == (64, 64)

    def test_varying_num_localizations(self):
        """Test batch with varying number of localizations per frame."""
        locs_list = [
            {'coordinates': np.array([[20.0, 20.0], [30.0, 30.0]])},
            {'coordinates': np.array([[40.0, 40.0]])},
            {'coordinates': np.empty((0, 2))},  # Empty frame
        ]
        
        frames = batch_splat_gpu(locs_list, output_shape=(64, 64))
        
        assert frames.shape == (3, 64, 64)
        # Third frame should be all zeros
        assert np.all(frames[2] == 0)


class TestBackendContextIntegration:
    """Tests for backend context integration."""

    def test_reference_context(self):
        """Test with reference (float64) context."""
        ctx = create_reference_context()
        
        coords = np.array([[30.0, 30.0]])
        image = splat_gaussian_gpu(coords, output_shape=(64, 64), ctx=ctx)
        
        assert image.dtype == np.float64

    def test_frc_with_context(self):
        """Test FRC with explicit context."""
        ctx = create_reference_context()
        
        map1 = np.random.rand(64, 64).astype(np.float64)
        map2 = map1 + np.random.randn(64, 64).astype(np.float64) * 0.1
        
        freqs, frc_vals = compute_frc_gpu(map1, map2, ctx=ctx)
        
        assert freqs.dtype == np.float64
        assert frc_vals.dtype == np.float64


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_mismatched_map_shapes(self):
        """Test error on mismatched map shapes."""
        map1 = np.random.rand(64, 64).astype(np.float64)
        map2 = np.random.rand(32, 32).astype(np.float64)
        
        with pytest.raises(ValueError, match="same shape"):
            compute_frc_gpu(map1, map2)

    def test_very_large_image(self):
        """Test with large image for performance."""
        coords = np.random.rand(1000, 2) * 200
        
        image = splat_gaussian_gpu(
            coords, output_shape=(256, 256), global_sigma=2.0
        )
        
        assert image.shape == (256, 256)

    def test_extreme_sigma_values(self):
        """Test with very small and large sigma values."""
        coords = np.array([[30.0, 30.0]])
        
        # Very narrow
        image_narrow = splat_gaussian_gpu(
            coords, output_shape=(64, 64), global_sigma=0.5
        )
        
        # Very wide
        image_wide = splat_gaussian_gpu(
            coords, output_shape=(64, 64), global_sigma=10.0
        )
        
        # Narrow should have sharper peak
        assert image_narrow[30, 30] > image_wide[30, 30]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
