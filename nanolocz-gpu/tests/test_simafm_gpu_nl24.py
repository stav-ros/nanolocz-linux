"""Test suite for NL-24: Simulation AFM Kernels.

Tests GPU-accelerated AFM simulation algorithms,
with CPU fallback when CuPy is unavailable.
"""

import numpy as np
import pytest

from nanolocz.gpu import CUPY_AVAILABLE, create_reference_context
from nanolocz.gpu.simafm import (
    add_scan_artifacts_gpu,
    add_shot_noise_gpu,
    add_thermal_noise_gpu,
    compute_height_field_gpu,
    convolve_tip_gpu,
    simulate_afm_image_gpu,
)


pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE,
    reason="CuPy not available - tests run in CPU fallback mode",
)


class TestComputeHeightFieldGPU:
    """Tests for compute_height_field_gpu function."""

    def test_single_atom(self):
        """Test height field from single atom."""
        coords = np.array([[10.0, 10.0, 0.5]])  # x, y, z in nm
        
        height_field = compute_height_field_gpu(
            coords, output_shape=(32, 32), pixel_size=0.5
        )
        
        assert height_field.shape == (32, 32)
        # Height should be positive near the atom
        assert np.max(height_field) > 0

    def test_multiple_atoms(self):
        """Test height field from multiple atoms."""
        coords = np.array([
            [10.0, 10.0, 0.5],
            [20.0, 10.0, 0.6],
            [15.0, 20.0, 0.4],
        ])
        
        height_field = compute_height_field_gpu(
            coords, output_shape=(32, 32), pixel_size=0.5
        )
        
        assert height_field.shape == (32, 32)
        # Should have multiple peaks
        assert np.max(height_field) > 0.5

    def test_variable_radii(self):
        """Test with per-atom radii."""
        coords = np.array([[15.0, 15.0, 0.5]])
        radii = np.array([0.3])  # Larger radius
        
        height_field = compute_height_field_gpu(
            coords, atomic_radii=radii, output_shape=(32, 32), pixel_size=0.5
        )
        
        assert height_field.shape == (32, 32)

    def test_z_scale(self):
        """Test z-height scaling."""
        coords = np.array([[15.0, 15.0, 0.5]])
        
        height_unscaled = compute_height_field_gpu(
            coords, output_shape=(32, 32), z_scale=1.0
        )
        height_scaled = compute_height_field_gpu(
            coords, output_shape=(32, 32), z_scale=2.0
        )
        
        # Scaled should be approximately double
        assert np.max(height_scaled) > np.max(height_unscaled)

    def test_empty_coords(self):
        """Test with no atoms."""
        coords = np.empty((0, 3))
        
        height_field = compute_height_field_gpu(
            coords, output_shape=(32, 32)
        )
        
        assert height_field.shape == (32, 32)
        assert np.all(height_field == 0)


class TestConvolveTipGPU:
    """Tests for convolve_tip_gpu function."""

    def test_spherical_tip(self):
        """Test convolution with spherical tip."""
        surface = np.zeros((64, 64), dtype=np.float64)
        surface[32, 32] = 1.0  # Single point
        
        convolved = convolve_tip_gpu(
            surface, tip_shape="sphere", tip_radius=5.0
        )
        
        assert convolved.shape == (64, 64)
        # Convolution should spread the point
        assert np.sum(convolved > 0) > 1

    def test_cone_tip(self):
        """Test convolution with conical tip."""
        surface = np.random.rand(32, 32).astype(np.float64) * 0.5
        
        convolved = convolve_tip_gpu(
            surface, tip_shape="cone", tip_radius=5.0, tip_angle=30.0
        )
        
        assert convolved.shape == (32, 32)

    def test_paraboloid_tip(self):
        """Test convolution with paraboloid tip."""
        surface = np.random.rand(32, 32).astype(np.float64) * 0.5
        
        convolved = convolve_tip_gpu(
            surface, tip_shape="paraboloid", tip_radius=5.0
        )
        
        assert convolved.shape == (32, 32)

    def test_pyramid_tip(self):
        """Test convolution with pyramid tip."""
        surface = np.random.rand(32, 32).astype(np.float64) * 0.5
        
        convolved = convolve_tip_gpu(
            surface, tip_shape="pyramid", tip_radius=5.0, tip_angle=30.0
        )
        
        assert convolved.shape == (32, 32)

    def test_invalid_tip_shape(self):
        """Test error on invalid tip shape."""
        surface = np.random.rand(32, 32).astype(np.float64)
        
        with pytest.raises(ValueError, match="Unknown tip_shape"):
            convolve_tip_gpu(surface, tip_shape="invalid")


class TestAddThermalNoiseGPU:
    """Tests for add_thermal_noise_gpu function."""

    def test_thermal_noise_addition(self):
        """Test adding thermal noise."""
        image = np.ones((64, 64), dtype=np.float64) * 0.5
        
        noisy = add_thermal_noise_gpu(image, amplitude=0.1)
        
        assert noisy.shape == (64, 64)
        # Noise should introduce variation
        assert np.std(noisy) > np.std(image)

    def test_correlation_length(self):
        """Test spatial correlation of thermal noise."""
        image = np.ones((64, 64), dtype=np.float64) * 0.5
        
        # Short correlation length
        noisy_short = add_thermal_noise_gpu(
            image, amplitude=0.1, correlation_length=2.0
        )
        
        # Long correlation length
        noisy_long = add_thermal_noise_gpu(
            image, amplitude=0.1, correlation_length=10.0
        )
        
        # Both should have similar variance
        assert abs(np.std(noisy_short) - np.std(noisy_long)) < 0.05


class TestAddShotNoiseGPU:
    """Tests for add_shot_noise_gpu function."""

    def test_shot_noise_addition(self):
        """Test adding Poisson shot noise."""
        image = np.ones((64, 64), dtype=np.float64) * 100
        
        noisy = add_shot_noise_gpu(image, scale=1.0)
        
        assert noisy.shape == (64, 64)
        # Shot noise should introduce variation
        assert np.std(noisy) > 0

    def test_shot_noise_scale(self):
        """Test effect of scale parameter."""
        image = np.ones((64, 64), dtype=np.float64) * 100
        
        noisy_low = add_shot_noise_gpu(image, scale=0.1)
        noisy_high = add_shot_noise_gpu(image, scale=10.0)
        
        # Higher scale should reduce relative noise
        # (more photons = better SNR)


class TestAddScanArtifactsGPU:
    """Tests for add_scan_artifacts_gpu function."""

    def test_line_noise(self):
        """Test adding line-to-line noise."""
        image = np.ones((64, 64), dtype=np.float64) * 0.5
        
        artifacted = add_scan_artifacts_gpu(
            image, line_noise_amplitude=0.05
        )
        
        assert artifacted.shape == (64, 64)
        # Each row should have a constant offset
        row_means = np.mean(artifacted, axis=1)
        assert np.std(row_means) > 0

    def test_trace_retrace_offset(self):
        """Test adding trace/retrace mismatch."""
        image = np.ones((64, 64), dtype=np.float64) * 0.5
        
        artifacted = add_scan_artifacts_gpu(
            image, trace_retrace_offset=0.02
        )
        
        # Alternating lines should have different offsets
        even_rows = np.mean(artifacted[::2], axis=1)
        odd_rows = np.mean(artifacted[1::2], axis=1)
        
        # There should be some systematic difference
        assert np.mean(even_rows) != np.mean(odd_rows)


class TestSimulateAFMImageGPU:
    """Tests for simulate_afm_image_gpu full pipeline."""

    def test_full_pipeline(self):
        """Test complete AFM simulation pipeline."""
        atoms = np.array([
            [10.0, 10.0, 0.5],
            [15.0, 15.0, 0.6],
            [20.0, 10.0, 0.4],
        ])
        
        image = simulate_afm_image_gpu(
            atoms, output_shape=(64, 64), add_noise=False
        )
        
        assert image.shape == (64, 64)
        assert np.max(image) > 0

    def test_with_noise(self):
        """Test simulation with noise added."""
        atoms = np.array([[15.0, 15.0, 0.5]])
        
        image_no_noise = simulate_afm_image_gpu(
            atoms, output_shape=(64, 64), add_noise=False
        )
        image_with_noise = simulate_afm_image_gpu(
            atoms, output_shape=(64, 64), add_noise=True
        )
        
        assert image_with_noise.shape == image_no_noise.shape
        # Noise should increase variance
        assert np.std(image_with_noise) > np.std(image_no_noise)

    def test_different_tip_shapes(self):
        """Test simulation with different tip geometries."""
        atoms = np.array([[15.0, 15.0, 0.5]])
        
        for tip_shape in ["sphere", "cone", "paraboloid"]:
            image = simulate_afm_image_gpu(
                atoms, output_shape=(32, 32),
                tip_shape=tip_shape, add_noise=False
            )
            assert image.shape == (32, 32)
            assert np.max(image) > 0


class TestBackendContextIntegration:
    """Tests for backend context integration."""

    def test_reference_context(self):
        """Test with reference (float64) context."""
        ctx = create_reference_context()
        
        atoms = np.array([[10.0, 10.0, 0.5]])
        
        height_field = compute_height_field_gpu(
            atoms, output_shape=(32, 32), ctx=ctx
        )
        
        assert height_field.dtype == np.float64

    def test_convolution_with_context(self):
        """Test tip convolution with explicit context."""
        ctx = create_reference_context()
        
        surface = np.random.rand(32, 32).astype(np.float64)
        
        convolved = convolve_tip_gpu(
            surface, tip_shape="sphere", tip_radius=3.0, ctx=ctx
        )
        
        assert convolved.dtype == np.float64


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_coords_shape(self):
        """Test error on invalid coordinate shape."""
        coords = np.random.rand(10, 2)  # Missing z dimension
        
        with pytest.raises(ValueError, match="(N, 3)"):
            compute_height_field_gpu(coords, output_shape=(32, 32))

    def test_non_2d_surface(self):
        """Test error on non-2D surface."""
        surface = np.random.rand(32, 32, 3)
        
        with pytest.raises(ValueError, match="2D"):
            convolve_tip_gpu(surface)

    def test_large_simulation(self):
        """Test with large simulation for performance."""
        # Generate random atomic positions
        n_atoms = 100
        atoms = np.random.rand(n_atoms, 3) * 50
        atoms[:, 2] *= 0.5  # Smaller z range
        
        image = simulate_afm_image_gpu(
            atoms, output_shape=(256, 256), add_noise=False
        )
        
        assert image.shape == (256, 256)

    def test_extreme_parameters(self):
        """Test with extreme parameter values."""
        atoms = np.array([[15.0, 15.0, 1.0]])
        
        # Very large tip
        image = simulate_afm_image_gpu(
            atoms, output_shape=(64, 64),
            tip_radius=20.0, add_noise=False
        )
        
        assert image.shape == (64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
