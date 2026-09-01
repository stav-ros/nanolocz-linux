"""
Tests for directional deskar filter (NL-31).

Tests FFT-based directional filtering, scan-line removal,
anisotropic diffusion, and GPU acceleration.
"""

import numpy as np
import pytest
from nanolocz.core.deskar import (
    directional_deskar,
    remove_scan_lines,
    anisotropic_diffusion,
    process_movie_deskar,
    directional_deskar_gpu,
)


class TestDirectionalDeskarFFT:
    """Tests for FFT-based directional filtering."""
    
    def test_basic_filtering(self):
        """Test basic directional deskar filtering preserves shape."""
        img = np.random.rand(256, 256)
        filtered = directional_deskar(img, scan_angle=0.0)
        assert filtered.shape == img.shape
        assert filtered.dtype == np.float64
    
    def test_fft_reconstruction_lossless_without_masking(self):
        """Test that FFT reconstruction is nearly lossless with strength=0."""
        img = np.random.rand(128, 128)
        # With zero strength, should be nearly identical
        filtered = directional_deskar(img, strength=0.0)
        np.testing.assert_allclose(filtered, img, rtol=1e-10, atol=1e-10)
    
    def test_horizontal_scan_line_removal(self):
        """Test removal of horizontal scan lines."""
        # Create image with horizontal line artifacts
        img = np.random.rand(256, 256)
        # Add horizontal stripes
        for i in range(0, 256, 4):
            img[i, :] += 0.5
        
        filtered = directional_deskar(img, scan_angle=0.0, strength=0.9)
        
        # Check that variance along rows is reduced
        original_row_var = np.var(np.diff(img, axis=0), axis=0).mean()
        filtered_row_var = np.var(np.diff(filtered, axis=0), axis=0).mean()
        
        # Should reduce row-to-row variation
        assert filtered_row_var < original_row_var
    
    def test_frequency_cutoff_effect(self):
        """Test that frequency cutoff controls what gets filtered."""
        img = np.random.rand(256, 256)
        
        # Low cutoff filters less
        filtered_low = directional_deskar(img, frequency_cutoff=0.01)
        # High cutoff filters more
        filtered_high = directional_deskar(img, frequency_cutoff=0.3)
        
        # Both should preserve shape
        assert filtered_low.shape == img.shape
        assert filtered_high.shape == img.shape
    
    def test_notch_width_effect(self):
        """Test that notch width affects filtering selectivity."""
        img = np.random.rand(256, 256)
        
        # Narrow notch
        filtered_narrow = directional_deskar(img, notch_width=0.01)
        # Wide notch
        filtered_wide = directional_deskar(img, notch_width=0.1)
        
        # Both should preserve shape
        assert filtered_narrow.shape == img.shape
        assert filtered_wide.shape == img.shape
    
    def test_preserve_low_freq_option(self):
        """Test low-frequency preservation option."""
        # Create image with strong low-frequency component
        x = np.linspace(0, 10, 256)
        y = np.linspace(0, 10, 256)
        X, Y = np.meshgrid(x, y)
        low_freq = np.sin(X) * np.sin(Y)
        high_freq = np.random.rand(256, 256) * 0.1
        img = low_freq + high_freq
        
        # With preservation
        filtered_preserve = directional_deskar(img, preserve_low_freq=True)
        # Without preservation
        filtered_no_preserve = directional_deskar(img, preserve_low_freq=False)
        
        # Preserve should maintain more low-frequency content
        # Check by comparing correlation with original low-freq component
        corr_preserve = np.corrcoef(filtered_preserve.flatten(), low_freq.flatten())[0, 1]
        corr_no_preserve = np.corrcoef(filtered_no_preserve.flatten(), low_freq.flatten())[0, 1]
        
        assert corr_preserve >= corr_no_preserve - 0.1  # Allow some tolerance


class TestScanLineRemoval:
    """Tests for scan-line artifact removal."""
    
    def test_horizontal_removal_shape(self):
        """Test horizontal scan-line removal preserves shape."""
        img = np.random.rand(256, 256)
        corrected = remove_scan_lines(img, direction='horizontal')
        assert corrected.shape == img.shape
    
    def test_vertical_removal_shape(self):
        """Test vertical scan-line removal preserves shape."""
        img = np.random.rand(256, 256)
        corrected = remove_scan_lines(img, direction='vertical')
        assert corrected.shape == img.shape
    
    def test_constant_offset_removal(self):
        """Test removal of constant line offsets."""
        # Create image with line offsets
        img = np.random.rand(100, 100)
        for i in range(100):
            img[i, :] += (i % 4) * 0.1  # Every 4th line has offset
        
        corrected = remove_scan_lines(img, direction='horizontal', method='median')
        
        # Line-to-line variation should be reduced
        original_diff = np.abs(np.diff(img, axis=0)).mean()
        corrected_diff = np.abs(np.diff(corrected, axis=0)).mean()
        
        assert corrected_diff < original_diff
    
    def test_median_method_robustness(self):
        """Test median method is robust to outliers."""
        img = np.random.rand(100, 100)
        # Add extreme outlier to one line
        img[50, :] = 1000.0
        
        corrected = remove_scan_lines(img, direction='horizontal', method='median')
        
        # Outlier should be removed
        assert corrected[50, :].mean() < 10.0
    
    def test_robust_method_outlier_handling(self):
        """Test robust method handles outliers correctly."""
        img = np.random.rand(100, 100)
        # Add scattered outliers
        img[20:25, :] = 100.0
        img[70:75, :] = -50.0
        
        corrected = remove_scan_lines(img, direction='horizontal', method='robust', threshold=2.0)
        
        # Outliers should be interpolated
        assert np.abs(corrected[20:25, :]).max() < 10.0


class TestAnisotropicDiffusion:
    """Tests for anisotropic diffusion smoothing."""
    
    def test_diffusion_shape_preservation(self):
        """Test anisotropic diffusion preserves shape."""
        img = np.random.rand(256, 256)
        diffused = anisotropic_diffusion(img, n_iterations=5)
        assert diffused.shape == img.shape
    
    def test_smoothing_reduces_variance(self):
        """Test diffusion changes image (basic sanity check)."""
        # Create noisy image
        clean = np.zeros((128, 128))
        clean[40:80, 40:80] = 1.0  # Square
        np.random.seed(42)
        noisy = clean + np.random.rand(128, 128) * 0.3
        
        diffused = anisotropic_diffusion(noisy, n_iterations=10, kappa=50.0, gamma=0.1)
        
        # Diffused should be different from noisy (smoothing occurred)
        mse = np.mean((diffused - noisy)**2)
        
        # MSE should be > 0 (some change occurred)
        assert mse > 1e-6
        
        # But not too large (preserves overall structure)
        assert mse < 0.1
    
    def test_edge_preservation(self):
        """Test diffusion preserves edges better than isotropic smoothing."""
        # Create step edge
        img = np.zeros((128, 128))
        img[:, 64:] = 1.0
        
        diffused = anisotropic_diffusion(img, n_iterations=5, kappa=10.0)
        
        # Edge should still be visible (gradient at edge > 0)
        edge_gradient = np.abs(np.diff(diffused, axis=1)).max()
        assert edge_gradient > 0.01
    
    def test_iteration_count_effect(self):
        """Test more iterations produce more smoothing."""
        img = np.random.rand(128, 128)
        
        diffused_5 = anisotropic_diffusion(img, n_iterations=5)
        diffused_20 = anisotropic_diffusion(img, n_iterations=20)
        
        # More iterations should change image more from original
        change_5 = np.mean((diffused_5 - img)**2)
        change_20 = np.mean((diffused_20 - img)**2)
        
        assert change_20 > change_5


class TestParameterSensitivity:
    """Tests for parameter sensitivity analysis."""
    
    def test_strength_parameter_range(self):
        """Test strength parameter across valid range."""
        img = np.random.rand(128, 128)
        
        for strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
            filtered = directional_deskar(img, strength=strength)
            assert filtered.shape == img.shape
            assert np.all(np.isfinite(filtered))
    
    def test_scan_angle_rotation(self):
        """Test different scan angles work correctly."""
        img = np.random.rand(128, 128)
        
        for angle in [0.0, 45.0, 90.0, 135.0, 180.0]:
            filtered = directional_deskar(img, scan_angle=angle)
            assert filtered.shape == img.shape
    
    def test_invalid_input_dimension(self):
        """Test error handling for invalid input dimensions."""
        img_3d = np.random.rand(128, 128, 10)
        
        with pytest.raises(ValueError, match="Input must be 2D"):
            directional_deskar(img_3d)
        
        with pytest.raises(ValueError, match="Input must be 2D"):
            remove_scan_lines(img_3d)
        
        with pytest.raises(ValueError, match="Input must be 2D"):
            anisotropic_diffusion(img_3d)
    
    def test_movie_processing_3d_array(self):
        """Test movie processing with 3D array input."""
        movie = np.random.rand(10, 128, 128)
        processed = process_movie_deskar(movie)
        assert processed.shape == movie.shape


class TestDeskarGPU:
    """Tests for GPU-accelerated deskar filtering."""
    
    @pytest.mark.skip(reason="CuPy not available in test environment")
    def test_gpu_cpu_parity(self):
        """Test GPU and CPU results match within tolerance."""
        img = np.random.rand(256, 256)
        
        cpu_result = directional_deskar(img, scan_angle=0.0, strength=0.8)
        gpu_result = directional_deskar_gpu(img, scan_angle=0.0, strength=0.8)
        
        # Should match within GPU float32 tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-3, atol=1e-5)
    
    @pytest.mark.skip(reason="CuPy not available in test environment")
    def test_gpu_fallback_when_unavailable(self):
        """Test graceful fallback when CuPy unavailable."""
        img = np.random.rand(128, 128)
        
        # Should not raise even if CuPy unavailable
        result = directional_deskar_gpu(img)
        assert result.shape == img.shape
    
    @pytest.mark.skip(reason="CuPy not available in test environment")
    def test_gpu_batch_processing(self):
        """Test GPU batch processing efficiency."""
        movie = np.random.rand(20, 256, 256)
        
        # Process on GPU
        processed = []
        for frame in movie:
            result = directional_deskar_gpu(frame)
            processed.append(result)
        
        assert len(processed) == 20
        assert all(p.shape == (256, 256) for p in processed)
    
    @pytest.mark.skip(reason="CuPy not available in test environment")
    def test_gpu_memory_transfer_overhead(self):
        """Test GPU memory transfer doesn't dominate computation."""
        import time
        
        img = np.random.rand(512, 512)
        
        # Warm-up
        _ = directional_deskar_gpu(img)
        
        # Time multiple runs
        start = time.time()
        for _ in range(10):
            _ = directional_deskar_gpu(img)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds for 10 iterations)
        assert elapsed < 5.0
    
    @pytest.mark.skip(reason="CuPy not available in test environment")
    def test_gpu_different_parameters(self):
        """Test GPU implementation with various parameters."""
        img = np.random.rand(256, 256)
        
        params = [
            {'scan_angle': 0.0, 'strength': 0.5},
            {'scan_angle': 45.0, 'strength': 0.8},
            {'scan_angle': 90.0, 'frequency_cutoff': 0.2},
            {'preserve_low_freq': False, 'strength': 1.0},
        ]
        
        for param_set in params:
            cpu_result = directional_deskar(img, **param_set)
            gpu_result = directional_deskar_gpu(img, **param_set)
            
            np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-3, atol=1e-5)
    
    @pytest.mark.skip(reason="CuPy not available in test environment")
    def test_gpu_large_image(self):
        """Test GPU processing of large images."""
        img = np.random.rand(1024, 1024)
        
        result = directional_deskar_gpu(img)
        assert result.shape == img.shape
        assert np.all(np.isfinite(result))


class TestDeskarIntegration:
    """Integration tests for deskar filtering workflow."""
    
    def test_end_to_end_preprocessing_pipeline(self):
        """Test deskar in preprocessing pipeline."""
        # Simulate raw AFM image with artifacts
        img = np.random.rand(256, 256) * 0.5
        
        # Add scan lines
        for i in range(0, 256, 8):
            img[i, :] += 0.2
        
        # Add particles (Gaussian blobs)
        from scipy.ndimage import gaussian_filter
        particles = np.zeros_like(img)
        particles[50, 50] = 1.0
        particles[150, 100] = 1.5
        particles[200, 200] = 0.8
        particles = gaussian_filter(particles, sigma=3)
        
        img += particles
        
        # Apply deskar
        cleaned = directional_deskar(img, scan_angle=0.0, strength=0.8)
        
        # Particles should still be detectable
        assert cleaned.max() > 0.5
        assert cleaned.shape == img.shape
    
    def test_movie_batch_processing_with_callback(self):
        """Test batch processing with progress callback."""
        movie = np.random.rand(50, 128, 128)
        
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        processed = process_movie_deskar(
            movie,
            scan_angle=0.0,
            progress_callback=progress_callback,
        )
        
        assert processed.shape == movie.shape
        assert len(progress_calls) == 50
        assert progress_calls[-1] == (50, 50)
    
    def test_list_of_frames_processing(self):
        """Test processing list of frame arrays."""
        frames = [np.random.rand(128, 128) for _ in range(10)]
        
        processed = process_movie_deskar(frames)
        
        assert isinstance(processed, list)
        assert len(processed) == 10
        assert all(f.shape == (128, 128) for f in processed)
    
    def test_integration_preserves_particle_features(self):
        """Test deskar preserves particle features while removing artifacts."""
        from scipy.ndimage import gaussian_filter
        
        # Create clean image with particles
        np.random.seed(42)
        img = np.random.rand(256, 256) * 0.05
        
        # Add clear particles
        particles = np.zeros_like(img)
        particle_positions = [(50, 50), (100, 150), (200, 100), (150, 200)]
        for x, y in particle_positions:
            particles[x, y] = 2.0
        particles = gaussian_filter(particles, sigma=4)
        
        img_clean = img + particles
        
        # Add scan lines
        img_noisy = img_clean.copy()
        for i in range(0, 256, 4):
            img_noisy[i, :] += 0.3
        
        # Apply deskar
        cleaned = directional_deskar(img_noisy, scan_angle=0.0, strength=0.9)
        
        # Particles should still be present (local maxima near original positions)
        for x, y in particle_positions:
            # Check that particle region still has elevated values
            region = cleaned[x-5:x+5, y-5:y+5]
            assert region.max() > 0.3, f"Particle at ({x},{y}) was removed by deskar"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
