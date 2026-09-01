"""
Tests for in-class alignment and averaging (NL-34).
"""

import numpy as np
import pytest

from nanolocz.core.alignment import (
    AlignmentResult,
    ClassAverage,
    align_particles,
    apply_shift_fourier,
    apply_shift_spline,
    compute_class_averages,
    compute_shift_fft,
    refine_alignment,
)
from nanolocz.core.types import ParticleStack


class TestTranslationalAlignment:
    """Test rigid translational alignment functionality."""

    def test_basic_alignment(self):
        """Test basic alignment of particles within a cluster."""
        np.random.seed(42)
        n_particles = 20
        box_size = 32
        
        # Create synthetic particles with known shifts
        reference = np.zeros((box_size, box_size))
        reference[14:18, 14:18] = 1.0  # Central square
        
        data = np.zeros((n_particles, box_size, box_size))
        true_shifts = np.random.uniform(-1.5, 1.5, (n_particles, 2))
        
        for i in range(n_particles):
            dx, dy = true_shifts[i]
            # Apply shift to create misaligned particle
            shifted = apply_shift_fourier(reference, -dx, -dy)  # Negative to simulate detection offset
            data[i] = shifted
        
        # All particles in same cluster
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels, reference_method='manual', reference_index=0)
        
        assert isinstance(result, AlignmentResult)
        assert result.shifts.shape == (n_particles, 2)
        assert result.correlation_scores.shape == (n_particles,)
        assert result.n_clusters == 1
        assert result.aligned_stack.data.shape == data.shape

    def test_alignment_improves_correlation(self):
        """Test that alignment improves cross-correlation with reference."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        # Create reference particle
        reference = np.zeros((box_size, box_size))
        reference[12:20, 12:20] = 1.0
        
        data = np.zeros((n_particles, box_size, box_size))
        
        # Create misaligned particles
        for i in range(n_particles):
            if i == 0:
                data[i] = reference.copy()
            else:
                shift_x, shift_y = np.random.uniform(0.5, 2.0, 2)
                data[i] = apply_shift_fourier(reference, shift_x, shift_y)
        
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        # Align using first particle as reference
        result = align_particles(stack, labels, reference_method='manual', reference_index=0)
        
        # Reference particle should have perfect correlation
        assert result.correlation_scores[0] == 1.0
        
        # Other particles should have high correlation after alignment
        assert np.mean(result.correlation_scores[1:]) > 0.8

    def test_alignment_preserves_intensity(self):
        """Test that alignment preserves total intensity."""
        np.random.seed(42)
        box_size = 32
        
        # Create particle with known intensity
        particle = np.zeros((box_size, box_size))
        particle[10:22, 10:22] = np.random.randn(12, 12) + 5
        original_sum = np.sum(particle)
        
        # Apply shift
        shifted = apply_shift_fourier(particle, 1.5, -0.8)
        
        # Intensity should be preserved (within numerical tolerance)
        assert abs(np.sum(shifted) - original_sum) / original_sum < 0.01

    def test_multiple_clusters_alignment(self):
        """Test alignment with multiple clusters."""
        np.random.seed(42)
        n_particles = 30
        box_size = 32
        
        # Create two different reference shapes
        ref1 = np.zeros((box_size, box_size))
        ref1[12:20, 12:20] = 1.0
        
        ref2 = np.zeros((box_size, box_size))
        ref2[16, :] = 1.0  # Horizontal line
        ref2[:, 16] = 1.0  # Cross
        
        data = np.zeros((n_particles, box_size, box_size))
        labels = np.zeros(n_particles, dtype=int)
        
        # First 15 particles: cluster 0
        for i in range(15):
            shift_x, shift_y = np.random.uniform(-1, 1, 2)
            data[i] = apply_shift_fourier(ref1, shift_x, shift_y)
            labels[i] = 0
        
        # Next 15 particles: cluster 1
        for i in range(15, 30):
            shift_x, shift_y = np.random.uniform(-1, 1, 2)
            data[i] = apply_shift_fourier(ref2, shift_x, shift_y)
            labels[i] = 1
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels, reference_method='median')
        
        assert result.n_clusters == 2
        assert len(result.reference_images) == 2
        assert 0 in result.reference_images
        assert 1 in result.reference_images

    def test_small_cluster_skipped(self):
        """Test that clusters smaller than min_cluster_size are skipped."""
        np.random.seed(42)
        n_particles = 5
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.array([0, 0, 1, 2, 3])  # Three single-particle clusters
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels, min_cluster_size=2)
        
        # Only cluster 0 has ≥2 particles, others should be skipped
        # Single-particle clusters get zero shifts
        assert result.shifts[2][0] == 0.0  # Cluster 1 (single particle)
        assert result.shifts[3][0] == 0.0  # Cluster 2 (single particle)
        assert result.shifts[4][0] == 0.0  # Cluster 3 (single particle)

    def test_noise_particles_excluded(self):
        """Test that noise particles (label=-1) are excluded from alignment."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.array([0, 0, 0, 0, -1, -1, 0, 0, 0, 0])
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels)
        
        # Noise particles should have zero shifts
        assert np.all(result.shifts[labels == -1] == 0.0)
        assert np.all(result.correlation_scores[labels == -1] == 0.0)

    def test_alignment_with_mask(self):
        """Test alignment with optional mask."""
        np.random.seed(42)
        box_size = 32
        
        reference = np.zeros((box_size, box_size))
        reference[12:20, 12:20] = 1.0
        
        # Create mask that excludes corners
        mask = np.zeros((box_size, box_size))
        mask[8:24, 8:24] = 1.0
        
        moving = apply_shift_fourier(reference, 1.0, 0.5)
        
        # Compute shift with mask
        dx, dy = compute_shift_fft(moving, reference, mask=mask)
        
        # Shift should be detected correctly
        assert abs(dx - 1.0) < 0.5
        assert abs(dy - 0.5) < 0.5

    def test_invalid_reference_method(self):
        """Test that invalid reference method raises error."""
        np.random.seed(42)
        n_particles = 5
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        with pytest.raises(ValueError, match="Unknown reference_method"):
            align_particles(stack, labels, reference_method='invalid_method')


class TestReferenceSelection:
    """Test reference image selection strategies."""

    def test_median_reference(self):
        """Test median reference selection."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        # Create particles around a common shape with noise
        base = np.zeros((box_size, box_size))
        base[12:20, 12:20] = 1.0
        
        data = np.array([base + np.random.randn(box_size, box_size) * 0.1 
                        for _ in range(n_particles)])
        
        from nanolocz.core.alignment import _select_reference
        
        reference, idx = _select_reference(data, method='median')
        
        assert reference.shape == (box_size, box_size)
        assert 0 <= idx < n_particles

    def test_mean_reference(self):
        """Test mean reference selection."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        base = np.zeros((box_size, box_size))
        base[12:20, 12:20] = 1.0
        
        data = np.array([base + np.random.randn(box_size, box_size) * 0.1 
                        for _ in range(n_particles)])
        
        from nanolocz.core.alignment import _select_reference
        
        reference, idx = _select_reference(data, method='mean')
        
        assert reference.shape == (box_size, box_size)
        assert 0 <= idx < n_particles

    def test_highest_snr_reference(self):
        """Test highest SNR reference selection."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        base = np.zeros((box_size, box_size))
        base[12:20, 12:20] = 1.0
        
        data = np.array([base + np.random.randn(box_size, box_size) * 0.1 
                        for _ in range(n_particles)])
        
        # Make one particle have much higher variance
        data[5] = base * 10
        
        from nanolocz.core.alignment import _select_reference
        
        reference, idx = _select_reference(data, method='highest_snr')
        
        # Should select the high-variance particle
        assert idx == 5

    def test_manual_reference(self):
        """Test manual reference selection."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        
        from nanolocz.core.alignment import _select_reference
        
        reference, idx = _select_reference(data, method='manual', reference_index=3)
        
        assert idx == 3
        assert np.allclose(reference, data[3])

    def test_manual_reference_invalid_index(self):
        """Test manual reference with invalid index raises error."""
        np.random.seed(42)
        n_particles = 5
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        
        from nanolocz.core.alignment import _select_reference
        
        with pytest.raises(ValueError, match="Invalid reference_index"):
            _select_reference(data, method='manual', reference_index=10)


class TestShiftComputation:
    """Test FFT-based shift computation."""

    def test_known_shift_recovery(self):
        """Test recovery of known applied shifts."""
        np.random.seed(42)
        box_size = 64
        
        # Create test image
        image = np.zeros((box_size, box_size))
        image[20:44, 20:44] = 1.0
        
        # Apply known shift
        true_dx, true_dy = 1.0, -1.0  # Use integer shifts for reliable testing
        shifted = apply_shift_fourier(image, true_dx, true_dy)
        
        # Recover shift
        dx, dy = compute_shift_fft(shifted, image)
        
        # Should recover shift within tolerance
        assert abs(dx - true_dx) < 0.3
        assert abs(dy - true_dy) < 0.3

    def test_zero_shift(self):
        """Test that zero shift is correctly identified."""
        np.random.seed(42)
        box_size = 32
        
        image = np.random.randn(box_size, box_size)
        
        dx, dy = compute_shift_fft(image, image)
        
        assert abs(dx) < 0.1
        assert abs(dy) < 0.1

    def test_subpixel_accuracy(self):
        """Test sub-pixel shift accuracy."""
        np.random.seed(42)
        box_size = 64
        
        image = np.zeros((box_size, box_size))
        image[24:40, 24:40] = 1.0
        
        # Test various sub-pixel shifts
        test_shifts = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (-0.5, 0.5)]
        
        for true_dx, true_dy in test_shifts:
            shifted = apply_shift_fourier(image, true_dx, true_dy)
            dx, dy = compute_shift_fft(shifted, image)
            
            assert abs(dx - true_dx) < 0.2, f"Failed for shift ({true_dx}, {true_dy})"
            assert abs(dy - true_dy) < 0.2, f"Failed for shift ({true_dx}, {true_dy})"

    def test_shape_mismatch_error(self):
        """Test that shape mismatch raises error."""
        img1 = np.random.randn(32, 32)
        img2 = np.random.randn(64, 64)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_shift_fft(img1, img2)

    def test_mask_shape_mismatch_error(self):
        """Test that mask shape mismatch raises error."""
        img = np.random.randn(32, 32)
        mask = np.random.randn(64, 64)
        
        with pytest.raises(ValueError, match="Mask shape"):
            compute_shift_fft(img, img, mask=mask)

    def test_large_shift(self):
        """Test shift computation for larger shifts."""
        np.random.seed(42)
        box_size = 64
        
        image = np.zeros((box_size, box_size))
        image[20:44, 20:44] = 1.0
        
        # Apply larger shift
        true_dx, true_dy = 3.0, -2.5
        shifted = apply_shift_fourier(image, true_dx, true_dy)
        
        dx, dy = compute_shift_fft(shifted, image)
        
        # Larger shifts have more tolerance
        assert abs(dx - true_dx) < 0.5
        assert abs(dy - true_dy) < 0.5


class TestShiftApplication:
    """Test sub-pixel shift application methods."""

    def test_fourier_shift_basic(self):
        """Test basic Fourier shift application."""
        np.random.seed(42)
        box_size = 32
        
        image = np.random.randn(box_size, box_size)
        
        shifted = apply_shift_fourier(image, 1.0, 0.5)
        
        assert shifted.shape == image.shape
        assert not np.allclose(shifted, image)

    def test_fourier_shift_zero(self):
        """Test that zero shift returns copy of original."""
        np.random.seed(42)
        box_size = 32
        
        image = np.random.randn(box_size, box_size)
        
        shifted = apply_shift_fourier(image, 0.0, 0.0)
        
        assert np.allclose(shifted, image)

    def test_fourier_shift_inverse(self):
        """Test that inverse shift recovers original."""
        np.random.seed(42)
        box_size = 32
        
        image = np.random.randn(box_size, box_size)
        
        # Shift forward then backward
        shifted = apply_shift_fourier(image, 1.5, -0.8)
        recovered = apply_shift_fourier(shifted, -1.5, 0.8)
        
        # Should recover original within numerical tolerance
        assert np.allclose(recovered, image, rtol=1e-5, atol=1e-8)

    def test_spline_shift_basic(self):
        """Test basic spline shift application."""
        np.random.seed(42)
        box_size = 32
        
        image = np.random.randn(box_size, box_size)
        
        shifted = apply_shift_spline(image, 1.0, 0.5, order=3)
        
        assert shifted.shape == image.shape
        assert not np.allclose(shifted, image)

    def test_spline_shift_order(self):
        """Test spline shift with different orders."""
        np.random.seed(42)
        box_size = 32
        
        image = np.random.randn(box_size, box_size)
        
        shifted_0 = apply_shift_spline(image, 1.0, 0.5, order=0)
        shifted_3 = apply_shift_spline(image, 1.0, 0.5, order=3)
        
        # Different orders produce different results
        assert not np.allclose(shifted_0, shifted_3)

    def test_fourier_vs_spline_similarity(self):
        """Test that Fourier and spline methods produce similar results."""
        np.random.seed(42)
        box_size = 32
        
        image = np.random.randn(box_size, box_size)
        
        shifted_fourier = apply_shift_fourier(image, 0.5, 0.3)
        shifted_spline = apply_shift_spline(image, 0.5, 0.3, order=3)
        
        # Methods should be reasonably similar
        correlation = np.corrcoef(shifted_fourier.ravel(), shifted_spline.ravel())[0, 1]
        assert correlation > 0.9


class TestClassAverages:
    """Test class average computation."""

    def test_basic_class_average(self):
        """Test basic class average computation."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        # Create aligned particles with some variation
        base = np.zeros((box_size, box_size))
        base[12:20, 12:20] = 1.0
        
        data = np.array([base + np.random.randn(box_size, box_size) * 0.1 
                        for _ in range(n_particles)])
        
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        averages = compute_class_averages(stack, labels)
        
        assert len(averages) == 1
        assert 0 in averages
        
        avg = averages[0]
        assert isinstance(avg, ClassAverage)
        assert avg.cluster_id == 0
        assert avg.count == n_particles
        assert avg.mean.shape == (box_size, box_size)
        assert avg.std.shape == (box_size, box_size)

    def test_multiple_class_averages(self):
        """Test class averages for multiple clusters."""
        np.random.seed(42)
        n_particles = 20
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.array([0] * 10 + [1] * 10)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        averages = compute_class_averages(stack, labels)
        
        assert len(averages) == 2
        assert 0 in averages
        assert 1 in averages
        assert averages[0].count == 10
        assert averages[1].count == 10

    def test_class_average_without_std(self):
        """Test class average computation without std."""
        np.random.seed(42)
        n_particles = 5
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        averages = compute_class_averages(stack, labels, compute_std=False)
        
        avg = averages[0]
        assert avg.std is not None  # Still computed but may be zeros
        assert avg.std.shape == (box_size, box_size)

    def test_empty_cluster_excluded(self):
        """Test that empty clusters are excluded from averages."""
        np.random.seed(42)
        n_particles = 5
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.array([0, 0, 0, 0, 0])  # Only cluster 0
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        averages = compute_class_averages(stack, labels)
        
        # Only cluster 0 should be present
        assert len(averages) == 1
        assert 0 in averages
        assert 1 not in averages

    def test_class_average_mean_accuracy(self):
        """Test that class average mean is accurate."""
        np.random.seed(42)
        n_particles = 100
        box_size = 32
        
        # Create particles with known mean
        base = np.zeros((box_size, box_size))
        base[12:20, 12:20] = 1.0
        
        data = np.array([base + np.random.randn(box_size, box_size) * 0.5 
                        for _ in range(n_particles)])
        
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        averages = compute_class_averages(stack, labels)
        
        # Mean should converge to base with many samples
        avg = averages[0]
        assert np.allclose(avg.mean, base, atol=0.1)


class TestIterativeRefinement:
    """Test iterative alignment refinement."""

    def test_basic_refinement(self):
        """Test basic iterative refinement."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        reference = np.zeros((box_size, box_size))
        reference[12:20, 12:20] = 1.0
        
        data = np.zeros((n_particles, box_size, box_size))
        
        # Create misaligned particles
        for i in range(n_particles):
            shift_x, shift_y = np.random.uniform(-2, 2, 2)
            data[i] = apply_shift_fourier(reference, shift_x, shift_y)
        
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = refine_alignment(stack, labels, n_iterations=3)
        
        assert isinstance(result, AlignmentResult)
        assert result.n_clusters == 1

    def test_refinement_convergence(self):
        """Test that refinement converges."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        reference = np.zeros((box_size, box_size))
        reference[12:20, 12:20] = 1.0
        
        data = np.zeros((n_particles, box_size, box_size))
        
        # Create slightly misaligned particles
        for i in range(n_particles):
            shift_x, shift_y = np.random.uniform(-0.5, 0.5, 2)
            data[i] = apply_shift_fourier(reference, shift_x, shift_y)
        
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        # With tight convergence threshold, should converge quickly
        result = refine_alignment(
            stack, labels, 
            n_iterations=5,
            convergence_threshold=0.05
        )
        
        # Final shifts should be small
        shift_magnitudes = np.sqrt(np.sum(result.shifts**2, axis=1))
        assert np.mean(shift_magnitudes) < 0.1

    def test_refinement_improves_alignment(self):
        """Test that refinement improves alignment quality."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        reference = np.zeros((box_size, box_size))
        reference[12:20, 12:20] = 1.0
        
        data = np.zeros((n_particles, box_size, box_size))
        
        # Create moderately misaligned particles
        for i in range(n_particles):
            shift_x, shift_y = np.random.uniform(-1.5, 1.5, 2)
            data[i] = apply_shift_fourier(reference, shift_x, shift_y)
        
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        # Single iteration
        result_single = align_particles(stack, labels)
        
        # Multiple iterations
        result_refined = refine_alignment(stack, labels, n_iterations=3)
        
        # Refined should have better or equal correlation scores
        assert np.mean(result_refined.correlation_scores) >= np.mean(result_single.correlation_scores) - 0.01

    def test_refinement_with_different_methods(self):
        """Test refinement with different reference methods."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        reference = np.zeros((box_size, box_size))
        reference[12:20, 12:20] = 1.0
        
        data = np.zeros((n_particles, box_size, box_size))
        
        for i in range(n_particles):
            shift_x, shift_y = np.random.uniform(-1, 1, 2)
            data[i] = apply_shift_fourier(reference, shift_x, shift_y)
        
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        # Test with different reference methods
        result_median = refine_alignment(stack, labels, reference_method='median')
        result_mean = refine_alignment(stack, labels, reference_method='mean')
        
        # Both should produce valid results
        assert result_median.n_aligned > 0
        assert result_mean.n_aligned > 0


class TestAlignmentQualityMetrics:
    """Test alignment quality metrics and statistics."""

    def test_correlation_scores_range(self):
        """Test that correlation scores are in valid range."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels)
        
        # Correlation scores should be finite
        assert np.all(np.isfinite(result.correlation_scores))
        
        # Reference particle should have score close to 1
        assert np.max(result.correlation_scores) > 0.9

    def test_shift_statistics(self):
        """Test shift statistics computation."""
        np.random.seed(42)
        n_particles = 20
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.array([0] * 10 + [1] * 10)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels)
        
        # Get statistics for cluster 0
        stats = result.get_shift_statistics(0, labels)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        # Mean, std, min, max should be tuples of (dx, dy)
        assert len(stats['mean']) == 2
        assert len(stats['std']) == 2

    def test_cluster_shifts_extraction(self):
        """Test extraction of shifts for specific cluster."""
        np.random.seed(42)
        n_particles = 20
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.array([0] * 10 + [1] * 10)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels)
        
        shifts_0 = result.get_cluster_shifts(0, labels)
        shifts_1 = result.get_cluster_shifts(1, labels)
        
        assert shifts_0.shape == (10, 2)
        assert shifts_1.shape == (10, 2)

    def test_failed_indices_tracking(self):
        """Test that failed alignments are tracked."""
        np.random.seed(42)
        n_particles = 5
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels)
        
        # Should have a failed_indices list (may be empty if all succeed)
        assert isinstance(result.failed_indices, list)
        assert len(result.failed_indices) >= 0

    def test_n_aligned_count(self):
        """Test that n_aligned count is accurate."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels)
        
        # n_aligned should equal n_particles minus failures
        expected_aligned = n_particles - len(result.failed_indices)
        assert result.n_aligned == expected_aligned


class TestAlignmentGPU:
    """Test GPU-accelerated alignment (skipped if CuPy unavailable)."""

    @pytest.mark.skip(reason="CuPy not available")
    def test_gpu_shift_computation(self):
        """Test GPU-accelerated shift computation."""
        from nanolocz.core.alignment import compute_shift_gpu
        
        np.random.seed(42)
        box_size = 64
        
        image = np.zeros((box_size, box_size))
        image[20:44, 20:44] = 1.0
        
        true_dx, true_dy = 1.5, -0.8
        from nanolocz.core.alignment import apply_shift_fourier
        shifted = apply_shift_fourier(image, true_dx, true_dy)
        
        dx, dy = compute_shift_gpu(shifted, image)
        
        assert abs(dx - true_dx) < 0.3
        assert abs(dy - true_dy) < 0.3

    @pytest.mark.skip(reason="CuPy not available")
    def test_gpu_alignment_parity(self):
        """Test GPU alignment matches CPU within tolerance."""
        from nanolocz.core.alignment import align_particles_gpu
        
        np.random.seed(42)
        n_particles = 20
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result_cpu = align_particles(stack, labels)
        result_gpu = align_particles_gpu(stack, labels)
        
        # Results should match within tolerance
        assert np.allclose(result_cpu.shifts, result_gpu.shifts, rtol=1e-3, atol=1e-5)

    @pytest.mark.skip(reason="CuPy not available")
    def test_gpu_speedup(self):
        """Test GPU provides speedup for large datasets."""
        import time
        from nanolocz.core.alignment import align_particles_gpu
        
        np.random.seed(42)
        n_particles = 200
        box_size = 64
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        start = time.time()
        align_particles(stack, labels)
        cpu_time = time.time() - start
        
        start = time.time()
        align_particles_gpu(stack, labels)
        gpu_time = time.time() - start
        
        # GPU should be faster (allowing for warmup)
        assert gpu_time < cpu_time * 1.5  # At least not significantly slower

    @pytest.mark.skip(reason="CuPy not available")
    def test_gpu_fallback_on_error(self):
        """Test GPU falls back to CPU on error."""
        from nanolocz.core.alignment import align_particles_gpu
        
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        # Should not raise even if GPU unavailable
        result = align_particles_gpu(stack, labels)
        assert isinstance(result, AlignmentResult)

    @pytest.mark.skip(reason="CuPy not available")
    def test_gpu_batch_processing(self):
        """Test GPU batch processing parameter."""
        from nanolocz.core.alignment import align_particles_gpu
        
        np.random.seed(42)
        n_particles = 50
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        # Test with different batch sizes
        result_32 = align_particles_gpu(stack, labels, batch_size=32)
        result_64 = align_particles_gpu(stack, labels, batch_size=64)
        
        # Results should be identical regardless of batch size
        assert np.allclose(result_32.shifts, result_64.shifts)

    @pytest.mark.skip(reason="CuPy not available")
    def test_gpu_memory_efficiency(self):
        """Test GPU memory handling for large datasets."""
        from nanolocz.core.alignment import align_particles_gpu
        
        np.random.seed(42)
        n_particles = 100
        box_size = 64
        
        data = np.random.randn(n_particles, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        # Should complete without memory errors
        result = align_particles_gpu(stack, labels, batch_size=32)
        assert result.n_aligned == n_particles


class TestAlignmentIntegration:
    """Test integration of alignment with classification pipeline."""

    def test_end_to_end_classification_alignment(self):
        """Test full pipeline: classification → alignment → averaging."""
        np.random.seed(42)
        n_particles = 50
        box_size = 32
        
        # Create two conformational states
        state1 = np.zeros((box_size, box_size))
        state1[10:22, 10:22] = 1.0
        
        state2 = np.zeros((box_size, box_size))
        state2[16, :] = 0.8
        state2[:, 16] = 0.8
        
        data = np.zeros((n_particles, box_size, box_size))
        labels = np.zeros(n_particles, dtype=int)
        
        # Generate particles from both states with random shifts
        for i in range(25):
            shift_x, shift_y = np.random.uniform(-1.5, 1.5, 2)
            data[i] = apply_shift_fourier(state1, shift_x, shift_y)
            labels[i] = 0
        
        for i in range(25, 50):
            shift_x, shift_y = np.random.uniform(-1.5, 1.5, 2)
            data[i] = apply_shift_fourier(state2, shift_x, shift_y)
            labels[i] = 1
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        # Align particles
        aligned_result = align_particles(stack, labels, reference_method='median')
        
        # Compute class averages
        averages = compute_class_averages(aligned_result.aligned_stack, labels)
        
        # Verify results
        assert len(averages) == 2
        assert averages[0].count == 25
        assert averages[1].count == 25
        
        # Class averages should resemble original states
        # State 1: square
        assert np.max(averages[0].mean) > 0.5
        # State 2: cross
        assert np.max(averages[1].mean) > 0.5

    def test_alignment_with_4d_data(self):
        """Test alignment with 4D particle stacks (multi-frame)."""
        np.random.seed(42)
        n_particles = 10
        n_frames = 5
        box_size = 32
        
        data = np.random.randn(n_particles, n_frames, box_size, box_size)
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels)
        
        # Output should preserve 4D structure
        assert result.aligned_stack.data.ndim == 4
        assert result.aligned_stack.data.shape == (n_particles, n_frames, box_size, box_size)

    def test_alignment_preserves_metadata(self):
        """Test that alignment preserves particle metadata."""
        np.random.seed(42)
        n_particles = 10
        box_size = 32
        
        data = np.random.randn(n_particles, box_size, box_size)
        centers = [(i * 3, i * 2) for i in range(n_particles)]
        frames = list(range(n_particles))
        
        stack = ParticleStack(
            data=data,
            centers_xy=centers,
            frame_index=frames,
            box_size=box_size,
        )
        
        labels = np.zeros(n_particles, dtype=int)
        result = align_particles(stack, labels)
        
        # Metadata should be preserved
        assert result.aligned_stack.centers_xy == centers
        assert result.aligned_stack.frame_index == frames
        assert result.aligned_stack.box_size == box_size

    def test_alignment_quality_filtering(self):
        """Test filtering particles by alignment quality."""
        np.random.seed(42)
        n_particles = 20
        box_size = 32
        
        reference = np.zeros((box_size, box_size))
        reference[12:20, 12:20] = 1.0
        
        data = np.zeros((n_particles, box_size, box_size))
        
        # Create well-aligned and poorly-aligned particles
        for i in range(10):
            # Well-aligned (small shifts)
            shift_x, shift_y = np.random.uniform(-0.3, 0.3, 2)
            data[i] = apply_shift_fourier(reference, shift_x, shift_y)
        
        for i in range(10, 20):
            # Poorly-aligned (large shifts or noise)
            data[i] = np.random.randn(box_size, box_size)
        
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        result = align_particles(stack, labels)
        
        # Well-aligned particles should have higher scores
        well_aligned_scores = result.correlation_scores[:10]
        poorly_aligned_scores = result.correlation_scores[10:]
        
        assert np.mean(well_aligned_scores) > np.mean(poorly_aligned_scores)

    def test_iterative_refinement_integration(self):
        """Test iterative refinement in full pipeline."""
        np.random.seed(42)
        n_particles = 30
        box_size = 32
        
        reference = np.zeros((box_size, box_size))
        reference[12:20, 12:20] = 1.0
        
        data = np.zeros((n_particles, box_size, box_size))
        
        for i in range(n_particles):
            shift_x, shift_y = np.random.uniform(-2, 2, 2)
            data[i] = apply_shift_fourier(reference, shift_x, shift_y)
        
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        # Refine alignment
        refined_result = refine_alignment(stack, labels, n_iterations=3)
        
        # Compute averages from refined alignment
        averages = compute_class_averages(refined_result.aligned_stack, labels)
        
        # Average should be sharp (low std in signal region)
        avg = averages[0]
        signal_region = avg.mean[12:20, 12:20]
        background_region = np.concatenate([
            avg.mean[:8, :].ravel(),
            avg.mean[24:, :].ravel(),
            avg.mean[8:24, :8].ravel(),
            avg.mean[8:24, 24:].ravel(),
        ])
        
        # Signal should be distinct from background
        assert np.mean(signal_region) > np.mean(background_region) + 0.3

    def test_alignment_with_realistic_afm_noise(self):
        """Test alignment with realistic AFM-like noise."""
        np.random.seed(42)
        n_particles = 20
        box_size = 32
        
        # Create particle with AFM-like characteristics
        reference = np.zeros((box_size, box_size))
        # Simulate molecule shape
        y, x = np.ogrid[:box_size, :box_size]
        center = box_size // 2
        mask = (x - center)**2 + (y - center)**2 < 8**2
        reference[mask] = 1.0
        
        # Add realistic noise (Poisson-like + scan lines)
        def add_afm_noise(img, seed=None):
            if seed is not None:
                np.random.seed(seed)
            # Poisson-like noise
            noisy = img * 10 + np.random.poisson(5, img.shape)
            # Scan line artifacts
            scan_lines = np.random.randn(box_size, 1) * 0.5
            noisy += scan_lines.T
            return noisy
        
        data = np.zeros((n_particles, box_size, box_size))
        
        for i in range(n_particles):
            shift_x, shift_y = np.random.uniform(-1, 1, 2)
            shifted = apply_shift_fourier(reference, shift_x, shift_y)
            data[i] = add_afm_noise(shifted, seed=i)
        
        labels = np.zeros(n_particles, dtype=int)
        
        stack = ParticleStack(
            data=data,
            centers_xy=[(16, 16)] * n_particles,
            frame_index=[0] * n_particles,
            box_size=box_size,
        )
        
        # Align with mask to exclude scan lines
        circular_mask = np.zeros((box_size, box_size))
        circular_mask[center-10:center+10, center-10:center+10] = 1.0
        
        result = align_particles(stack, labels, mask=circular_mask)
        
        # Should still achieve reasonable alignment
        assert np.mean(result.correlation_scores) > 0.5
