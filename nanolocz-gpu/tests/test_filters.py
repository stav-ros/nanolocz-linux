"""Tests for filters module (NL-15)."""

import numpy as np
import pytest
from nanolocz.core.filters import (
    gaussian_blur,
    median_blur,
    uniform_blur,
    compute_gradient,
    compute_laplacian,
    create_mask_from_threshold,
    create_circular_mask,
    create_rectangular_mask,
    remove_scars,
    extract_profile,
    extract_radial_profile,
    morphological_operations,
    distance_transform
)


class TestGaussianBlur:
    """Test Gaussian filtering."""
    
    def test_basic_gaussian_blur(self):
        img = np.random.rand(100, 100)
        blurred = gaussian_blur(img, sigma=2.0)
        assert blurred.shape == img.shape
        assert blurred.dtype == np.float64
    
    def test_gaussian_reduces_noise(self):
        clean = np.ones((100, 100)) * 0.5
        noisy = clean + np.random.randn(100, 100) * 0.1
        blurred = gaussian_blur(noisy, sigma=3.0)
        assert np.mean((blurred - clean)**2) < np.mean((noisy - clean)**2)
    
    def test_gaussian_preserves_mean(self):
        img = np.random.rand(100, 100)
        blurred = gaussian_blur(img, sigma=1.0)
        assert np.isclose(np.mean(img), np.mean(blurred), rtol=0.01)


class TestMedianBlur:
    """Test median filtering."""
    
    def test_basic_median_blur(self):
        img = np.random.rand(100, 100)
        denoised = median_blur(img, size=3)
        assert denoised.shape == img.shape
        assert denoised.dtype == np.float64
    
    def test_median_removes_salt_pepper(self):
        clean = np.ones((100, 100)) * 0.5
        noisy = clean.copy()
        noisy[np.random.rand(100, 100) < 0.05] = 0.0
        noisy[np.random.rand(100, 100) < 0.05] = 1.0
        denoised = median_blur(noisy, size=3)
        assert np.mean((denoised - clean)**2) < np.mean((noisy - clean)**2)


class TestGradient:
    """Test gradient computation."""
    
    def test_gradient_magnitude_shape(self):
        img = np.random.rand(100, 100)
        mag, direction = compute_gradient(img)
        assert mag.shape == img.shape
        assert direction.shape == img.shape
    
    def test_gradient_constant_image(self):
        img = np.ones((50, 50)) * 5.0
        mag, _ = compute_gradient(img)
        assert np.allclose(mag, 0.0)
    
    def test_gradient_direction_range(self):
        img = np.random.rand(100, 100)
        _, direction = compute_gradient(img)
        assert np.all(direction >= -np.pi)
        assert np.all(direction <= np.pi)


class TestMasks:
    """Test mask creation."""
    
    def test_threshold_mask_above(self):
        img = np.array([[0.1, 0.6], [0.4, 0.8]])
        mask = create_mask_from_threshold(img, 0.5, mode='above')
        expected = np.array([[False, True], [False, True]])
        assert np.array_equal(mask, expected)
    
    def test_threshold_mask_below(self):
        img = np.array([[0.1, 0.6], [0.4, 0.8]])
        mask = create_mask_from_threshold(img, 0.5, mode='below')
        expected = np.array([[True, False], [True, False]])
        assert np.array_equal(mask, expected)
    
    def test_circular_mask_center(self):
        mask = create_circular_mask((100, 100), center=(50, 50), radius=25)
        assert mask[50, 50]
        assert not mask[0, 0]
        assert not mask[0, 99]
        assert not mask[99, 0]
        assert not mask[99, 99]
    
    def test_rectangular_mask(self):
        mask = create_rectangular_mask((100, 100), top_left=(20, 30), bottom_right=(40, 50))
        assert mask[30, 40]
        assert not mask[10, 10]
        assert not mask[80, 80]


class TestScarRemoval:
    """Test scar/artifact removal."""
    
    def test_remove_horizontal_scars(self):
        """Test removal of horizontal scan lines with controlled scar."""
        np.random.seed(42)
        img = np.random.rand(100, 100) * 0.5  # Lower base intensity
        original_line = img[50, :].copy()
        img[50, :] = 2.0  # Add consistent bright line
        
        corrected = remove_scars(img, direction='horizontal', threshold=2.0)
        
        # Scar should be reduced toward neighborhood values
        neighborhood_mean = np.mean(img[49, :])
        assert np.abs(corrected[50, :].mean() - neighborhood_mean) < 0.2
    
    def test_remove_vertical_scars(self):
        """Test removal of vertical scan lines with controlled scar."""
        np.random.seed(42)
        img = np.random.rand(100, 100) * 0.5
        img[:, 30] = 2.0  # Add consistent bright line
        
        corrected = remove_scars(img, direction='vertical', threshold=2.0)
        
        # Scar should be reduced
        neighborhood_mean = np.mean(img[:, 29])
        assert np.abs(corrected[:, 30].mean() - neighborhood_mean) < 0.2
    
    def test_scar_removal_preserves_structure(self):
        img = np.random.rand(100, 100)
        corrected = remove_scars(img, direction='horizontal')
        assert np.mean(np.abs(corrected - img)) < 0.1


class TestProfiles:
    """Test profile extraction."""
    
    def test_extract_linear_profile(self):
        img = np.random.rand(100, 100)
        distances, intensities = extract_profile(img, (0, 0), (99, 99))
        assert len(distances) == len(intensities)
        assert distances[0] == 0.0
        assert distances[-1] > 0
    
    def test_extract_radial_profile(self):
        y, x = np.ogrid[:100, :100]
        center = (50, 50)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        img = 1.0 / (r + 1)
        
        radii, mean_int, std_int = extract_radial_profile(img, center)
        
        assert len(radii) == len(mean_int)
        assert len(radii) == len(std_int)
        assert mean_int[0] > mean_int[-1]


class TestMorphology:
    """Test morphological operations."""
    
    def test_morphological_erosion(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        eroded = morphological_operations(mask, 'erode')
        assert np.sum(eroded) < np.sum(mask)
    
    def test_morphological_dilation(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        dilated = morphological_operations(mask, 'dilate')
        assert np.sum(dilated) > np.sum(mask)
    
    def test_morphological_opening(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        mask[0, 0] = True
        opened = morphological_operations(mask, 'open')
        assert not opened[0, 0]
    
    def test_morphological_closing(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        mask[10, 10] = False
        closed = morphological_operations(mask, 'close')
        assert closed[10, 10]
    
    def test_morphological_invalid_operation(self):
        mask = np.random.rand(20, 20) > 0.5
        with pytest.raises(ValueError, match="Unknown operation"):
            morphological_operations(mask, 'invalid')


class TestDistanceTransform:
    """Test distance transform."""
    
    def test_distance_transform_basic(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        dist = distance_transform(mask)
        assert dist.shape == mask.shape
        assert dist.max() > 0
        assert dist[10, 10] == dist.max()
    
    def test_distance_transform_empty(self):
        mask = np.zeros((20, 20), dtype=bool)
        dist = distance_transform(mask)
        assert np.all(dist == 0)
