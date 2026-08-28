"""Tests for leveling module (NL-14)."""

import numpy as np
import pytest
from nanolocz.core.leveling import (
    line_leveling,
    plane_leveling,
    level_image,
    batch_level_movie,
    weighted_multi_plane_leveling
)


class TestLineLeveling:
    """Test line-by-line leveling."""
    
    def test_basic_line_leveling(self):
        """Test basic line leveling operation."""
        img = np.random.rand(100, 100)
        leveled, offsets = line_leveling(img)
        
        assert leveled.shape == img.shape
        assert offsets.shape == (100,)
        assert np.allclose(np.median(leveled, axis=1), np.median(leveled[0, :]))
    
    def test_line_leveling_with_mask(self):
        """Test line leveling with pixel mask."""
        img = np.random.rand(100, 100)
        mask = np.ones_like(img, dtype=bool)
        mask[:20, :] = False  # Mask out first 20 lines
        
        leveled, offsets = line_leveling(img, mask=mask)
        
        assert leveled.shape == img.shape
        assert not np.isnan(leveled).any()
    
    def test_line_leveling_reference_preserved(self):
        """Test that reference line is preserved."""
        img = np.random.rand(100, 100)
        ref_line = 50
        
        leveled, offsets = line_leveling(img, reference_line=ref_line)
        
        # Reference line offset should be 0
        assert np.isclose(offsets[ref_line], 0.0)
    
    def test_line_leveling_constant_image(self):
        """Test line leveling on constant image."""
        img = np.ones((50, 50)) * 5.0
        leveled, offsets = line_leveling(img)
        
        assert np.allclose(leveled, img)
        assert np.allclose(offsets, 0.0)


class TestPlaneLeveling:
    """Test plane fitting and subtraction."""
    
    def test_basic_plane_leveling(self):
        """Test basic plane leveling."""
        img = np.random.rand(100, 100)
        leveled, info = plane_leveling(img)
        
        assert leveled.shape == img.shape
        assert 'plane' in info
        assert 'residual_std' in info
        assert 'r_squared' in info
        assert len(info['plane']) == 3  # [a, b, c]
    
    def test_plane_leveling_with_tilt(self):
        """Test plane leveling removes tilt."""
        # Create image with known tilt
        x = np.linspace(0, 99, 100)
        y = np.linspace(0, 99, 100)
        xx, yy = np.meshgrid(x, y)
        tilt = 0.1 * xx + 0.05 * yy
        img = np.random.rand(100, 100) + tilt
        
        leveled, info = plane_leveling(img)
        
        # Leveled image should have reduced tilt
        assert np.abs(np.mean(leveled)) < np.abs(np.mean(img))
    
    def test_plane_leveling_with_mask(self):
        """Test plane leveling with masked regions."""
        img = np.random.rand(100, 100)
        mask = np.ones_like(img, dtype=bool)
        mask[40:60, 40:60] = False  # Mask out center
        
        leveled, info = plane_leveling(img, mask=mask)
        
        assert leveled.shape == img.shape
        assert not np.isnan(leveled).any()
    
    def test_plane_leveling_goodness_of_fit(self):
        """Test plane fit quality metrics."""
        # Create flat image with noise
        img = np.random.randn(100, 100) * 0.1
        
        leveled, info = plane_leveling(img)
        
        # R-squared should be low for random noise
        assert 0 <= info['r_squared'] <= 1
    
    def test_plane_fits_exactly(self):
        """Test that perfect plane is removed exactly."""
        x = np.linspace(0, 99, 100)
        y = np.linspace(0, 99, 100)
        xx, yy = np.meshgrid(x, y)
        plane = 2.0 * xx + 3.0 * yy + 10.0
        
        leveled, info = plane_leveling(plane)
        
        # Should remove plane almost exactly
        assert np.allclose(leveled, 0.0, atol=1e-10)


class TestWeightedMultiPlane:
    """Test multi-region leveling."""
    
    def test_two_region_leveling(self):
        """Test leveling with two distinct regions."""
        img = np.random.rand(100, 100)
        regions = np.zeros_like(img, dtype=int)
        regions[:50, :] = 1  # Top half is region 1
        
        leveled, info = weighted_multi_plane_leveling(img, regions)
        
        assert leveled.shape == img.shape
        assert len(info['region_planes']) == 2
        assert 'blend_weights' in info
    
    def test_single_region_same_as_plane(self):
        """Test that single region equals plane leveling."""
        img = np.random.rand(100, 100)
        regions = np.zeros_like(img, dtype=int)
        
        leveled_multi, info_multi = weighted_multi_plane_leveling(img, regions)
        leveled_plane, info_plane = plane_leveling(img)
        
        assert np.allclose(leveled_multi, leveled_plane)


class TestLevelImage:
    """Test unified leveling interface."""
    
    def test_level_image_line_method(self):
        """Test line method through unified interface."""
        img = np.random.rand(100, 100)
        leveled, info = level_image(img, method='line')
        
        assert info['method'] == 'line'
        assert 'offsets' in info
    
    def test_level_image_plane_method(self):
        """Test plane method through unified interface."""
        img = np.random.rand(100, 100)
        leveled, info = level_image(img, method='plane')
        
        assert info['method'] == 'plane'
        assert 'plane' in info
    
    def test_level_image_invalid_method(self):
        """Test error on invalid method."""
        img = np.random.rand(100, 100)
        
        with pytest.raises(ValueError, match="Unknown leveling method"):
            level_image(img, method='invalid')
    
    def test_level_image_weighted_requires_regions(self):
        """Test that weighted_plane requires regions."""
        img = np.random.rand(100, 100)
        
        with pytest.raises(ValueError, match="regions required"):
            level_image(img, method='weighted_plane')


class TestBatchLevelMovie:
    """Test batch movie leveling."""
    
    def test_batch_level_3d_movie(self):
        """Test leveling a 3D movie stack."""
        movie = np.random.rand(50, 100, 100)
        leveled, info = batch_level_movie(movie)
        
        assert leveled.shape == movie.shape
        assert len(info) == 50  # One info dict per frame
    
    def test_batch_level_plane_method(self):
        """Test batch leveling with plane method."""
        movie = np.random.rand(20, 64, 64)
        leveled, info = batch_level_movie(movie, method='plane')
        
        assert leveled.shape == movie.shape
        assert all('plane' in frame_info for frame_info in info)
    
    def test_batch_level_with_mask(self):
        """Test batch leveling with common mask."""
        movie = np.random.rand(10, 50, 50)
        mask = np.ones((50, 50), dtype=bool)
        mask[:5, :] = False
        
        leveled, info = batch_level_movie(movie, mask=mask)
        
        assert leveled.shape == movie.shape
        assert not np.isnan(leveled).any()
    
    def test_batch_level_invalid_ndim(self):
        """Test error on non-3D input."""
        img_2d = np.random.rand(100, 100)
        
        with pytest.raises(ValueError, match="Expected 3D movie"):
            batch_level_movie(img_2d)
