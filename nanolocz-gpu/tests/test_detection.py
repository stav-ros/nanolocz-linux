"""
Unit tests for particle detection module.
"""

import numpy as np
import pytest
from nanolocz.core.detection import fast_peaks2d, detect_particles


def test_fast_peaks2d_basic():
    """Test basic peak detection functionality."""
    # Create simple test image with known peaks
    img = np.zeros((50, 50))
    img[10, 10] = 1.0
    img[30, 30] = 0.8
    img[40, 20] = 0.6
    
    # Add some noise
    img += np.random.normal(0, 0.05, img.shape)
    
    # Detect peaks
    peaks = fast_peaks2d(img, thresh=0.3, kernel_size=3)
    
    # Should find at least 2 peaks
    assert len(peaks) >= 2, f"Expected at least 2 peaks, found {len(peaks)}"
    
    # Check peak format (x, y, height, prominence)
    assert peaks.shape[1] == 4, "Peaks should have 4 columns"


def test_fast_peaks2d_gpu():
    """Test peak detection with GPU acceleration (if available)."""
    img = np.random.rand(100, 100)
    
    # Run on CPU
    peaks_cpu = fast_peaks2d(img, thresh=0.5, kernel_size=3, use_gpu=False)
    
    # Run on GPU (will fallback to CPU if CuPy not available)
    peaks_gpu = fast_peaks2d(img, thresh=0.5, kernel_size=3, use_gpu=True)
    
    # Results should be similar
    assert len(peaks_cpu) == len(peaks_gpu), \
        f"CPU found {len(peaks_cpu)} peaks, GPU found {len(peaks_gpu)}"


def test_detect_particles_direct():
    """Test direct particle detection mode."""
    img = np.random.rand(100, 100)
    
    result = detect_particles(img, method='direct', thresh=0.7)
    
    assert 'locs' in result
    assert result['locs'] is not None or len(result['locs']) == 0


def test_detect_particles_crosscorr():
    """Test cross-correlation particle detection mode."""
    # Create test image and reference
    img = np.random.rand(100, 100)
    ref = np.random.rand(20, 20)
    
    result = detect_particles(
        img, 
        method='crosscorr', 
        ref_img=ref,
        thresh=0.5
    )
    
    assert 'locs' in result
    assert 'scores' in result
    assert 'angle' in result


def test_detect_particles_invalid_method():
    """Test error handling for invalid detection method."""
    img = np.random.rand(50, 50)
    
    with pytest.raises(ValueError):
        detect_particles(img, method='invalid_method')


def test_empty_result():
    """Test detection with no peaks above threshold."""
    img = np.zeros((50, 50)) + 0.1  # Low uniform signal
    
    peaks = fast_peaks2d(img, thresh=0.9, kernel_size=3)
    
    assert len(peaks) == 0
    assert peaks.shape == (0, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
