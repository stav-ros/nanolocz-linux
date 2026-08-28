"""
Unit tests for file I/O operations.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from nanolocz.formats.tiff_reader import read_tiff, write_tiff
from nanolocz.formats.hdf5_reader import read_h5_afm, write_h5


def test_tiff_roundtrip():
    """Test TIFF read/write roundtrip."""
    # Create test data
    data = np.random.rand(10, 64, 64).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.tif"
        
        # Write
        write_tiff(data, filepath)
        
        # Read
        loaded_data, metadata = read_tiff(filepath)
        
        # Verify
        assert np.allclose(data, loaded_data), "Data mismatch after roundtrip"
        assert metadata['format'] == 'TIFF'
        assert metadata['num_frames'] == 10


def test_h5_roundtrip():
    """Test HDF5 read/write roundtrip."""
    # Create test data
    data = np.random.rand(5, 32, 32).astype(np.float32)
    metadata_in = {'pixel_size_nm': 1.5, 'scan_size_nm': 50}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.h5"
        
        # Write
        write_h5(data, filepath, metadata=metadata_in)
        
        # Read
        loaded_data, loaded_meta = read_h5_afm(filepath)
        
        # Verify
        assert np.allclose(data, loaded_data), "Data mismatch after roundtrip"
        assert loaded_meta['format'] == 'HDF5'


def test_tiff_single_frame():
    """Test reading single-frame TIFF."""
    data = np.random.rand(128, 128).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "single.tif"
        write_tiff(data, filepath)
        
        loaded_data, metadata = read_tiff(filepath)
        
        assert loaded_data.shape == (128, 128)
        assert metadata['num_frames'] == 1


def test_h5_partial_load():
    """Test loading subset of frames from HDF5."""
    data = np.random.rand(20, 64, 64).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "stack.h5"
        write_h5(data, filepath)
        
        # Load only first 5 frames
        loaded_data, _ = read_h5_afm(filepath, frames=5)
        
        assert loaded_data.shape[0] == 5
        assert np.allclose(data[:5], loaded_data)


def test_file_not_found():
    """Test error handling for missing files."""
    with pytest.raises(FileNotFoundError):
        read_tiff("/nonexistent/path/file.tif")
    
    with pytest.raises(FileNotFoundError):
        read_h5_afm("/nonexistent/path/file.h5")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
