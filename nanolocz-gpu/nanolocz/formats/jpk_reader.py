"""Reader for JPK image files with the ``.jpk`` extension.

Supports both:
- Legacy binary JPK format (older instruments)
- HDF5-based JPK format (newer instruments, also .h5-jpk, .jpks)
"""

import struct
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Union

from nanolocz.formats.h5jpk_reader import read_h5jpk, write_h5jpk


def _is_hdf5_file(filepath: Path) -> bool:
    """Check if file has HDF5 signature."""
    try:
        with open(filepath, 'rb') as f:
            signature = f.read(8)
            # HDF5 files start with b'\x89HDF\r\n\x1a\n'
            return signature == b'\x89HDF\r\n\x1a\n'
    except Exception:
        return False


def _read_legacy_jpk(filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read legacy binary JPK file format.
    
    JPK binary files have a header followed by raw image data.
    The header contains metadata about image dimensions, pixel size, etc.
    
    Parameters
    ----------
    filepath : Path
        Path to .jpk file
        
    Returns
    -------
    data : ndarray
        Image data (2D array)
    metadata : dict
        Image metadata
    """
    with open(filepath, 'rb') as f:
        # Read header - JPK binary format varies by instrument version
        # Common structure: 512-byte header followed by data
        
        header_size = 512
        header = f.read(header_size)
        
        # Try to extract basic info from header
        # This is a simplified parser - full spec would need reverse engineering
        # or JPK's documentation
        
        # Look for common markers in header
        metadata = {
            'filepath': str(filepath),
            'format': 'JPK-Binary',
            'header_size': header_size,
        }
        
        # Try to detect image dimensions from header
        # Often stored at specific offsets in JPK binary files
        width = None
        height = None
        pixel_size = None
        
        # Common offsets for dimensions in JPK files (varies by software version)
        # These are educated guesses based on common patterns
        try:
            # Try offset 0x3C (60) for width, 0x40 (64) for height (common in older files)
            if len(header) >= 68:
                w_candidate = struct.unpack('<I', header[60:64])[0]
                h_candidate = struct.unpack('<I', header[64:68])[0]
                
                # Sanity check: dimensions should be reasonable for AFM images
                if 16 <= w_candidate <= 8192 and 16 <= h_candidate <= 8192:
                    width = w_candidate
                    height = h_candidate
                    
            # Alternative: try offset 0x20 (32) and 0x24 (36)
            if width is None and len(header) >= 40:
                w_candidate = struct.unpack('<I', header[32:36])[0]
                h_candidate = struct.unpack('<I', header[36:40])[0]
                
                if 16 <= w_candidate <= 8192 and 16 <= h_candidate <= 8192:
                    width = w_candidate
                    height = h_candidate
                    
        except Exception:
            pass
        
        # If we couldn't determine dimensions, try common sizes
        if width is None or height is None:
            # Get file size to estimate dimensions
            f.seek(0, 2)  # End of file
            file_size = f.tell()
            data_size = file_size - header_size
            
            # Try common AFM image sizes
            common_sizes = [
                (256, 256), (512, 512), (1024, 1024),
                (256, 128), (512, 256), (1024, 512),
                (128, 128), (64, 64),
            ]
            
            for w, h in common_sizes:
                # Try float32 (4 bytes per pixel)
                if w * h * 4 == data_size:
                    width, height = w, h
                    break
                # Try float64 (8 bytes per pixel)
                elif w * h * 8 == data_size:
                    width, height = w, h
                    break
                # Try int16 (2 bytes per pixel)
                elif w * h * 2 == data_size:
                    width, height = w, h
                    break
        
        if width is None or height is None:
            raise ValueError(
                f"Could not determine image dimensions for {filepath}. "
                f"File size: {file_size} bytes. "
                f"This may be an unsupported JPK binary format. "
                f"Try converting to HDF5 format using JPK's software."
            )
        
        metadata['width'] = width
        metadata['height'] = height
        
        # Determine data type by trying common formats
        f.seek(header_size)
        data_size = file_size - header_size
        expected_pixels = width * height
        
        # Try float32 first (most common)
        if data_size == expected_pixels * 4:
            dtype = np.float32
            data = np.fromfile(f, dtype=dtype, count=expected_pixels)
        # Try float64
        elif data_size == expected_pixels * 8:
            dtype = np.float64
            data = np.fromfile(f, dtype=dtype, count=expected_pixels)
        # Try int16
        elif data_size == expected_pixels * 2:
            dtype = np.int16
            data = np.fromfile(f, dtype=dtype, count=expected_pixels)
        # Try uint16
        elif data_size == expected_pixels * 2:
            dtype = np.uint16
            data = np.fromfile(f, dtype=dtype, count=expected_pixels)
        else:
            raise ValueError(
                f"Unknown data format: {data_size} bytes for {expected_pixels} pixels. "
                f"Bytes per pixel: {data_size / expected_pixels if expected_pixels > 0 else 'N/A'}"
            )
        
        # Reshape to 2D image (row-major order assumed)
        data = data.reshape((height, width))
        
        # Try to extract pixel size from header (if present)
        # Often stored in nanometers or micrometers
        try:
            # Try common offsets for pixel size (varies by version)
            if len(header) >= 100:
                # Check for scan size information
                scan_x = struct.unpack('<d', header[80:88])[0]  # Double precision
                scan_y = struct.unpack('<d', header[88:96])[0]
                
                if 1e-9 < scan_x < 1e-3:  # Reasonable scan size in meters
                    pixel_size = (scan_x / width, scan_y / height)
                    metadata['pixel_size'] = pixel_size
                    metadata['pixel_size_units'] = 'm'
                    metadata['scan_size_x'] = scan_x
                    metadata['scan_size_y'] = scan_y
        except Exception:
            pass
        
        metadata['shape'] = data.shape
        metadata['dtype'] = str(data.dtype)
        
        return data, metadata


def read_jpk(filepath: Union[str, Path], channel: str = "topography", frame: str = "all"):
    """
    Read a JPK image file.
    
    Automatically detects whether the file is HDF5-based or legacy binary format.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .jpk file
    channel : str, optional
        Channel name (only used for HDF5 format, default: 'topography')
        Common channels: 'topography', 'deflection', 'phase', 'amplitude'
    frame : int or 'all', optional
        For multi-frame/time-series files (only HDF5 format)
        
    Returns
    -------
    data : ndarray
        Image data (2D or 3D array)
    metadata : dict
        Image metadata including:
        - shape: Image dimensions
        - dtype: Data type
        - filepath: Source file path
        - format: 'JPK' or 'JPK-Binary' or 'H5-JPK'
        - pixel_size: Physical pixel size if available
        - units: Physical units if available
        
    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If file format is not recognized or corrupted
        
    Examples
    --------
    >>> from nanolocz.formats import read_jpk
    >>> data, meta = read_jpk("sample.jpk")
    >>> print(f"Shape: {data.shape}, Format: {meta['format']}")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JPK file not found: {filepath}")
    
    # Check if it's HDF5 format
    if _is_hdf5_file(filepath):
        data, metadata = read_h5jpk(str(filepath), channel=channel, frame=frame)
        metadata["format"] = "H5-JPK"
        return data, metadata
    else:
        # It's a legacy binary JPK file
        if channel != "topography" or frame != "all":
            import warnings
            warnings.warn(
                f"Channel and frame parameters are ignored for legacy binary JPK files. "
                f"These parameters only work with HDF5-based JPK files."
            )
        return _read_legacy_jpk(filepath)


def write_jpk(data, filepath, **kwargs):
    """Write the portable JPK-compatible HDF5 representation."""
    return write_h5jpk(data, filepath, **kwargs)
