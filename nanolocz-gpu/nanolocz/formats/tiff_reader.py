"""
TIFF file format reader for NanoLocz.

Supports multi-page TIFF stacks commonly used in AFM imaging.
"""

import numpy as np
from tifffile import imread, imwrite
from pathlib import Path


def read_tiff(filepath, frames='all', channel=0):
    """
    Read AFM data from TIFF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to TIFF file
    frames : int or 'all', optional
        Number of frames to load (default: 'all')
    channel : int, optional
        Channel index to load for multi-channel files (default: 0)
        
    Returns
    -------
    data : ndarray
        Image data (2D or 3D array)
    metadata : dict
        Image metadata including:
        - shape: Image dimensions
        - dtype: Data type
        - filepath: Source file path
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"TIFF file not found: {filepath}")
    
    # Read all pages/metadata
    with imread(filepath, aszarr=True) as zarr:
        # Get shape info
        shape = zarr.shape
        
        # Load requested frames
        if frames == 'all':
            data = zarr[:]
        else:
            data = zarr[:frames]
    
    # Extract metadata
    metadata = {
        'shape': data.shape,
        'dtype': str(data.dtype),
        'filepath': str(filepath),
        'format': 'TIFF',
        'num_frames': data.shape[0] if data.ndim == 3 else 1,
    }
    
    # Try to get additional metadata from TIFF tags
    try:
        import tifffile
        with tifffile.TiffFile(filepath) as tif:
            if tif.pages and len(tif.pages) > 0:
                page = tif.pages[0]
                # Extract common AFM metadata if available
                if hasattr(page, 'tags'):
                    tags = page.tags
                    # Check for common metadata fields
                    for tag_name in ['XResolution', 'YResolution', 'Software', 
                                     'DateTime', 'ImageDescription']:
                        if tag_name in tags:
                            metadata[tag_name.lower()] = str(tags[tag_name].value)
    except Exception as e:
        # Metadata extraction failed, continue with basic info
        metadata['warning'] = f"Could not extract full metadata: {str(e)}"
    
    return data, metadata


def write_tiff(data, filepath, metadata=None, compression=None):
    """
    Write image data to TIFF file.
    
    Parameters
    ----------
    data : ndarray
        Image data (2D or 3D array)
    filepath : str or Path
        Output file path
    metadata : dict, optional
        Metadata to save in TIFF tags
    compression : str, optional
        Compression method ('lzma', 'zlib', 'jpeg', etc.)
    """
    filepath = Path(filepath)
    
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata for TIFF tags
    description = None
    if metadata:
        description = str(metadata)
    
    # Write TIFF
    imwrite(
        filepath,
        data,
        description=description,
        compression=compression,
        metadata=metadata if metadata else {}
    )
    
    return filepath
