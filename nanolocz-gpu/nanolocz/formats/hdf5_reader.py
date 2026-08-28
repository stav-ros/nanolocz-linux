"""
HDF5 file format reader for NanoLocz.

Supports HDF5-based AFM data formats including JPK .h5-jpk files.
"""

import numpy as np
import h5py
from pathlib import Path


def read_h5_afm(filepath, channel='height', frames='all'):
    """
    Read AFM data from HDF5 file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file
    channel : str, optional
        Channel name to read (default: 'height')
        Common channels: 'height', 'amplitude', 'phase', 'error'
    frames : int or 'all', optional
        Number of frames to load (default: 'all')
        
    Returns
    -------
    data : ndarray
        Image data (2D or 3D array)
    metadata : dict
        Image metadata including pixel size, scan size, etc.
        
    Examples
    --------
    >>> data, meta = read_h5_afm('sample.h5')
    >>> print(f"Image shape: {data.shape}")
    >>> print(f"Pixel size: {meta.get('pixel_size_nm', 'N/A')}")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        # Try to find the requested channel
        if channel in f:
            dataset = f[channel]
        else:
            # Search for channel in nested groups
            dataset = _find_channel(f, channel)
            if dataset is None:
                available = _list_channels(f)
                raise ValueError(
                    f"Channel '{channel}' not found. Available: {available}"
                )
        
        # Load data
        if frames == 'all':
            data = dataset[:]
        else:
            data = dataset[:frames]
        
        # Extract metadata
        metadata = _extract_h5_metadata(f, filepath)
    
    return data, metadata


def _find_channel(h5file, channel_name):
    """Recursively search for a channel in HDF5 file."""
    def search_group(group, name):
        if name in group:
            item = group[name]
            if isinstance(item, h5py.Dataset):
                return item
        
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                result = search_group(item, name)
                if result is not None:
                    return result
        return None
    
    return search_group(h5file, channel_name)


def _list_channels(h5file):
    """List all datasets in HDF5 file."""
    channels = []
    
    def collect_datasets(group, path=''):
        for key in group.keys():
            item = group[key]
            full_path = f"{path}/{key}" if path else key
            if isinstance(item, h5py.Dataset):
                channels.append(full_path)
            elif isinstance(item, h5py.Group):
                collect_datasets(item, full_path)
    
    collect_datasets(h5file)
    return channels


def _extract_h5_metadata(h5file, filepath):
    """Extract metadata from HDF5 file attributes."""
    metadata = {
        'filepath': str(filepath),
        'format': 'HDF5',
    }
    
    # Extract top-level attributes
    for key, value in h5file.attrs.items():
        try:
            # Convert numpy types to Python types for JSON serialization
            if hasattr(value, 'item'):
                metadata[key] = value.item()
            else:
                metadata[key] = str(value)
        except Exception:
            pass
    
    # Try to get pixel size info from common attribute names
    pixel_size_attrs = ['pixel_size', 'pixel_size_nm', 'x_pixel_size', 
                        'y_pixel_size', 'Resolution']
    for attr in pixel_size_attrs:
        if attr in h5file.attrs:
            metadata['pixel_size_nm'] = float(h5file.attrs[attr])
            break
    
    # Try to get scan size
    scan_size_attrs = ['scan_size', 'scan_size_nm', 'image_size', 
                       'x_scan_size', 'y_scan_size']
    for attr in scan_size_attrs:
        if attr in h5file.attrs:
            metadata['scan_size_nm'] = float(h5file.attrs[attr])
            break
    
    return metadata


def write_h5(data, filepath, metadata=None, compression='gzip'):
    """
    Write image data to HDF5 file.
    
    Parameters
    ----------
    data : ndarray
        Image data (2D or 3D array)
    filepath : str or Path
        Output file path
    metadata : dict, optional
        Metadata to save as attributes
    compression : str, optional
        Compression method ('gzip', 'lzf', etc.)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Create main dataset
        dset = f.create_dataset('height', data=data, compression=compression)
        
        # Add metadata as attributes
        if metadata:
            for key, value in metadata.items():
                try:
                    f.attrs[key] = value
                except Exception:
                    # Skip attributes that can't be stored
                    pass
    
    return filepath
