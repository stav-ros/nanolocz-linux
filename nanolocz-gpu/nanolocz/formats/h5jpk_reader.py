"""
JPK (.h5-jpk) file format reader for NanoLocz.

Supports JPK Instruments HDF5-based AFM file format.
"""

import numpy as np
from pathlib import Path
from typing import Any

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


def read_h5jpk(filepath, channel='topography', frame='all'):
    """
    Read AFM data from JPK .h5-jpk file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .h5-jpk file
    channel : str, optional
        Channel name to load (default: 'topography')
        Common channels: 'topography', 'deflection', 'phase', 'amplitude'
    frame : int or 'all', optional
        For multi-frame/time-series files, specify which frame(s) to load.
        Use integer index or 'all' for all frames.
        
    Returns
    -------
    data : ndarray
        Image data (2D or 3D array)
    metadata : dict
        Image metadata including:
        - shape: Image dimensions
        - dtype: Data type
        - filepath: Source file path
        - pixel_size: Physical pixel size if available
        - units: Physical units if available
        - channel_name: Name of the loaded channel
        - scan_size: Scan area dimensions
        - jpk_metadata: Raw JPK metadata
        
    Raises
    ------
    ImportError
        If h5py package is not installed
    FileNotFoundError
        If file does not exist
    ValueError
        If file format is invalid or channel not found
        If file is not a valid JPK HDF5 file
        
    Examples
    --------
    >>> from nanolocz.formats import read_h5jpk
    >>> data, meta = read_h5jpk("sample.h5-jpk")
    >>> print(f"Shape: {data.shape}, Units: {meta.get('units')}")
    
    Load specific channel:
    
    >>> data, meta = read_h5jpk("sample.h5-jpk", channel="deflection")
    """
    if not H5PY_AVAILABLE:
        raise ImportError(
            "h5py package required for .h5-jpk support. "
            "Install with: pip install h5py"
        )
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JPK file not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    if suffix not in ['.h5-jpk', '.h5', '.jpks', '.jpk']:
        raise ValueError(
            f"Expected .h5-jpk file, got: {suffix}. "
            f"JPK files typically have .h5-jpk extension."
        )
    
    # Open HDF5 file
    try:
        with h5py.File(str(filepath), 'r') as f:
            return _extract_jpk_data(f, channel, frame, str(filepath))
    except Exception as e:
        if "not an HDF5 file" in str(e):
            raise ValueError(f"Not a valid HDF5 file: {e}")
        raise


def _extract_jpk_data(h5file, channel, frame, filepath_str):
    """
    Extract data from JPK HDF5 structure.
    
    JPK files have various structures depending on instrument and software version.
    Common structures:
    - /data/channel_0/topography
    - /measurement/data/channel_0/topography
    - /raw/channel_0/height
    """
    # Try to find the data in common JPK locations
    data_path = None
    possible_paths = [
        f'/data/{channel}',
        f'/data/channel_0/{channel}',
        f'/measurement/data/{channel}',
        f'/measurement/data/channel_0/{channel}',
        f'/raw/{channel}',
        f'/raw/channel_0/{channel}',
        f'/{channel}',
    ]
    
    # Also try common variations
    channel_variants = [channel, channel.lower(), channel.upper()]
    data_variants = ['topography', 'height', 'data', 'image', 'z_sensor']
    
    for base_channel in channel_variants:
        for data_name in data_variants:
            possible_paths.extend([
                f'/data/{base_channel}/{data_name}',
                f'/data/channel_0/{base_channel}/{data_name}',
                f'/measurement/data/{base_channel}/{data_name}',
            ])
    
    # Search for the data
    found_path = None
    for path in possible_paths:
        if path in h5file:
            obj = h5file[path]
            if isinstance(obj, h5py.Dataset):
                found_path = path
                break
    
    if found_path is None:
        # List available datasets to help user
        available = []
        def visit(name):
            obj = h5file[name]
            if isinstance(obj, h5py.Dataset):
                available.append(name)
        h5file.visit(visit)
        
        raise ValueError(
            f"Channel '{channel}' not found in {filepath_str}.\n"
            f"Available datasets: {available[:10]}{'...' if len(available) > 10 else ''}"
        )
    
    # Load the data
    dataset = h5file[found_path]
    data = np.array(dataset[:])
    
    # Handle frame selection for 3D data
    if data.ndim == 3 and frame != 'all':
        if isinstance(frame, int):
            data = data[frame:frame+1]  # Keep dimensionality
        else:
            # Assume slice or list
            data = data[frame]
    
    # Extract metadata
    metadata = {
        'shape': data.shape,
        'dtype': str(data.dtype),
        'filepath': filepath_str,
        'format': 'H5-JPK',
        'channel': channel,
        'dataset_path': found_path,
    }
    
    # Try to extract physical calibration
    # JPK stores calibration in various places
    calib_paths = [
        '/data/calibration',
        '/measurement/calibration',
        '/calibration',
        '/data/scan',
        '/measurement/data/scan',
    ]
    
    for calib_path in calib_paths:
        if calib_path in h5file:
            calib_group = h5file[calib_path]
            
            # Look for size/pixel information
            for key in ['x_size', 'y_size', 'scan_size', 'width', 'height']:
                if key in calib_group:
                    val = calib_group[key][()]
                    if isinstance(val, np.ndarray):
                        val = val.item()
                    metadata[f'{key}'] = val
            
            # Look for units
            for key in ['unit', 'units', 'z_unit', 'xy_unit']:
                if key in calib_group:
                    val = calib_group[key][()]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    elif isinstance(val, np.ndarray):
                        val = val.item()
                    metadata['units'] = val
            
            # Calculate pixel size if we have scan size and image dimensions
            if 'x_size' in metadata and 'y_size' in metadata:
                x_size = metadata['x_size']
                y_size = metadata['y_size']
                if data.ndim >= 2:
                    ny, nx = data.shape[-2:]
                    metadata['pixel_size'] = (x_size / nx, y_size / ny)
                    metadata['pixel_size_units'] = metadata.get('units', 'unknown')
            break
    
    # Extract additional metadata from attributes
    if hasattr(dataset, 'attrs') and len(dataset.attrs) > 0:
        metadata['dataset_attrs'] = dict(dataset.attrs)
    
    # Try to get instrument metadata
    instrument_paths = ['/instrument', '/measurement/instrument']
    for inst_path in instrument_paths:
        if inst_path in h5file:
            inst_group = h5file[inst_path]
            inst_meta = {}
            for key in inst_group.keys():
                try:
                    val = inst_group[key][()]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    elif isinstance(val, np.ndarray) and val.ndim == 0:
                        val = val.item()
                    inst_meta[key] = val
                except:
                    pass
            if inst_meta:
                metadata['instrument'] = inst_meta
            break
    
    # Store reference to full metadata for advanced users
    metadata['jpk_metadata'] = {
        'available_datasets': [k for k in h5file.keys()],
        'channel_path': found_path,
    }
    
    return data, metadata


def write_h5jpk(data, filepath, metadata=None, pixel_size=None, units='nm', channel_name='topography'):
    """
    Write image data to JPK .h5-jpk format.
    
    Parameters
    ----------
    data : ndarray
        Image data (2D or 3D array)
    filepath : str or Path
        Output file path
    metadata : dict, optional
        Additional metadata to store as attributes
    pixel_size : tuple, optional
        Physical pixel size (x, y) in specified units
    units : str, optional
        Physical units (default: 'nm')
    channel_name : str, optional
        Name for the data channel (default: 'topography')
        
    Returns
    -------
    filepath : Path
        Path to written file
        
    Notes
    -----
    This creates a simplified HDF5 structure compatible with JPK readers.
    For full JPK compatibility with all metadata, use JPK's own tools.
    """
    if not H5PY_AVAILABLE:
        raise ImportError(
            "h5py package required for .h5-jpk support. "
            "Install with: pip install h5py"
        )
    
    filepath = Path(filepath)
    
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure proper extension
    if filepath.suffix.lower() not in ['.h5-jpk', '.h5']:
        filepath = filepath.with_suffix('.h5-jpk')
    
    # Ensure 2D or 3D data
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Add frame dimension
    elif data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    
    try:
        with h5py.File(str(filepath), 'w') as f:
            # Create standard JPK-like structure
            data_group = f.create_group('data/channel_0')
            
            # Create dataset
            dset = data_group.create_dataset(channel_name, data=data)
            
            # Add basic attributes
            dset.attrs['units'] = units
            dset.attrs['created_by'] = 'nanolocz'
            
            # Add pixel size calibration
            if pixel_size is not None:
                ny, nx = data.shape[-2:]
                x_size = nx * pixel_size[0]
                y_size = ny * pixel_size[1]
                
                calib_group = f.create_group('data/calibration')
                calib_group.attrs['x_size'] = x_size
                calib_group.attrs['y_size'] = y_size
                calib_group.attrs['unit'] = units
                calib_group.attrs['pixel_size_x'] = pixel_size[0]
                calib_group.attrs['pixel_size_y'] = pixel_size[1]
            
            # Add user metadata
            if metadata:
                for key, value in metadata.items():
                    try:
                        dset.attrs[key] = value
                    except (TypeError, ValueError):
                        # Skip non-serializable metadata
                        pass
                        
    except Exception as e:
        raise IOError(f"Failed to write JPK file: {e}")
    
    return filepath


__all__ = ['read_h5jpk', 'write_h5jpk']
