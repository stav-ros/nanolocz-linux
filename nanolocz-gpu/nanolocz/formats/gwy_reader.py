"""
Gwyddion (.gwy) file format reader for NanoLocz.

Supports Gwyddion native format commonly used in AFM imaging.
"""

import numpy as np
from pathlib import Path
from typing import Any

try:
    import gwyfile
    from gwyfile import objects
    GWYFILE_AVAILABLE = True
except ImportError:
    GWYFILE_AVAILABLE = False


def read_gwy(filepath, channel=0, frame='all'):
    """
    Read AFM data from Gwyddion .gwy file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .gwy file
    channel : int, optional
        Channel/data field index to load (default: 0)
        Gwyddion files can contain multiple data fields/channels
    frame : int or 'all', optional
        For multi-frame files, specify which frame(s) to load.
        Currently supports loading a single frame or all frames.
        
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
        - gwy_metadata: Raw Gwyddion metadata
        
    Raises
    ------
    ImportError
        If gwyfile package is not installed
    FileNotFoundError
        If file does not exist
    ValueError
        If file format is invalid or channel not found
        
    Examples
    --------
    >>> from nanolocz.formats import read_gwy
    >>> data, meta = read_gwy("sample.gwy")
    >>> print(f"Shape: {data.shape}, Units: {meta.get('units')}")
    
    Load specific channel:
    
    >>> data, meta = read_gwy("sample.gwy", channel=1)
    """
    if not GWYFILE_AVAILABLE:
        raise ImportError(
            "gwyfile package required for .gwy support. "
            "Install with: pip install gwyfile"
        )
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Gwyddion file not found: {filepath}")
    
    if filepath.suffix.lower() != '.gwy':
        raise ValueError(f"Expected .gwy file, got: {filepath.suffix}")
    
    # Load the Gwyddion container
    try:
        gwy_container = gwyfile.load(str(filepath))
    except Exception as e:
        raise ValueError(f"Failed to parse Gwyddion file: {e}")
    
    # Extract data fields from container
    # Gwyddion stores data in channels like "/0/data", "/1/data", etc.
    data_fields = []
    channel_names = []
    
    for key in gwy_container.keys():
        if '/data' in key:
            obj = gwy_container[key]
            if isinstance(obj, objects.GwyDataField):
                data_fields.append((key, obj))
                channel_names.append(key)
    
    if not data_fields:
        raise ValueError(f"No data fields found in {filepath}")
    
    if channel < 0 or channel >= len(data_fields):
        raise ValueError(
            f"Channel {channel} out of range. "
            f"Available channels: 0-{len(data_fields)-1}"
        )
    
    # Get the requested data field
    field_key, data_field = data_fields[channel]
    
    # Extract data array from GwyDataField
    # GwyDataField has methods to get data as numpy array
    try:
        # Get dimensions
        xres = data_field.xres
        yres = data_field.yres
        xreal = data_field.xreal
        yreal = data_field.yreal
        
        # Get data as numpy array
        data = np.array(data_field.get_data()).reshape((yres, xres))
        
        # Handle multi-frame if requested
        if frame != 'all':
            # For now, single frame support - future enhancement for volumes
            pass
            
    except Exception as e:
        raise ValueError(f"Failed to extract data from channel {channel}: {e}")
    
    # Extract metadata
    metadata = {
        'shape': data.shape,
        'dtype': str(data.dtype),
        'filepath': str(filepath),
        'format': 'GWY',
        'channel': channel,
        'channel_name': field_key,
        'xres': xres,
        'yres': yres,
        'xreal': xreal,
        'yreal': yreal,
    }
    
    # Extract physical units if available
    if hasattr(data_field, 'si') and data_field.si is not None:
        si = data_field.si
        metadata['units'] = getattr(si, 'unitstr', 'unknown')
        metadata['valuenames'] = getattr(si, 'valuenames', '')
        
        # Calculate pixel size
        if xreal > 0 and xres > 0:
            metadata['pixel_size'] = (xreal / xres, yreal / yres)
            metadata['pixel_size_units'] = metadata.get('units', 'unknown')
    
    # Store additional Gwyddion metadata
    metadata['gwy_metadata'] = {
        'title': getattr(gwy_container.get('/0/title', None), 'get', lambda: '')(),
        'comment': getattr(gwy_container.get('/0/comment', None), 'get', lambda: '')(),
    }
    
    return data, metadata


def write_gwy(data, filepath, metadata=None, pixel_size=None, units='nm'):
    """
    Write image data to Gwyddion .gwy file.
    
    Parameters
    ----------
    data : ndarray
        Image data (2D or 3D array). If 3D, only first frame is saved.
    filepath : str or Path
        Output file path
    metadata : dict, optional
        Additional metadata to store
    pixel_size : tuple, optional
        Physical pixel size (x, y) in specified units
    units : str, optional
        Physical units (default: 'nm')
        
    Returns
    -------
    filepath : Path
        Path to written file
        
    Notes
    -----
    Writing Gwyddion files requires careful construction of the GwyContainer
    with proper GwyDataField and GwySIUnit objects. This function provides
    basic support; for advanced features, use gwyfile directly.
    """
    if not GWYFILE_AVAILABLE:
        raise ImportError(
            "gwyfile package required for .gwy support. "
            "Install with: pip install gwyfile"
        )
    
    filepath = Path(filepath)
    
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure 2D data
    if data.ndim == 3:
        data = data[0]  # Take first frame
    elif data.ndim != 2:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    
    # Create GwyDataField - gwyfile expects 2D numpy array
    data_field = objects.GwyDataField(data)
    
    # Set physical dimensions if provided
    if pixel_size is not None:
        yres, xres = data.shape
        xreal = xres * pixel_size[0]
        yreal = yres * pixel_size[1]
        data_field.xreal = xreal
        data_field.yreal = yreal
    
    # Create SI unit information
    si_unit = objects.GwySIUnit()
    si_unit.unitstr = units
    si_unit.valuenames = ""
    data_field.si = si_unit
    
    # Create container and add data field
    container = objects.GwyContainer()
    container['/0/data'] = data_field
    
    # Add title/metadata if provided
    if metadata:
        if 'title' in metadata:
            title_obj = type('obj', (object,), {'get': lambda self: metadata['title']})()
            container['/0/title'] = title_obj
        if 'comment' in metadata:
            comment_obj = type('obj', (object,), {'get': lambda self: metadata['comment']})()
            container['/0/comment'] = comment_obj
    
    # Serialize and write
    try:
        buffer = objects.serialize_component(container)
        with open(filepath, 'wb') as f:
            f.write(buffer)
    except Exception as e:
        raise IOError(f"Failed to write Gwyddion file: {e}")
    
    return filepath


__all__ = ['read_gwy', 'write_gwy']
