"""
Napari-compatible file readers for NanoLocz.

These functions wrap the nanolocz format readers and return data in the
format expected by napari (LayerDataTuple).
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    from napari.types import LayerDataTuple
except ImportError:
    # Fallback type definition if napari is not installed
    LayerDataTuple = tuple


def _prepare_layer_data(data: np.ndarray, metadata: dict) -> LayerDataTuple:
    """
    Prepare data and metadata for napari layer creation.
    
    Parameters
    ----------
    data : ndarray
        Image data
    metadata : dict
        Metadata from reader
        
    Returns
    -------
    LayerDataTuple
        Tuple of (data, metadata_dict, layer_type)
    """
    # Ensure data is 2D or 3D for image layer
    if data.ndim > 3:
        # Take first few frames if too many dimensions
        data = data[:10] if data.shape[0] > 10 else data
    
    # Prepare napari-compatible metadata
    napari_meta = {
        'name': Path(metadata.get('filepath', '')).stem,
        'colormap': 'gray',
        'contrast_limits': [float(np.percentile(data, 1)), float(np.percentile(data, 99))],
    }
    
    # Add physical calibration if available
    if 'pixel_size' in metadata:
        pixel_size = metadata['pixel_size']
        if isinstance(pixel_size, (tuple, list)) and len(pixel_size) >= 2:
            napari_meta['scale'] = [float(pixel_size[1]), float(pixel_size[0])]
            if 'pixel_size_units' in metadata:
                napari_meta['unit'] = metadata['pixel_size_units']
    
    # Add other metadata
    for key in ['channel', 'units', 'format', 'dtype']:
        if key in metadata:
            napari_meta[key] = metadata[key]
    
    return (data, napari_meta, 'image')


def napari_read_tiff(filepath: str) -> List[LayerDataTuple]:
    """
    Napari reader for TIFF files.
    
    Parameters
    ----------
    filepath : str
        Path to TIFF file
        
    Returns
    -------
    List[LayerDataTuple]
        List containing single layer data tuple
    """
    from nanolocz.formats import read_tiff
    
    data, metadata = read_tiff(filepath)
    return [_prepare_layer_data(data, metadata)]


def napari_read_h5_afm(filepath: str) -> List[LayerDataTuple]:
    """
    Napari reader for HDF5 AFM files.
    
    Parameters
    ----------
    filepath : str
        Path to HDF5 file
        
    Returns
    -------
    List[LayerDataTuple]
        List containing single layer data tuple
    """
    from nanolocz.formats import read_h5_afm
    
    data, metadata = read_h5_afm(filepath)
    return [_prepare_layer_data(data, metadata)]


def napari_read_gwy(filepath: str) -> List[LayerDataTuple]:
    """
    Napari reader for Gwyddion .gwy files.
    
    Parameters
    ----------
    filepath : str
        Path to .gwy file
        
    Returns
    -------
    List[LayerDataTuple]
        List containing single layer data tuple
    """
    from nanolocz.formats import read_gwy
    
    data, metadata = read_gwy(filepath)
    return [_prepare_layer_data(data, metadata)]


def napari_read_jpk(filepath: str) -> List[LayerDataTuple]:
    """
    Napari reader for JPK .jpk files.
    
    Parameters
    ----------
    filepath : str
        Path to .jpk file
        
    Returns
    -------
    List[LayerDataTuple]
        List containing single layer data tuple
    """
    from nanolocz.formats import read_jpk
    
    data, metadata = read_jpk(filepath)
    return [_prepare_layer_data(data, metadata)]


def napari_read_spm(filepath: str) -> List[LayerDataTuple]:
    """
    Napari reader for SPM files.
    
    Parameters
    ----------
    filepath : str
        Path to .spm file
        
    Returns
    -------
    List[LayerDataTuple]
        List containing single layer data tuple
    """
    from nanolocz.formats import read_spm
    
    data, metadata = read_spm(filepath)
    return [_prepare_layer_data(data, metadata)]


def napari_read_ibw(filepath: str) -> List[LayerDataTuple]:
    """
    Napari reader for IBW files.
    
    Parameters
    ----------
    filepath : str
        Path to .ibw file
        
    Returns
    -------
    List[LayerDataTuple]
        List containing single layer data tuple
    """
    from nanolocz.formats import read_ibw
    
    data, metadata = read_ibw(filepath)
    return [_prepare_layer_data(data, metadata)]


def napari_read_asd(filepath: str) -> List[LayerDataTuple]:
    """
    Napari reader for ASD files.
    
    Parameters
    ----------
    filepath : str
        Path to .asd file
        
    Returns
    -------
    List[LayerDataTuple]
        List containing single layer data tuple
    """
    from nanolocz.formats import read_asd
    
    data, metadata = read_asd(filepath)
    return [_prepare_layer_data(data, metadata)]


__all__ = [
    'napari_read_tiff',
    'napari_read_h5_afm',
    'napari_read_gwy',
    'napari_read_jpk',
    'napari_read_spm',
    'napari_read_ibw',
    'napari_read_asd',
]
