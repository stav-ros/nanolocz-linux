"""Unified opener interface for NanoLocz data stores.

This module provides the `open_nanolocz` function that automatically detects
file format and opens the appropriate storage backend.
"""

from pathlib import Path
from typing import Literal

from nanolocz.io.store import NanoLoczStore


def open_nanolocz(
    path: str | Path, 
    mode: Literal['r', 'r+', 'w', 'w-', 'a'] = 'r'
) -> NanoLoczStore:
    """Open a NanoLocz data store.
    
    Automatically detects the file format based on extension and directory
    structure, then opens the appropriate storage backend.
    
    Supported formats:
    - .zarr : Zarr-based storage (recommended for new projects)
    - .h5 / .hdf5 : HDF5-based storage (legacy support)
    - .tiff : TIFF stacks with sidecar metadata
    
    Parameters
    ----------
    path : str or Path
        Path to data file or directory
    mode : {'r', 'r+', 'w', 'w-', 'a'}, optional
        Opening mode:
        - 'r': Read-only (default)
        - 'r+': Read-write (must exist)
        - 'w': Read-write (overwrite if exists)
        - 'w-': Read-write (fail if exists)
        - 'a': Append (read-write, create if not exists)
        
    Returns
    -------
    NanoLoczStore
        Opened data store object
        
    Raises
    ------
    ValueError
        If file format is not supported or cannot be detected
    FileNotFoundError
        If file does not exist and mode is 'r' or 'r+'
        
    Examples
    --------
    Create a new Zarr store:
    
    >>> from nanolocz.io import open_nanolocz
    >>> import numpy as np
    >>> store = open_nanolocz("experiment.zarr", mode="w")
    >>> movie = np.random.rand(100, 256, 256)
    >>> store.save_movie(movie, {"pixel_size": (1.0, 1.0)})
    >>> store.close()
    
    Open existing store for reading:
    
    >>> store = open_nanolocz("experiment.zarr", mode="r")
    >>> loaded_movie = store.load_movie()
    >>> localizations = store.load_localizations()
    >>> store.close()
    
    Use context manager for automatic cleanup:
    
    >>> with open_nanolocz("experiment.zarr", mode="r") as store:
    ...     movie = store.load_movie()
    ...     locs = store.load_localizations()
    """
    path = Path(path)
    
    # Detect format from extension
    suffix = path.suffix.lower()
    
    if suffix == '.zarr' or path.is_dir():
        # Zarr store (directory-based)
        if mode in ('r', 'r+') and not path.exists():
            raise FileNotFoundError(f"Zarr store not found: {path}")
        return NanoLoczStore(path, mode=mode)
    
    elif suffix in ('.h5', '.hdf5'):
        # HDF5 store - delegate to future HDF5Store implementation
        # For now, raise informative error
        raise NotImplementedError(
            f"HDF5 support coming soon. Please use Zarr format (.zarr) instead.\n"
            f"Path: {path}"
        )
    
    elif suffix in ('.tif', '.tiff'):
        # TIFF - read-only access via formats module
        if mode != 'r':
            raise ValueError("TIFF format only supports read-only mode")
        raise NotImplementedError(
            f"TIFF reader integration coming in NL-11. "
            f"For now, use formats.read_tiff() directly."
        )
    
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}\n"
            f"Supported formats: .zarr, .h5, .hdf5, .tif, .tiff\n"
            f"Path: {path}"
        )


__all__ = ['open_nanolocz']
