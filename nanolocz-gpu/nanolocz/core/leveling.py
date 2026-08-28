"""
AFM image leveling module for NanoLocz.

Implements line, plane, and weighted multi-plane leveling algorithms.
Ports MATLAB toolbox functionality to Python with NumPy/CuPy support.
"""

import numpy as np
from typing import Literal, Optional, Tuple
from dataclasses import dataclass

from nanolocz.gpu.utils import get_array_module, GPUArrayModule


@dataclass
class LevelingParams:
    """Parameters for leveling operations."""
    method: Literal['line', 'plane', 'weighted_plane'] = 'plane'
    mask: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    reference_line: int = 0
    polynomial_order: int = 1


def line_leveling(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    reference_line: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Line-by-line leveling (row flattening).
    
    Subtracts the mean or median of each line from that line.
    
    Parameters
    ----------
    image : ndarray
        2D AFM image (rows x columns)
    mask : ndarray, optional
        Boolean mask indicating valid pixels (True = valid)
    reference_line : int
        Index of reference line to preserve (default: 0)
    
    Returns
    -------
    leveled_image : ndarray
        Line-leveled image
    offsets : ndarray
        Per-line offset values that were subtracted
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> leveled, offsets = line_leveling(img)
    >>> assert leveled.shape == img.shape
    """
    xp = get_array_module(False)
    image = xp.asarray(image)
    
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {image.ndim}D")
    
    n_lines, n_cols = image.shape
    offsets = xp.zeros(n_lines)
    leveled = image.copy()
    
    # Create mask if not provided
    if mask is None:
        mask = xp.ones_like(image, dtype=bool)
    else:
        mask = xp.asarray(mask)
    
    # Calculate offset for each line
    for i in range(n_lines):
        line_data = image[i, :]
        line_mask = mask[i, :]
        
        if xp.any(line_mask):
            # Use median for robustness against outliers
            offset = xp.median(line_data[line_mask])
        else:
            offset = 0.0
        
        offsets[i] = offset
        leveled[i, :] = line_data - offset
    
    # Restore reference line to original level
    ref_offset = offsets[reference_line]
    leveled += ref_offset
    offsets -= ref_offset
    
    return xp.asanyarray(leveled), xp.asanyarray(offsets)


def plane_leveling(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, dict]:
    """
    Plane fitting and subtraction (background removal).
    
    Fits a 2D plane z = ax + by + c to the image and subtracts it.
    
    Parameters
    ----------
    image : ndarray
        2D AFM image (rows x columns)
    mask : ndarray, optional
        Boolean mask indicating valid pixels (True = valid)
    weights : ndarray, optional
        Per-pixel weights for weighted least squares
    
    Returns
    -------
    leveled_image : ndarray
        Plane-leveled image
    info : dict
        Dictionary containing:
        - 'plane': fitted plane parameters [a, b, c]
        - 'residual_std': standard deviation of residuals
        - 'r_squared': coefficient of determination
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> leveled, info = plane_leveling(img)
    >>> assert leveled.shape == img.shape
    >>> assert 'plane' in info
    """
    xp = get_array_module(False)
    image = xp.asarray(image, dtype=xp.float64)
    
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {image.ndim}D")
    
    n_rows, n_cols = image.shape
    
    # Create coordinate grids
    y_coords, x_coords = xp.meshgrid(
        xp.arange(n_rows, dtype=xp.float64),
        xp.arange(n_cols, dtype=xp.float64),
        indexing='ij'
    )
    
    # Flatten arrays
    z = image.ravel()
    x = x_coords.ravel()
    y = y_coords.ravel()
    
    # Apply mask if provided
    if mask is not None:
        mask = xp.asarray(mask).ravel()
        valid = mask.astype(bool)
    else:
        valid = xp.ones_like(z, dtype=bool)
    
    # Apply weights if provided
    if weights is not None:
        weights = xp.asarray(weights).ravel()
        w = xp.sqrt(weights[valid])
    else:
        w = xp.ones(xp.sum(valid))
    
    # Build design matrix for plane: z = ax + by + c
    X = xp.column_stack([
        x[valid] * w,
        y[valid] * w,
        w
    ])
    y_obs = z[valid] * w
    
    # Solve least squares
    try:
        params, residuals, rank, s = xp.linalg.lstsq(X, y_obs, rcond=None)
    except Exception:
        # Fallback to simple mean subtraction if fit fails
        params = xp.array([0.0, 0.0, xp.mean(z[valid])])
        residuals = xp.array([0.0])
    
    a, b, c = params
    
    # Calculate fitted plane
    plane = a * x_coords + b * y_coords + c
    leveled = image - plane
    
    # Calculate goodness of fit
    if len(residuals) > 0 and xp.sum(valid) > 3:
        residual_std = xp.sqrt(residuals[0] / (xp.sum(valid) - 3))
        ss_res = residuals[0]
        ss_tot = xp.sum((z[valid] - xp.mean(z[valid]))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        residual_std = 0.0
        r_squared = 0.0
    
    info = {
        'plane': xp.asanyarray(params),
        'residual_std': float(residual_std),
        'r_squared': float(r_squared)
    }
    
    return xp.asanyarray(leveled), info


def weighted_multi_plane_leveling(
    image: np.ndarray,
    regions: np.ndarray,
    masks: Optional[list] = None
) -> Tuple[np.ndarray, dict]:
    """
    Weighted multi-plane leveling for images with distinct regions.
    
    Fits separate planes to different regions and blends them smoothly.
    
    Parameters
    ----------
    image : ndarray
        2D AFM image (rows x columns)
    regions : ndarray
        Integer array labeling different regions (0, 1, 2, ...)
    masks : list of ndarray, optional
        List of boolean masks for each region
    
    Returns
    -------
    leveled_image : ndarray
        Multi-plane leveled image
    info : dict
        Dictionary containing:
        - 'region_planes': list of plane parameters for each region
        - 'blend_weights': blending weights used
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> regions = np.zeros_like(img, dtype=int)
    >>> regions[:128, :] = 1  # Two regions
    >>> leveled, info = weighted_multi_plane_leveling(img, regions)
    """
    xp = get_array_module(False)
    image = xp.asarray(image, dtype=xp.float64)
    regions = xp.asarray(regions, dtype=xp.int32)
    
    if image.ndim != 2 or regions.ndim != 2:
        raise ValueError("Expected 2D arrays")
    
    if image.shape != regions.shape:
        raise ValueError("Image and regions must have same shape")
    
    n_regions = int(xp.max(regions)) + 1
    
    # Fit plane to each region
    region_planes = []
    region_leveled = []
    region_weights = []
    
    for r in range(n_regions):
        region_mask = (regions == r)
        
        if masks is not None and r < len(masks):
            region_mask = region_mask & xp.asarray(masks[r])
        
        # Fit plane to this region
        if xp.any(region_mask):
            leveled_r, plane_info = plane_leveling(image, region_mask)
            region_planes.append(plane_info['plane'])
            
            # Calculate distance transform for smooth blending
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(xp.asanyarray(region_mask))
            weight = xp.asarray(dist > 0, dtype=xp.float64)
            region_weights.append(weight)
        else:
            region_planes.append(xp.array([0.0, 0.0, 0.0]))
            region_leveled.append(image)
            region_weights.append(xp.zeros_like(image))
        
        region_leveled.append(leveled_r)
    
    # Blend regions using weights
    total_weight = xp.sum(xp.stack(region_weights), axis=0)
    total_weight = xp.where(total_weight > 0, total_weight, 1.0)
    
    blended = xp.zeros_like(image)
    for r in range(n_regions):
        normalized_weight = region_weights[r] / total_weight
        blended += region_leveled[r] * normalized_weight
    
    info = {
        'region_planes': region_planes,
        'blend_weights': xp.asanyarray(region_weights)
    }
    
    return xp.asanyarray(blended), info


def level_image(
    image: np.ndarray,
    method: Literal['line', 'plane', 'weighted_plane'] = 'plane',
    mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    regions: Optional[np.ndarray] = None,
    reference_line: int = 0
) -> Tuple[np.ndarray, dict]:
    """
    Unified interface for image leveling operations.
    
    Parameters
    ----------
    image : ndarray
        2D AFM image
    method : str
        Leveling method: 'line', 'plane', or 'weighted_plane'
    mask : ndarray, optional
        Boolean mask for valid pixels
    weights : ndarray, optional
        Per-pixel weights
    regions : ndarray, optional
        Region labels for multi-plane leveling
    reference_line : int
        Reference line index for line leveling
    
    Returns
    -------
    leveled_image : ndarray
        Leveled image
    info : dict
        Method-specific information
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> leveled, info = level_image(img, method='plane')
    >>> assert 'plane' in info or 'offsets' in info
    """
    if method == 'line':
        leveled, offsets = line_leveling(image, mask, reference_line)
        return leveled, {'offsets': offsets, 'method': 'line'}
    
    elif method == 'plane':
        leveled, plane_info = plane_leveling(image, mask, weights)
        return leveled, {**plane_info, 'method': 'plane'}
    
    elif method == 'weighted_plane':
        if regions is None:
            raise ValueError("regions required for weighted_plane method")
        leveled, multi_info = weighted_multi_plane_leveling(image, regions, [mask] if mask is not None else None)
        return leveled, {**multi_info, 'method': 'weighted_plane'}
    
    else:
        raise ValueError(f"Unknown leveling method: {method}")


def batch_level_movie(
    movie: np.ndarray,
    method: Literal['line', 'plane'] = 'plane',
    mask: Optional[np.ndarray] = None,
    per_frame: bool = True
) -> Tuple[np.ndarray, list]:
    """
    Level all frames in a movie stack.
    
    Parameters
    ----------
    movie : ndarray
        3D array (frames x rows x cols)
    method : str
        Leveling method
    mask : ndarray, optional
        Mask applied to all frames
    per_frame : bool
        If True, level each frame independently
        If False, use global statistics
    
    Returns
    -------
    leveled_movie : ndarray
        Leveled movie stack
    frame_info : list
        List of info dicts for each frame
    
    Examples
    --------
    >>> import numpy as np
    >>> movie = np.random.rand(100, 256, 256)
    >>> leveled, info = batch_level_movie(movie)
    >>> assert leveled.shape == movie.shape
    """
    xp = get_array_module(False)
    movie = xp.asarray(movie)
    
    if movie.ndim != 3:
        raise ValueError(f"Expected 3D movie, got {movie.ndim}D")
    
    n_frames = movie.shape[0]
    leveled_frames = []
    frame_info = []
    
    for i in range(n_frames):
        frame = movie[i, :, :]
        leveled, info = level_image(frame, method=method, mask=mask)
        leveled_frames.append(leveled)
        frame_info.append(info)
    
    leveled_movie = xp.stack(leveled_frames, axis=0)
    
    return xp.asanyarray(leveled_movie), frame_info
