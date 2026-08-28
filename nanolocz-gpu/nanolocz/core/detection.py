"""
Particle detection module for NanoLocz.

Ports MATLAB Fast_peaks2D.m and Detector.m functionality to Python.
Supports both CPU and GPU acceleration.
"""

import numpy as np
from scipy.ndimage import maximum_filter
from skimage.measure import profile_line

from nanolocz.gpu.utils import get_array_module, GPUArrayModule


def fast_peaks2d(img, thresh, kernel_size, min_prom=None, use_gpu=False):
    """
    Fast 2D peak detection for particle localization.
    
    Ports MATLAB Fast_peaks2D.m function.
    
    Parameters
    ----------
    img : ndarray
        2D grayscale image
    thresh : float
        Intensity threshold for peak detection
    kernel_size : int
        Size of local neighborhood for maximum filtering
    min_prom : float, optional
        Minimum prominence for peak filtering
    use_gpu : bool, optional
        Enable GPU acceleration (default: False)
        
    Returns
    -------
    locs : ndarray
        Nx4 array of [x, y, height, prominence]
        
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(100, 100)
    >>> peaks = fast_peaks2d(img, thresh=0.5, kernel_size=3)
    >>> print(f"Found {len(peaks)} peaks")
    """
    xp = get_array_module(use_gpu)
    gpu_mod = GPUArrayModule(use_gpu=use_gpu)
    
    # Transfer to GPU if enabled
    img_dev = gpu_mod.to_device(img)
    
    kernel_size = kernel_size + 2
    
    # Maximum filter to find local maxima
    max_filtered = gpu_mod.maximum_filter(img_dev, size=kernel_size)
    local_maxima_mask = (max_filtered == img_dev) & (img_dev > thresh)
    
    # Exclude edge pixels
    h, w = img_dev.shape
    edge_margin = 2
    local_maxima_mask[:edge_margin, :] = False
    local_maxima_mask[-edge_margin:, :] = False
    local_maxima_mask[:, :edge_margin] = False
    local_maxima_mask[:, -edge_margin:] = False
    
    # Get coordinates and heights
    y_coords, x_coords = xp.where(local_maxima_mask)
    peak_heights = img_dev[local_maxima_mask]
    
    if len(x_coords) == 0:
        return xp.array([]).reshape(0, 4)
    
    locs = xp.column_stack([x_coords, y_coords, peak_heights])
    
    # Add prominence calculation if requested
    if min_prom is not None and min_prom > 0:
        # Need to calculate on CPU for now (profile_line not in CuPy)
        locs_cpu = gpu_mod.from_device(locs)
        img_cpu = gpu_mod.from_device(img_dev)
        peaks_cpu = locs_cpu[:, :2]
        heights_cpu = locs_cpu[:, 2]
        
        prominences = _calculate_prominence(img_cpu, peaks_cpu, heights_cpu)
        keep = prominences > min_prom
        locs = xp.column_stack([locs[keep, :3], xp.asarray(prominences[keep])])
    else:
        prominences = xp.zeros(len(x_coords))
        locs = xp.column_stack([locs, prominences])
    
    # Return to CPU
    return gpu_mod.from_device(locs)


def _calculate_prominence(img, peaks, heights):
    """
    Calculate peak prominence (similar to MATLAB impofile approach).
    
    Parameters
    ----------
    img : ndarray
        2D grayscale image
    peaks : ndarray
        Nx2 array of [x, y] coordinates
    heights : ndarray
        N array of peak heights
        
    Returns
    -------
    prominences : ndarray
        N array of prominence values
    """
    n_peaks = len(peaks)
    prominences = np.zeros(n_peaks)
    
    for j in range(n_peaks):
        # Calculate distances to all other peaks
        dists = np.sqrt(np.sum((peaks - peaks[j])**2, axis=1))
        
        # Sort by distance
        order = np.argsort(dists)
        
        # Find closest higher peak
        higher = heights[order] > heights[j]
        if np.any(higher):
            first_higher_idx = np.where(higher)[0][0]
            neighbor_idx = order[first_higher_idx]
            
            # Sample intensity profile between peaks
            profile = profile_line(
                img, 
                peaks[j], 
                peaks[neighbor_idx]
            )
            prominences[j] = heights[j] - np.min(profile)
        else:
            prominences[j] = heights[j]
    
    return prominences


def detect_particles(img, method='direct', ref_img=None, 
                     thresh=None, kernel_size=5, min_prom=None,
                     rotation_angles=None, use_gpu=False):
    """
    Particle detection with multiple modes.
    
    Ports MATLAB Detector.m functionality.
    
    Parameters
    ----------
    img : ndarray
        2D grayscale image or image stack
    method : str
        Detection method: 'direct' or 'crosscorr'
    ref_img : ndarray, optional
        Reference image for cross-correlation mode
    thresh : float, optional
        Intensity threshold (auto-calculated if None)
    kernel_size : int
        Size of local neighborhood for peak detection
    min_prom : float, optional
        Minimum prominence for peak filtering
    rotation_angles : array, optional
        Angles to test for rotation search (crosscorr mode)
    use_gpu : bool
        Enable GPU acceleration
        
    Returns
    -------
    detections : dict
        Dictionary containing:
        - 'locs': Nx4 array of [x, y, height, prominence]
        - 'scores': Detection scores (crosscorr mode only)
    """
    if method == 'direct':
        # Direct peak picking
        if thresh is None:
            thresh = np.mean(img) + 2 * np.std(img)
        
        locs = fast_peaks2d(img, thresh, kernel_size, min_prom, use_gpu)
        
        return {
            'locs': locs,
            'scores': None
        }
    
    elif method == 'crosscorr':
        if ref_img is None:
            raise ValueError("ref_img required for crosscorr method")
        
        # Cross-correlation based detection with rotation search
        from scipy.signal import correlate2d
        
        best_score = -np.inf
        best_locs = None
        best_angle = None
        
        angles_to_test = rotation_angles if rotation_angles is not None else [0]
        
        for angle in angles_to_test:
            if angle != 0:
                from scipy.ndimage import rotate
                ref_rotated = rotate(ref_img, angle, reshape=False)
            else:
                ref_rotated = ref_img
            
            # Normalize reference
            ref_norm = (ref_rotated - np.mean(ref_rotated)) / np.std(ref_rotated)
            
            # Cross-correlation
            corr = correlate2d(img, ref_norm, mode='same', boundary='symm')
            
            # Detect peaks in correlation map
            if thresh is None:
                auto_thresh = np.mean(corr) + 3 * np.std(corr)
            else:
                auto_thresh = thresh
            
            locs = fast_peaks2d(corr, auto_thresh, kernel_size, min_prom, use_gpu)
            
            if len(locs) > 0:
                score = np.max(locs[:, 2])
                if score > best_score:
                    best_score = score
                    best_locs = locs
                    best_angle = angle
        
        return {
            'locs': best_locs,
            'scores': best_score,
            'angle': best_angle
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")
