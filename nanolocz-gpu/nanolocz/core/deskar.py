"""
Directional deskar filtering for AFM image processing.

Implements directional filtering to remove scan-line artifacts and periodic noise
patterns from AFM images while preserving particle features.

Functions
---------
directional_deskar : FFT-based directional frequency filtering
remove_scan_lines : Scan-line artifact removal using robust statistics
anisotropic_diffusion : Edge-preserving smoothing with directional preference
process_movie_deskar : Batch processing for movies
"""

import numpy as np
from typing import Literal, Optional, Tuple, Union, Callable
from scipy.ndimage import median_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def directional_deskar(
    image: np.ndarray,
    scan_angle: float = 0.0,
    frequency_cutoff: float = 0.1,
    notch_width: float = 0.02,
    strength: float = 0.8,
    preserve_low_freq: bool = True,
) -> np.ndarray:
    """
    Apply directional deskar filtering using FFT-based frequency domain masking.
    
    Removes scan-line artifacts and periodic noise aligned with the scan direction
    while preserving isotropic particle features.
    
    Parameters
    ----------
    image : ndarray
        Input 2D image array
    scan_angle : float
        Direction of scan lines in degrees (default 0.0 = horizontal)
    frequency_cutoff : float
        Normalized frequency threshold for artifact removal (0-0.5)
    notch_width : float
        Width of frequency mask around artifact peaks (normalized)
    strength : float
        Filter intensity from 0.0 (no filtering) to 1.0 (full filtering)
    preserve_low_freq : bool
        If True, preserve low-frequency sample topography
        
    Returns
    -------
    filtered : ndarray
        Filtered image with reduced directional artifacts
        
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> filtered = directional_deskar(img, scan_angle=0.0, frequency_cutoff=0.1)
    >>> assert filtered.shape == img.shape
    
    Notes
    -----
    Algorithm:
    1. Compute 2D FFT of input image
    2. Rotate frequency domain by -scan_angle
    3. Create mask for frequencies aligned with scan direction
    4. Apply notch/band-stop filter at identified artifact frequencies
    5. Rotate back by +scan_angle
    6. Compute inverse FFT
    """
    image = np.asarray(image, dtype=np.float64)
    
    if image.ndim != 2:
        raise ValueError(f"Input must be 2D, got {image.ndim}D")
    
    # Clamp parameters to valid ranges
    strength = np.clip(strength, 0.0, 1.0)
    frequency_cutoff = np.clip(frequency_cutoff, 0.0, 0.5)
    notch_width = np.clip(notch_width, 0.0, 0.5)
    
    # No filtering needed if strength is zero
    if strength < 1e-10:
        return image.copy()
    
    # Compute 2D FFT
    fft_image = fftshift(fft2(image))
    magnitude = np.abs(fft_image)
    phase = np.angle(fft_image)
    
    # Create frequency coordinate grids
    ny, nx = image.shape
    freq_y = np.fft.fftfreq(ny)
    freq_x = np.fft.fftfreq(nx)
    freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)
    
    # Rotate frequency coordinates by -scan_angle
    angle_rad = -np.deg2rad(scan_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    freq_x_rot = freq_x_grid * cos_a - freq_y_grid * sin_a
    freq_y_rot = freq_x_grid * sin_a + freq_y_grid * cos_a
    
    # Create directional mask (perpendicular to scan direction)
    # Artifacts appear as lines perpendicular to scan direction in freq domain
    freq_magnitude = np.sqrt(freq_x_rot**2 + freq_y_rot**2)
    
    # Mask for directional artifacts (narrow band around perpendicular axis)
    directional_mask = np.abs(freq_x_rot) < notch_width
    
    # High-pass component (remove low frequencies if not preserving)
    if not preserve_low_freq:
        high_pass_mask = freq_magnitude > frequency_cutoff
        directional_mask = directional_mask & high_pass_mask
    
    # Low-frequency preservation mask
    low_freq_mask = freq_magnitude < frequency_cutoff
    
    # Combine masks
    if preserve_low_freq:
        # Don't filter low frequencies
        filter_mask = directional_mask & ~low_freq_mask
    else:
        filter_mask = directional_mask
    
    # Apply filter with strength blending
    filtered_fft = fft_image.copy()
    filtered_fft[filter_mask] *= (1.0 - strength)
    
    # Inverse FFT
    filtered_image = np.real(ifft2(ifftshift(filtered_fft)))
    
    return filtered_image


def remove_scan_lines(
    image: np.ndarray,
    direction: Union[str, float] = 'horizontal',
    method: Literal['median', 'mean', 'robust'] = 'median',
    threshold: float = 2.0,
    interpolation_method: Literal['linear', 'nearest'] = 'linear',
) -> np.ndarray:
    """
    Remove scan-line artifacts using line-by-line correction.
    
    Detects and corrects offsets in individual scan lines using robust statistics.
    
    Parameters
    ----------
    image : ndarray
        Input 2D image array
    direction : str or float
        Scan direction: 'horizontal', 'vertical', or angle in degrees
    method : {'median', 'mean', 'robust'}
        Method for computing line statistics
    threshold : float
        Sigma threshold for scar/outlier detection
    interpolation_method : {'linear', 'nearest'}
        Method for interpolating corrected values
        
    Returns
    -------
    corrected : ndarray
        Image with scan-line artifacts removed
        
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> corrected = remove_scan_lines(img, direction='horizontal', method='median')
    >>> assert corrected.shape == img.shape
    """
    image = np.asarray(image, dtype=np.float64)
    
    if image.ndim != 2:
        raise ValueError(f"Input must be 2D, got {image.ndim}D")
    
    # Determine axis based on direction
    if direction == 'horizontal':
        axis = 0  # Process rows
    elif direction == 'vertical':
        axis = 1  # Process columns
    else:
        # Custom angle - rotate image, process horizontally, rotate back
        angle_rad = np.deg2rad(direction)
        # For simplicity, use nearest cardinal direction
        if abs(np.cos(angle_rad)) > abs(np.sin(angle_rad)):
            axis = 0  # More horizontal than vertical
        else:
            axis = 1
    
    corrected = image.copy()
    
    # Process each line
    if axis == 0:
        n_lines = image.shape[0]
        for i in range(n_lines):
            line = image[i, :]
            
            # Compute line statistic
            if method == 'median':
                line_stat = np.median(line)
            elif method == 'mean':
                line_stat = np.mean(line)
            else:  # robust
                # Use trimmed mean (exclude outliers)
                q1, q3 = np.percentile(line, [25, 75])
                iqr = q3 - q1
                mask = (line >= q1 - threshold * iqr) & (line <= q3 + threshold * iqr)
                line_stat = np.mean(line[mask]) if np.any(mask) else np.median(line)
            
            # Detect outliers (scars)
            if method == 'robust' or threshold is not None:
                q1, q3 = np.percentile(line, [25, 75])
                iqr = q3 - q1
                outlier_mask = (line < q1 - threshold * iqr) | (line > q3 + threshold * iqr)
                
                # Replace outliers with interpolated values
                if np.any(outlier_mask) and interpolation_method == 'linear':
                    # Simple linear interpolation
                    valid_indices = np.where(~outlier_mask)[0]
                    outlier_indices = np.where(outlier_mask)[0]
                    
                    if len(valid_indices) > 1:
                        # Interpolate from neighbors
                        for idx in outlier_indices:
                            left_valid = valid_indices[valid_indices < idx]
                            right_valid = valid_indices[valid_indices > idx]
                            
                            if len(left_valid) > 0 and len(right_valid) > 0:
                                left_val = line[left_valid[-1]]
                                right_val = line[right_valid[0]]
                                line[idx] = (left_val + right_val) / 2
                            elif len(left_valid) > 0:
                                line[idx] = line[left_valid[-1]]
                            elif len(right_valid) > 0:
                                line[idx] = line[right_valid[0]]
            
            # Subtract line offset to center the line
            global_median = np.median(image)
            corrected[i, :] = line - line_stat + global_median
            
    else:  # axis == 1 (vertical)
        n_lines = image.shape[1]
        for i in range(n_lines):
            line = image[:, i]
            
            if method == 'median':
                line_stat = np.median(line)
            elif method == 'mean':
                line_stat = np.mean(line)
            else:  # robust
                q1, q3 = np.percentile(line, [25, 75])
                iqr = q3 - q1
                mask = (line >= q1 - threshold * iqr) & (line <= q3 + threshold * iqr)
                line_stat = np.mean(line[mask]) if np.any(mask) else np.median(line)
            
            if method == 'robust' or threshold is not None:
                q1, q3 = np.percentile(line, [25, 75])
                iqr = q3 - q1
                outlier_mask = (line < q1 - threshold * iqr) | (line > q3 + threshold * iqr)
                
                if np.any(outlier_mask) and interpolation_method == 'linear':
                    valid_indices = np.where(~outlier_mask)[0]
                    outlier_indices = np.where(outlier_mask)[0]
                    
                    if len(valid_indices) > 1:
                        for idx in outlier_indices:
                            left_valid = valid_indices[valid_indices < idx]
                            right_valid = valid_indices[valid_indices > idx]
                            
                            if len(left_valid) > 0 and len(right_valid) > 0:
                                left_val = line[left_valid[-1]]
                                right_val = line[right_valid[0]]
                                line[idx] = (left_val + right_val) / 2
                            elif len(left_valid) > 0:
                                line[idx] = line[left_valid[-1]]
                            elif len(right_valid) > 0:
                                line[idx] = line[right_valid[0]]
            
            global_median = np.median(image)
            corrected[:, i] = line - line_stat + global_median
    
    return corrected


def anisotropic_diffusion(
    image: np.ndarray,
    n_iterations: int = 10,
    kappa: float = 50.0,
    gamma: float = 0.1,
    scan_angle: float = 0.0,
) -> np.ndarray:
    """
    Apply anisotropic diffusion for edge-preserving smoothing.
    
    Performs iterative diffusion with preferential smoothing along the scan direction,
    preserving edges perpendicular to scan lines.
    
    Parameters
    ----------
    image : ndarray
        Input 2D image array
    n_iterations : int
        Number of diffusion iterations
    kappa : float
        Conductance parameter controlling edge sensitivity
    gamma : float
        Step size (must be <= 0.25 for stability)
    scan_angle : float
        Preferred diffusion direction in degrees (0.0 = horizontal)
        
    Returns
    -------
    diffused : ndarray
        Smoothed image with preserved edges
        
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> diffused = anisotropic_diffusion(img, n_iterations=10, kappa=50.0)
    >>> assert diffused.shape == img.shape
    
    Notes
    -----
    Algorithm based on Perona-Malik anisotropic diffusion with directional tensor:
    I(t+1) = I(t) + γ * div(D * ∇I)
    
    where D is the diffusion tensor with preferred orientation.
    """
    image = np.asarray(image, dtype=np.float64)
    
    if image.ndim != 2:
        raise ValueError(f"Input must be 2D, got {image.ndim}D")
    
    # Clamp gamma for stability
    gamma = np.clip(gamma, 0.0, 0.25)
    
    diffused = image.copy()
    angle_rad = np.deg2rad(scan_angle)
    
    # Precompute directional coefficients
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cos2_a, sin2_a = cos_a**2, sin_a**2
    sin_2a = np.sin(2 * angle_rad)
    
    for _ in range(n_iterations):
        # Compute gradients
        grad_y, grad_x = np.gradient(diffused)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Conductance function (edge-stopping)
        conductance = np.exp(-(grad_mag / kappa)**2)
        
        # Directional diffusion tensor components
        # Prefer diffusion along scan direction
        D_xx = cos2_a * conductance
        D_yy = sin2_a * conductance
        D_xy = 0.5 * sin_2a * conductance
        
        # Compute divergence of flux
        # Flux in x direction
        flux_x = D_xx * grad_x + D_xy * grad_y
        # Flux in y direction
        flux_y = D_xy * grad_x + D_yy * grad_y
        
        # Divergence
        div_flux_x = np.gradient(flux_x, axis=1)[1] if image.shape[1] > 2 else np.zeros_like(image)
        div_flux_y = np.gradient(flux_y, axis=0)[1] if image.shape[0] > 2 else np.zeros_like(image)
        
        # Handle boundary conditions
        if div_flux_x.ndim == 1:
            div_flux_x = np.tile(div_flux_x, (image.shape[0], 1))
        if div_flux_y.ndim == 1:
            div_flux_y = np.tile(div_flux_y.reshape(-1, 1), (1, image.shape[1]))
        
        # Update image
        if div_flux_x.shape == image.shape and div_flux_y.shape == image.shape:
            diffused += gamma * (div_flux_x + div_flux_y)
        else:
            # Fallback to simpler scheme
            diffused += gamma * (
                np.gradient(conductance * grad_x, axis=1)[0] +
                np.gradient(conductance * grad_y, axis=0)[0]
            )
    
    return diffused


def process_movie_deskar(
    movie: Union[np.ndarray, list],
    scan_angle: float = 0.0,
    frequency_cutoff: float = 0.1,
    notch_width: float = 0.02,
    strength: float = 0.8,
    preserve_low_freq: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Union[np.ndarray, list]:
    """
    Apply directional deskar filtering to a movie (sequence of frames).
    
    Parameters
    ----------
    movie : ndarray or list
        Input movie as 3D array (frames, height, width) or list of 2D arrays
    scan_angle : float
        Direction of scan lines in degrees
    frequency_cutoff : float
        Normalized frequency threshold for artifact removal
    notch_width : float
        Width of frequency mask around artifact peaks
    strength : float
        Filter intensity from 0.0 to 1.0
    preserve_low_freq : bool
        If True, preserve low-frequency sample topography
    progress_callback : callable, optional
        Function(current_frame, total_frames) called after each frame
        
    Returns
    -------
    processed : ndarray or list
        Processed movie in same format as input
        
    Examples
    --------
    >>> import numpy as np
    >>> movie = np.random.rand(100, 256, 256)
    >>> processed = process_movie_deskar(movie, scan_angle=0.0)
    >>> assert processed.shape == movie.shape
    """
    if isinstance(movie, list):
        # List of frames
        n_frames = len(movie)
        processed = []
        
        for i, frame in enumerate(movie):
            filtered = directional_deskar(
                frame,
                scan_angle=scan_angle,
                frequency_cutoff=frequency_cutoff,
                notch_width=notch_width,
                strength=strength,
                preserve_low_freq=preserve_low_freq,
            )
            processed.append(filtered)
            
            if progress_callback is not None:
                progress_callback(i + 1, n_frames)
        
        return processed
    
    elif isinstance(movie, np.ndarray):
        if movie.ndim != 3:
            raise ValueError(f"Movie must be 3D (frames, height, width), got {movie.ndim}D")
        
        n_frames = movie.shape[0]
        processed = np.zeros_like(movie)
        
        for i in range(n_frames):
            processed[i] = directional_deskar(
                movie[i],
                scan_angle=scan_angle,
                frequency_cutoff=frequency_cutoff,
                notch_width=notch_width,
                strength=strength,
                preserve_low_freq=preserve_low_freq,
            )
            
            if progress_callback is not None:
                progress_callback(i + 1, n_frames)
        
        return processed
    
    else:
        raise TypeError("movie must be ndarray or list of ndarrays")


# GPU stub module placeholder
def get_gpu_module():
    """
    Attempt to import CuPy for GPU acceleration.
    
    Returns
    -------
    module or None
        CuPy module if available, None otherwise
    """
    try:
        import cupy as cp
        return cp
    except ImportError:
        return None


def directional_deskar_gpu(
    image: np.ndarray,
    scan_angle: float = 0.0,
    frequency_cutoff: float = 0.1,
    notch_width: float = 0.02,
    strength: float = 0.8,
    preserve_low_freq: bool = True,
) -> np.ndarray:
    """
    GPU-accelerated directional deskar filtering using CuPy.
    
    Falls back to CPU implementation if CuPy is unavailable.
    
    Parameters
    ----------
    image : ndarray
        Input 2D image array
    scan_angle : float
        Direction of scan lines in degrees
    frequency_cutoff : float
        Normalized frequency threshold for artifact removal
    notch_width : float
        Width of frequency mask around artifact peaks
    strength : float
        Filter intensity from 0.0 to 1.0
    preserve_low_freq : bool
        If True, preserve low-frequency sample topography
        
    Returns
    -------
    filtered : ndarray
        Filtered image (returned to CPU memory)
    """
    cp = get_gpu_module()
    
    if cp is None:
        # Fall back to CPU
        return directional_deskar(
            image, scan_angle, frequency_cutoff, notch_width, strength, preserve_low_freq
        )
    
    # Transfer to GPU
    image_gpu = cp.asarray(image, dtype=cp.float64)
    
    # GPU implementation mirrors CPU logic but uses CuPy operations
    strength = float(cp.clip(strength, 0.0, 1.0))
    frequency_cutoff = float(cp.clip(frequency_cutoff, 0.0, 0.5))
    notch_width = float(cp.clip(notch_width, 0.0, 0.5))
    
    if strength < 1e-10:
        return cp.asnumpy(image_gpu)
    
    # 2D FFT on GPU
    fft_image = cp.fft.fftshift(cp.fft.fft2(image_gpu))
    
    # Create frequency grids on GPU
    ny, nx = image_gpu.shape
    freq_y = cp.fft.fftfreq(ny)
    freq_x = cp.fft.fftfreq(nx)
    freq_x_grid, freq_y_grid = cp.meshgrid(freq_x, freq_y)
    
    # Rotate frequency coordinates
    angle_rad = -cp.deg2rad(scan_angle)
    cos_a, sin_a = cp.cos(angle_rad), cp.sin(angle_rad)
    
    freq_x_rot = freq_x_grid * cos_a - freq_y_grid * sin_a
    freq_y_rot = freq_x_grid * sin_a + freq_y_grid * cos_a
    
    # Create directional mask
    freq_magnitude = cp.sqrt(freq_x_rot**2 + freq_y_rot**2)
    directional_mask = cp.abs(freq_x_rot) < notch_width
    
    if not preserve_low_freq:
        high_pass_mask = freq_magnitude > frequency_cutoff
        directional_mask = directional_mask & high_pass_mask
    
    low_freq_mask = freq_magnitude < frequency_cutoff
    
    if preserve_low_freq:
        filter_mask = directional_mask & ~low_freq_mask
    else:
        filter_mask = directional_mask
    
    # Apply filter
    filtered_fft = fft_image.copy()
    filtered_fft[filter_mask] *= (1.0 - strength)
    
    # Inverse FFT and transfer back to CPU
    filtered_image = cp.real(cp.fft.ifft2(cp.fft.ifftshift(filtered_fft)))
    
    return cp.asnumpy(filtered_image)
