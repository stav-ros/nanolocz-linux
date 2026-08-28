"""
Filters, masks, and profile operations for AFM image processing.

Implements Gaussian filtering, median filtering, scar removal,
and custom profile operations for AFM data preprocessing.
"""

import numpy as np
from typing import Literal, Optional, Tuple, Union
from scipy.ndimage import (
    gaussian_filter,
    median_filter,
    uniform_filter,
    sobel,
    laplace,
    distance_transform_edt,
    binary_erosion,
    binary_dilation,
    generate_binary_structure
)


def gaussian_blur(
    image: np.ndarray,
    sigma: Union[float, Tuple[float, float]] = 1.0,
    truncate: float = 4.0
) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    
    Parameters
    ----------
    image : ndarray
        Input image (2D or 3D)
    sigma : float or tuple
        Standard deviation for Gaussian kernel
    truncate : float
        Truncate the filter at this many standard deviations
    
    Returns
    -------
    filtered : ndarray
        Gaussian-filtered image
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> blurred = gaussian_blur(img, sigma=2.0)
    >>> assert blurred.shape == img.shape
    """
    return gaussian_filter(image.astype(np.float64), sigma=sigma, truncate=truncate)


def median_blur(
    image: np.ndarray,
    size: int = 3,
    footprint: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply median filter to remove salt-and-pepper noise.
    
    Parameters
    ----------
    image : ndarray
        Input image (2D or 3D)
    size : int
        Size of the median filter window
    footprint : ndarray, optional
        Custom footprint for the filter
    
    Returns
    -------
    filtered : ndarray
        Median-filtered image
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> denoised = median_blur(img, size=3)
    >>> assert denoised.shape == img.shape
    """
    return median_filter(image.astype(np.float64), size=size, footprint=footprint)


def uniform_blur(
    image: np.ndarray,
    size: int = 3
) -> np.ndarray:
    """
    Apply uniform (box) filter.
    
    Parameters
    ----------
    image : ndarray
        Input image
    size : int
        Size of the uniform filter window
    
    Returns
    -------
    filtered : ndarray
        Uniform-filtered image
    """
    return uniform_filter(image.astype(np.float64), size=size)


def compute_gradient(
    image: np.ndarray,
    mode: Literal['sobel', 'gradient'] = 'sobel'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradient magnitude and direction.
    
    Parameters
    ----------
    image : ndarray
        Input 2D image
    mode : str
        Gradient computation method
    
    Returns
    -------
    magnitude : ndarray
        Gradient magnitude
    direction : ndarray
        Gradient direction (radians)
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> mag, direction = compute_gradient(img)
    >>> assert mag.shape == img.shape
    """
    if mode == 'sobel':
        # Compute Sobel gradients
        gy = sobel(image, axis=0)
        gx = sobel(image, axis=1)
    else:
        # Simple finite differences
        gy = np.gradient(image, axis=0)
        gx = np.gradient(image, axis=1)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    return magnitude, direction


def compute_laplacian(image: np.ndarray) -> np.ndarray:
    """
    Compute Laplacian of the image for edge detection.
    
    Parameters
    ----------
    image : ndarray
        Input 2D image
    
    Returns
    -------
    laplacian : ndarray
        Laplacian-filtered image
    """
    return laplace(image.astype(np.float64))


def create_mask_from_threshold(
    image: np.ndarray,
    threshold: float,
    mode: Literal['above', 'below'] = 'above',
    fill_holes: bool = False,
    min_size: int = 0
) -> np.ndarray:
    """
    Create binary mask from intensity threshold.
    
    Parameters
    ----------
    image : ndarray
        Input image
    threshold : float
        Threshold value
    mode : str
        'above' for values > threshold, 'below' for < threshold
    fill_holes : bool
        Fill holes in the mask
    min_size : int
        Minimum object size to keep (in pixels)
    
    Returns
    -------
    mask : ndarray
        Boolean mask
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> mask = create_mask_from_threshold(img, 0.5)
    >>> assert mask.dtype == bool
    """
    if mode == 'above':
        mask = image > threshold
    else:
        mask = image < threshold
    
    if fill_holes or min_size > 0:
        from scipy.ndimage import label, binary_fill_holes
        
        if fill_holes:
            mask = binary_fill_holes(mask)
        
        if min_size > 0:
            # Remove small objects
            labeled, n_objects = label(mask)
            for i in range(1, n_objects + 1):
                if np.sum(labeled == i) < min_size:
                    mask[labeled == i] = False
    
    return mask


def create_circular_mask(
    shape: Tuple[int, int],
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[float] = None
) -> np.ndarray:
    """
    Create a circular mask.
    
    Parameters
    ----------
    shape : tuple
        Image shape (rows, cols)
    center : tuple, optional
        Center coordinates (row, col). Defaults to image center.
    radius : float, optional
        Circle radius. Defaults to half the minimum dimension.
    
    Returns
    -------
    mask : ndarray
        Boolean circular mask
    
    Examples
    --------
    >>> mask = create_circular_mask((256, 256), center=(128, 128), radius=50)
    >>> assert mask.shape == (256, 256)
    >>> assert mask.dtype == bool
    """
    rows, cols = shape
    
    if center is None:
        center = (rows // 2, cols // 2)
    
    if radius is None:
        radius = min(rows, cols) / 2
    
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    return dist_from_center <= radius


def create_rectangular_mask(
    shape: Tuple[int, int],
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int]
) -> np.ndarray:
    """
    Create a rectangular ROI mask.
    
    Parameters
    ----------
    shape : tuple
        Image shape (rows, cols)
    top_left : tuple
        Top-left corner (row, col)
    bottom_right : tuple
        Bottom-right corner (row, col)
    
    Returns
    -------
    mask : ndarray
        Boolean rectangular mask
    """
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    
    r1, c1 = top_left
    r2, c2 = bottom_right
    
    mask[r1:r2, c1:c2] = True
    
    return mask


def remove_scars(
    image: np.ndarray,
    direction: Literal['horizontal', 'vertical'] = 'horizontal',
    threshold: float = 3.0,
    max_width: int = 5
) -> np.ndarray:
    """
    Remove scan line artifacts (scars) from AFM images.
    
    Detects and interpolates over abnormally bright/dark lines.
    
    Parameters
    ----------
    image : ndarray
        Input 2D image
    direction : str
        Orientation of scars to remove
    threshold : float
        Number of standard deviations to consider as scar
    max_width : int
        Maximum width of scars to remove (in pixels/lines)
    
    Returns
    -------
    corrected : ndarray
        Image with scars removed
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> img[50, :] *= 5  # Add horizontal scar
    >>> corrected = remove_scars(img, direction='horizontal')
    >>> assert corrected.shape == img.shape
    """
    image = image.astype(np.float64)
    corrected = image.copy()
    
    if direction == 'horizontal':
        # Process row by row
        axis = 0
        profile_metric = np.std(image, axis=1)
    else:
        # Process column by column
        axis = 1
        profile_metric = np.std(image, axis=0)
    
    # Identify outlier lines
    median_metric = np.median(profile_metric)
    mad = np.median(np.abs(profile_metric - median_metric))
    
    # Avoid division by zero
    if mad == 0:
        mad = 1e-10
    
    z_scores = np.abs(profile_metric - median_metric) / (1.4826 * mad)
    scar_indices = np.where(z_scores > threshold)[0]
    
    # Interpolate over scar lines
    for idx in scar_indices:
        if direction == 'horizontal':
            # Get neighboring good lines
            start = max(0, idx - max_width)
            end = min(image.shape[0], idx + max_width + 1)
            
            good_rows = []
            for i in range(start, idx):
                if i not in scar_indices:
                    good_rows.append(i)
            for i in range(idx + 1, end):
                if i not in scar_indices:
                    good_rows.append(i)
            
            if len(good_rows) > 0:
                # Interpolate from neighbors
                corrected[idx, :] = np.mean(corrected[good_rows, :], axis=0)
        else:
            # Vertical scars
            start = max(0, idx - max_width)
            end = min(image.shape[1], idx + max_width + 1)
            
            good_cols = []
            for i in range(start, idx):
                if i not in scar_indices:
                    good_cols.append(i)
            for i in range(idx + 1, end):
                if i not in scar_indices:
                    good_cols.append(i)
            
            if len(good_cols) > 0:
                corrected[:, idx] = np.mean(corrected[:, good_cols], axis=1)
    
    return corrected


def extract_profile(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    width: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract intensity profile along a line.
    
    Parameters
    ----------
    image : ndarray
        Input 2D image
    start : tuple
        Start point (row, col)
    end : tuple
        End point (row, col)
    width : int
        Width of the profile line (pixels)
    
    Returns
    -------
    distances : ndarray
        Distance along profile
    intensities : ndarray
        Intensity values along profile
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> distances, intensities = extract_profile(img, (0, 0), (255, 255))
    >>> assert len(distances) == len(intensities)
    """
    from skimage.measure import profile_line
    
    # profile_line expects (row, col) format
    profile = profile_line(image, start, end, linewidth=width, mode='nearest')
    
    # Calculate distances
    dy = end[0] - start[0]
    dx = end[1] - start[1]
    total_distance = np.sqrt(dx**2 + dy**2)
    
    distances = np.linspace(0, total_distance, len(profile))
    
    return distances, profile


def extract_radial_profile(
    image: np.ndarray,
    center: Tuple[int, int],
    max_radius: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract radial intensity profile from a center point.
    
    Parameters
    ----------
    image : ndarray
        Input 2D image
    center : tuple
        Center point (row, col)
    max_radius : int, optional
        Maximum radius to consider
    
    Returns
    -------
    radii : ndarray
        Radius values
    mean_intensity : ndarray
        Mean intensity at each radius
    std_intensity : ndarray
        Standard deviation at each radius
    
    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> radii, mean_int, std_int = extract_radial_profile(img, (128, 128))
    >>> assert len(radii) == len(mean_int)
    """
    rows, cols = image.shape
    
    if max_radius is None:
        max_radius = min(rows, cols) // 2
    
    # Create coordinate grids relative to center
    y, x = np.ogrid[:rows, :cols]
    y = y - center[0]
    x = x - center[1]
    
    radii = np.sqrt(x**2 + y**2).astype(int)
    
    # Calculate statistics for each radius
    mean_intensity = []
    std_intensity = []
    valid_radii = []
    
    for r in range(max_radius):
        mask = (radii == r)
        if np.any(mask):
            mean_intensity.append(np.mean(image[mask]))
            std_intensity.append(np.std(image[mask]))
            valid_radii.append(r)
    
    return np.array(valid_radii), np.array(mean_intensity), np.array(std_intensity)


def morphological_operations(
    mask: np.ndarray,
    operation: Literal['erode', 'dilate', 'open', 'close'],
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply morphological operations to a binary mask.
    
    Parameters
    ----------
    mask : ndarray
        Binary input mask
    operation : str
        Type of operation: 'erode', 'dilate', 'open', 'close'
    iterations : int
        Number of iterations
    structure : ndarray, optional
        Structuring element
    
    Returns
    -------
    result : ndarray
        Processed binary mask
    
    Examples
    --------
    >>> import numpy as np
    >>> mask = np.zeros((100, 100), dtype=bool)
    >>> mask[40:60, 40:60] = True
    >>> eroded = morphological_operations(mask, 'erode')
    """
    if structure is None:
        structure = generate_binary_structure(2, 1)
    
    if operation == 'erode':
        return binary_erosion(mask, structure=structure, iterations=iterations)
    elif operation == 'dilate':
        return binary_dilation(mask, structure=structure, iterations=iterations)
    elif operation == 'open':
        eroded = binary_erosion(mask, structure=structure, iterations=iterations)
        return binary_dilation(eroded, structure=structure, iterations=iterations)
    elif operation == 'close':
        dilated = binary_dilation(mask, structure=structure, iterations=iterations)
        return binary_erosion(dilated, structure=structure, iterations=iterations)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance transform of a binary mask.
    
    Parameters
    ----------
    mask : ndarray
        Binary mask (True = foreground)
    
    Returns
    -------
    distances : ndarray
        Distance to nearest background pixel
    
    Examples
    --------
    >>> import numpy as np
    >>> mask = np.zeros((100, 100), dtype=bool)
    >>> mask[40:60, 40:60] = True
    >>> dist = distance_transform(mask)
    >>> assert dist.max() > 0
    """
    return distance_transform_edt(mask)
