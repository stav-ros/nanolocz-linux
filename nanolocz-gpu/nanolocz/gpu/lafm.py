"""GPU-accelerated LAFM splatting and FRC computation.

This module provides CuPy-based implementations for:
- Gaussian splatting from localizations (super-resolution reconstruction)
- Fourier Ring Correlation (FRC) for resolution estimation
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nanolocz.gpu.backend import (
    CUPY_AVAILABLE,
    BackendContext,
    create_reference_context,
    get_backend_context,
)

# Import CuPy if available
if CUPY_AVAILABLE:
    import cupy as cp


def _validate_array(arr: Any, ctx: BackendContext) -> Any:
    """Validate and convert array to backend-appropriate format."""
    if CUPY_AVAILABLE and ctx.is_gpu:
        if isinstance(arr, cp.ndarray):
            return arr
        return cp.asarray(arr, dtype=ctx.dtype)
    return np.asarray(arr, dtype=ctx.dtype)


def splat_gaussian_gpu(
    coordinates: Any,
    intensities: Any | None = None,
    sigmas: Any | None = None,
    output_shape: tuple[int, int] | None = None,
    global_sigma: float = 2.0,
    ctx: BackendContext | None = None,
) -> Any:
    """Render Gaussian peaks at localization coordinates.

    Creates a super-resolution reconstruction by splatting Gaussian
    functions at each localization position.

    Args:
        coordinates: Array of coordinates (N, 2) with columns [x, y]
        intensities: Optional intensity weights (N,) - defaults to ones
        sigmas: Optional per-localization sigma values (N,) or None
        output_shape: Output image shape (height, width) - auto-computed if None
        global_sigma: Global sigma if per-localization sigmas not provided
        ctx: Backend context

    Returns:
        Reconstructed image as GPU or CPU array
    """
    if ctx is None:
        ctx = get_backend_context()

    coords = _validate_array(coordinates, ctx)
    if len(coords) == 0:
        if output_shape is None:
            return cp.zeros((100, 100), dtype=ctx.dtype) if CUPY_AVAILABLE and ctx.is_gpu else np.zeros((100, 100), dtype=ctx.dtype)
        return cp.zeros(output_shape, dtype=ctx.dtype) if CUPY_AVAILABLE and ctx.is_gpu else np.zeros(output_shape, dtype=ctx.dtype)

    # Default intensities
    if intensities is None:
        intensities = cp.ones(len(coords), dtype=ctx.dtype) if CUPY_AVAILABLE else np.ones(len(coords), dtype=ctx.dtype)
    else:
        intensities = _validate_array(intensities, ctx)

    # Determine output shape
    if output_shape is None:
        max_coords = cp.max(coords, axis=0) if CUPY_AVAILABLE else np.max(coords, axis=0)
        output_shape = (
            int(cp.ceil(max_coords[1]).item() + 10) if CUPY_AVAILABLE else int(np.ceil(max_coords[1])) + 10,
            int(cp.ceil(max_coords[0]).item() + 10) if CUPY_AVAILABLE else int(np.ceil(max_coords[0])) + 10,
        )

    # Create output image
    xp = cp if CUPY_AVAILABLE and ctx.is_gpu else np
    image = xp.zeros(output_shape, dtype=ctx.dtype)

    # Handle sigma
    if sigmas is None:
        sigmas_arr = xp.full(len(coords), global_sigma, dtype=ctx.dtype)
    else:
        sigmas_arr = _validate_array(sigmas, ctx)

    # Render each Gaussian
    n_points = len(coords)
    for i in range(n_points):
        x = float(coords[i, 0])
        y = float(coords[i, 1])
        sigma = float(sigmas_arr[i]) if len(sigmas_arr) > 1 else float(sigmas_arr)
        intensity = float(intensities[i])

        # Determine kernel bounds
        kernel_radius = int(max(3 * sigma, 1))
        x_min = max(0, int(x - kernel_radius))
        x_max = min(output_shape[1], int(x + kernel_radius) + 1)
        y_min = max(0, int(y - kernel_radius))
        y_max = min(output_shape[0], int(y + kernel_radius) + 1)

        # Create coordinate grids
        yy, xx = xp.meshgrid(
            xp.arange(y_min, y_max, dtype=ctx.dtype),
            xp.arange(x_min, x_max, dtype=ctx.dtype),
            indexing='ij'
        )

        # Compute Gaussian
        dx = xx - x
        dy = yy - y
        r_sq = dx ** 2 + dy ** 2
        sigma_sq = sigma ** 2

        gaussian = xp.exp(-r_sq / (2 * sigma_sq))
        image[y_min:y_max, x_min:x_max] += intensity * gaussian

    return image


def splat_localizations_gpu(
    localizations: dict[str, Any] | Any,
    output_shape: tuple[int, int] | None = None,
    global_sigma: float = 2.0,
    use_uncertainty: bool = True,
    ctx: BackendContext | None = None,
) -> Any:
    """Splat localizations into a super-resolution image.

    High-level interface that accepts a localization dictionary or array.

    Args:
        localizations: Either:
            - dict with 'coordinates', 'intensities', 'sigmas' keys
            - Array of coordinates (N, 2)
        output_shape: Output image shape (height, width)
        global_sigma: Default sigma if uncertainties not used
        use_uncertainty: Use per-localization sigmas if available
        ctx: Backend context

    Returns:
        Reconstructed super-resolution image
    """
    if ctx is None:
        ctx = get_backend_context()

    # Extract data from dict or use as coordinates
    if isinstance(localizations, dict):
        coordinates = localizations.get('coordinates', localizations.get('coords'))
        intensities = localizations.get('intensities')
        sigmas = localizations.get('sigmas') if use_uncertainty else None
    else:
        coordinates = localizations
        intensities = None
        sigmas = None

    return splat_gaussian_gpu(
        coordinates=coordinates,
        intensities=intensities,
        sigmas=sigmas,
        output_shape=output_shape,
        global_sigma=global_sigma,
        ctx=ctx,
    )


def compute_frc_gpu(
    map1: Any,
    map2: Any,
    mask: Any | None = None,
    ctx: BackendContext | None = None,
) -> tuple[Any, Any]:
    """Compute Fourier Ring Correlation between two maps.

    FRC measures the correlation between two independent reconstructions
    as a function of spatial frequency, used for resolution estimation.

    Args:
        map1: First input map (2D array)
        map2: Second input map (2D array)
        mask: Optional mask to restrict analysis region
        ctx: Backend context

    Returns:
        Tuple of (frequencies, frc_values) arrays
    """
    if ctx is None:
        ctx = get_backend_context()

    m1 = _validate_array(map1, ctx)
    m2 = _validate_array(map2, ctx)

    if m1.shape != m2.shape:
        raise ValueError(f"Maps must have same shape: {m1.shape} vs {m2.shape}")

    # Apply mask if provided
    if mask is not None:
        mask_arr = _validate_array(mask, ctx)
        m1 = m1 * mask_arr
        m2 = m2 * mask_arr

    xp = cp if CUPY_AVAILABLE and ctx.is_gpu else np

    # Compute FFTs
    fft1 = xp.fft.fft2(m1)
    fft2 = xp.fft.fft2(m2)

    # Compute complex conjugate product
    product = fft1 * xp.conj(fft2)

    # Get frequency coordinates
    ny, nx = m1.shape
    cy, cx = ny // 2, nx // 2

    # Create frequency grid
    y_freq = xp.fft.fftfreq(ny)
    x_freq = xp.fft.fftfreq(nx)
    yy, xx = xp.meshgrid(y_freq, x_freq, indexing='ij')

    # Compute radial frequencies
    freq_radii = xp.sqrt(xx ** 2 + yy ** 2)

    # Bin by frequency
    max_freq = float(xp.max(freq_radii))
    n_bins = min(int(max_freq * 2 * min(ny, nx)), 100)
    if n_bins < 2:
        n_bins = 2

    freq_bins = xp.linspace(0, max_freq, n_bins + 1)
    frc_values = xp.zeros(n_bins, dtype=xp.float64)

    for i in range(n_bins):
        freq_mask = (freq_radii >= freq_bins[i]) & (freq_radii < freq_bins[i + 1])
        if xp.sum(freq_mask) > 0:
            ring_product = product[freq_mask]
            ring_sum = xp.sum(ring_product)
            
            # Normalize by magnitudes
            mag1 = xp.abs(fft1[freq_mask])
            mag2 = xp.abs(fft2[freq_mask])
            norm = xp.sqrt(xp.sum(mag1 ** 2) * xp.sum(mag2 ** 2))
            
            if norm > 0:
                frc_values[i] = float(xp.abs(ring_sum) / norm)
            else:
                frc_values[i] = 0.0
        else:
            frc_values[i] = 0.0

    # Return bin centers
    freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2

    return freq_centers, frc_values


def frc_resolution(
    frequencies: Any,
    frc_values: Any,
    threshold: str = "1/7",
) -> float:
    """Estimate resolution from FRC curve.

    Args:
        frequencies: Array of spatial frequencies
        frc_values: Array of FRC values at each frequency
        threshold: Resolution criterion:
            - "1/7": Standard 1/7 threshold
            - "1/2bit": Half-bit information threshold
            - "0.5": Fixed 0.5 threshold
            - numeric: Custom threshold value

    Returns:
        Estimated resolution (inverse of cutoff frequency)
        Returns infinity if threshold never crossed
    """
    xp = cp if CUPY_AVAILABLE and hasattr(frequencies, 'get') else np

    # Convert to numpy for processing
    freq_np = xp.asarray(frequencies)
    frc_np = xp.asarray(frc_values)

    # Determine threshold value
    if threshold == "1/7":
        thresh_val = 1.0 / 7.0
    elif threshold == "1/2bit" or threshold == "half-bit":
        # Half-bit threshold: varies with sample size, simplified here
        thresh_val = 0.5
    elif isinstance(threshold, (int, float)):
        thresh_val = float(threshold)
    else:
        thresh_val = 1.0 / 7.0

    # Find where FRC crosses threshold
    above = frc_np >= thresh_val

    if not xp.any(above):
        return float('inf')

    # Find last frequency above threshold
    indices = xp.where(above)[0]
    last_index = int(indices[-1])

    if last_index >= len(frequencies) - 1:
        return float('inf')

    # Interpolate to find exact crossing
    if last_index < len(frequencies) - 1:
        f1, f2 = float(frc_np[last_index]), float(frc_np[last_index + 1])
        freq1, freq2 = float(freq_np[last_index]), float(freq_np[last_index + 1])

        if f1 != f2:
            # Linear interpolation
            t = (thresh_val - f1) / (f2 - f1)
            cutoff_freq = freq1 + t * (freq2 - freq1)
        else:
            cutoff_freq = freq2
    else:
        cutoff_freq = float(freq_np[last_index])

    if cutoff_freq <= 0:
        return float('inf')

    # Resolution is inverse of cutoff frequency
    return 1.0 / cutoff_freq


def batch_splat_gpu(
    localizations_list: list[dict[str, Any] | Any],
    output_shape: tuple[int, int] | None = None,
    global_sigma: float = 2.0,
    ctx: BackendContext | None = None,
) -> Any:
    """Splat multiple frames of localizations in batch.

    Args:
        localizations_list: List of localization dicts or arrays
        output_shape: Output shape for each frame
        global_sigma: Default sigma value
        ctx: Backend context

    Returns:
        Stack of reconstructed images (T, H, W)
    """
    if ctx is None:
        ctx = get_backend_context()

    if len(localizations_list) == 0:
        if output_shape is None:
            output_shape = (100, 100)
        xp = cp if CUPY_AVAILABLE and ctx.is_gpu else np
        return xp.zeros((0,) + output_shape, dtype=ctx.dtype)

    # Process each frame
    frames = []
    for locs in localizations_list:
        frame = splat_localizations_gpu(
            locs,
            output_shape=output_shape,
            global_sigma=global_sigma,
            ctx=ctx,
        )
        frames.append(frame)

    # Stack frames
    xp = cp if CUPY_AVAILABLE and ctx.is_gpu else np
    if CUPY_AVAILABLE and ctx.is_gpu:
        return cp.stack(frames)
    return xp.stack(frames)


# Export public API
__all__ = [
    'splat_gaussian_gpu',
    'splat_localizations_gpu',
    'compute_frc_gpu',
    'frc_resolution',
    'batch_splat_gpu',
]
