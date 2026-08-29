"""GPU-accelerated particle detection kernels.

This module provides CuPy-based implementations of detection algorithms
for GPU acceleration, maintaining parity with the CPU reference in
nanolocz.core.detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nanolocz.gpu.backend import (
    CUPY_AVAILABLE,
    BackendContext,
    TolerancePolicy,
    assert_close,
    create_reference_context,
    get_backend_context,
)

# Import CuPy if available
if CUPY_AVAILABLE:
    import cupy as cp
    import cupyx.scipy.ndimage


def _validate_image_gpu(img: Any, ctx: BackendContext) -> Any:
    """Validate and convert image to GPU array."""
    if CUPY_AVAILABLE:
        if isinstance(img, cp.ndarray):
            return img
        return cp.asarray(img, dtype=ctx.dtype)
    # Fallback: convert to numpy
    return np.asarray(img, dtype=ctx.dtype)


def local_maxima_gpu(
    image: Any,
    kernel_size: int = 5,
    threshold: float | None = None,
    mask: Any = None,
    ctx: BackendContext | None = None,
) -> tuple[Any, Any]:
    """Find local maxima in an image using maximum filter on GPU.

    Args:
        image: Input 2D image (numpy or cupy array)
        kernel_size: Size of the maximum filter kernel
        threshold: Minimum intensity threshold (None for auto)
        mask: Boolean mask to restrict detection region
        ctx: Backend context (auto-created if None)

    Returns:
        Tuple of (y_coords, x_coords) as GPU arrays
    """
    if ctx is None:
        ctx = get_backend_context()

    # Convert to GPU array
    img_gpu = _validate_image_gpu(image, ctx)

    if img_gpu.ndim != 2:
        raise ValueError(f"Image must be 2D, got {img_gpu.ndim}D")

    # Auto threshold if not provided
    if threshold is None:
        finite_vals = img_gpu[cp.isfinite(img_gpu)] if CUPY_AVAILABLE else img_gpu[np.isfinite(img_gpu)]
        threshold = float(cp.mean(finite_vals) + 2 * cp.std(finite_vals)) if CUPY_AVAILABLE else float(np.mean(finite_vals) + 2 * np.std(finite_vals))

    # Create mask
    if mask is None:
        allowed = cp.ones_like(img_gpu, dtype=cp.bool_) if CUPY_AVAILABLE else np.ones_like(img_gpu, dtype=bool)
    else:
        allowed = _validate_image_gpu(mask, ctx).astype(cp.bool_ if CUPY_AVAILABLE else bool)

    # Apply maximum filter
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1

    if CUPY_AVAILABLE and ctx.is_gpu:
        local_max = cupyx.scipy.ndimage.maximum_filter(img_gpu, size=size, mode='nearest')
        candidate_mask = allowed & cp.isfinite(img_gpu) & (img_gpu >= threshold)
        local_maxima = candidate_mask & (local_max == img_gpu)
    else:
        # CPU fallback
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(img_gpu, size=size, mode='nearest')
        candidate_mask = allowed & np.isfinite(img_gpu) & (img_gpu >= threshold)
        local_maxima = candidate_mask & (local_max == img_gpu)

    # Get coordinates
    y_coords, x_coords = cp.where(local_maxima) if CUPY_AVAILABLE and ctx.is_gpu else np.where(local_maxima)

    return y_coords, x_coords


def prominence_gpu(
    image: Any,
    peaks: Any,
    heights: Any,
    ctx: BackendContext | None = None,
) -> Any:
    """Calculate prominence for each peak along line to nearest higher peak.

    Args:
        image: Input 2D image
        peaks: Array of peak coordinates (N, 2) with columns [x, y]
        heights: Array of peak heights (N,)
        ctx: Backend context

    Returns:
        Array of prominence values (N,)
    """
    if ctx is None:
        ctx = get_backend_context()

    img_gpu = _validate_image_gpu(image, ctx)
    if CUPY_AVAILABLE:
        peaks_gpu = cp.asarray(peaks) if not isinstance(peaks, cp.ndarray) else peaks
        heights_gpu = cp.asarray(heights) if not isinstance(heights, cp.ndarray) else heights
    else:
        peaks_gpu = np.asarray(peaks)
        heights_gpu = np.asarray(heights)

    n_peaks = len(peaks_gpu)
    prominences = cp.zeros(n_peaks, dtype=img_gpu.dtype) if CUPY_AVAILABLE else np.zeros(n_peaks, dtype=img_gpu.dtype)

    # For each peak, find nearest higher peak and compute prominence
    for i in range(n_peaks):
        # Find peaks with higher height
        if CUPY_AVAILABLE:
            higher_mask = heights_gpu > heights_gpu[i]
            higher_indices = cp.where(higher_mask)[0]
        else:
            higher_mask = heights_gpu > heights_gpu[i]
            higher_indices = np.where(higher_mask)[0]

        if len(higher_indices) > 0:
            # Compute distances to all higher peaks
            diff = peaks_gpu[higher_indices] - peaks_gpu[i]
            distances_sq = cp.sum(diff ** 2, axis=1) if CUPY_AVAILABLE else np.sum(diff ** 2, axis=1)

            # Find nearest (minimum distance)
            nearest_idx = higher_indices[cp.argmin(distances_sq) if CUPY_AVAILABLE else np.argmin(distances_sq)]

            # Extract profile along line
            start_pt = peaks_gpu[i][::-1]  # [y, x]
            end_pt = peaks_gpu[nearest_idx][::-1]

            if CUPY_AVAILABLE and ctx.is_gpu:
                # Use cupyx for profile_line
                from cupyx.scipy.ndimage import map_coordinates
                # Simple linear interpolation along the line
                n_points = max(int(cp.linalg.norm(end_pt - start_pt)), 1)
                t = cp.linspace(0, 1, n_points)
                y_coords = start_pt[0] + t * (end_pt[0] - start_pt[0])
                x_coords = start_pt[1] + t * (end_pt[1] - start_pt[1])

                # Sample image along line
                coords = cp.stack([y_coords, x_coords])
                profile = map_coordinates(img_gpu, coords, order=1, mode='nearest')
                min_val = cp.min(profile)
            else:
                from skimage.measure import profile_line
                profile = profile_line(
                    cp.asnumpy(img_gpu) if CUPY_AVAILABLE else img_gpu,
                    cp.asnumpy(start_pt) if CUPY_AVAILABLE else start_pt,
                    cp.asnumpy(end_pt) if CUPY_AVAILABLE else end_pt,
                    mode="nearest"
                )
                min_val = cp.min(cp.asarray(profile)) if CUPY_AVAILABLE else np.min(profile)

            prominences[i] = max(heights_gpu[i] - min_val, 0)
        else:
            # No higher peak: use global minimum
            finite_vals = img_gpu[cp.isfinite(img_gpu)] if CUPY_AVAILABLE else img_gpu[np.isfinite(img_gpu)]
            if len(finite_vals) > 0:
                min_val = cp.min(finite_vals) if CUPY_AVAILABLE else np.min(finite_vals)
                prominences[i] = max(heights_gpu[i] - min_val, 0)
            else:
                prominences[i] = 0

    return prominences


def min_distance_suppression_gpu(
    peaks: Any,
    heights: Any,
    min_distance: float,
    ctx: BackendContext | None = None,
) -> Any:
    """Apply minimum distance suppression to peaks.

    Greedy algorithm: select strongest peaks that are separated by
    at least min_distance pixels.

    Args:
        peaks: Array of peak coordinates (N, 2) with columns [x, y]
        heights: Array of peak heights (N,)
        min_distance: Minimum separation distance in pixels
        ctx: Backend context

    Returns:
        Array of indices of kept peaks
    """
    if ctx is None:
        ctx = get_backend_context()

    if CUPY_AVAILABLE:
        peaks_gpu = cp.asarray(peaks) if not isinstance(peaks, cp.ndarray) else peaks
        heights_gpu = cp.asarray(heights) if not isinstance(heights, cp.ndarray) else heights
    else:
        peaks_gpu = np.asarray(peaks)
        heights_gpu = np.asarray(heights)

    if min_distance <= 0 or len(peaks_gpu) < 2:
        return cp.arange(len(peaks_gpu), dtype=cp.int64) if CUPY_AVAILABLE else np.arange(len(peaks_gpu), dtype=np.int64)

    # Sort by height (descending), then by position for determinism
    if CUPY_AVAILABLE:
        order = cp.lexsort((peaks_gpu[:, 0], peaks_gpu[:, 1], -heights_gpu))
    else:
        order = np.lexsort((peaks_gpu[:, 0], peaks_gpu[:, 1], -heights_gpu))

    kept = []
    min_dist_sq = float(min_distance) ** 2

    for candidate_idx in order:
        candidate = peaks_gpu[candidate_idx]
        should_keep = True

        for kept_idx in kept:
            kept_peak = peaks_gpu[kept_idx]
            dist_sq = float(cp.sum((candidate - kept_peak) ** 2)) if CUPY_AVAILABLE else float(np.sum((candidate - kept_peak) ** 2))
            if dist_sq < min_dist_sq:
                should_keep = False
                break

        if should_keep:
            kept.append(int(candidate_idx))

    # Return sorted indices
    kept_array = cp.array(sorted(kept), dtype=cp.int64) if CUPY_AVAILABLE else np.array(sorted(kept), dtype=np.int64)
    return kept_array


def detect_particles_gpu(
    image: Any,
    threshold: float | None = None,
    kernel_size: int = 5,
    min_prominence: float | None = None,
    min_distance: float = 0.0,
    mask: Any = None,
    ctx: BackendContext | None = None,
) -> Any:
    """Detect particles using GPU-accelerated algorithms.

    This is the main entry point for GPU detection, matching the API
    of the CPU implementation in nanolocz.core.detection.

    Args:
        image: Input 2D image
        threshold: Intensity threshold (None for auto)
        kernel_size: Size of maximum filter kernel
        min_prominence: Minimum prominence threshold (None for no filtering)
        min_distance: Minimum distance between peaks
        mask: Boolean mask to restrict detection
        ctx: Backend context

    Returns:
        Array of detections with columns [x, y, height, prominence]
    """
    if ctx is None:
        ctx = get_backend_context()

    # Find local maxima
    y_coords, x_coords = local_maxima_gpu(
        image, kernel_size=kernel_size, threshold=threshold, mask=mask, ctx=ctx
    )

    if len(x_coords) == 0:
        # Return empty array with correct shape
        return cp.empty((0, 4), dtype=ctx.dtype) if CUPY_AVAILABLE else np.empty((0, 4), dtype=ctx.dtype)

    # Stack coordinates
    if CUPY_AVAILABLE:
        peaks = cp.column_stack([x_coords, y_coords]).astype(ctx.dtype)
    else:
        peaks = np.column_stack([x_coords, y_coords]).astype(ctx.dtype)

    # Get heights at peak locations
    img_gpu = _validate_image_gpu(image, ctx)
    if CUPY_AVAILABLE:
        heights = img_gpu[y_coords, x_coords]
    else:
        heights = img_gpu[y_coords, x_coords]

    # Calculate prominence
    prominences = prominence_gpu(img_gpu, peaks, heights, ctx=ctx)

    # Apply minimum distance suppression
    keep_indices = min_distance_suppression_gpu(peaks, heights, min_distance, ctx=ctx)

    # Filter by prominence if specified
    if min_prominence is not None:
        prom_mask = prominences[keep_indices] >= min_prominence
        keep_indices = keep_indices[prom_mask] if CUPY_AVAILABLE else keep_indices[prom_mask]

    # Build result array
    if CUPY_AVAILABLE:
        result = cp.column_stack([
            peaks[keep_indices, 0],
            peaks[keep_indices, 1],
            heights[keep_indices],
            prominences[keep_indices]
        ])
    else:
        result = np.column_stack([
            peaks[keep_indices, 0],
            peaks[keep_indices, 1],
            heights[keep_indices],
            prominences[keep_indices]
        ])

    return result


def statistics_gpu(
    image: Any,
    coordinates: Any,
    radius: int = 2,
    ctx: BackendContext | None = None,
) -> dict[str, Any]:
    """Compute detection statistics (area, volume, eccentricity) on GPU.

    Args:
        image: Input 2D image
        coordinates: Array of peak coordinates (N, 2) with columns [x, y]
        radius: Radius for region extraction
        ctx: Backend context

    Returns:
        Dictionary with keys 'area', 'volume', 'eccentricity'
    """
    if ctx is None:
        ctx = get_backend_context()

    img_gpu = _validate_image_gpu(image, ctx)
    if CUPY_AVAILABLE:
        coords_gpu = cp.asarray(coordinates) if not isinstance(coordinates, cp.ndarray) else coordinates
    else:
        coords_gpu = np.asarray(coordinates)

    n_peaks = len(coords_gpu)
    xp = cp if CUPY_AVAILABLE and ctx.is_gpu else np

    area = xp.zeros(n_peaks, dtype=xp.float64)
    volume = xp.zeros(n_peaks, dtype=xp.float64)
    eccentricity = xp.zeros(n_peaks, dtype=xp.float64)

    for i in range(n_peaks):
        x = int(round(float(coords_gpu[i, 0])))
        y = int(round(float(coords_gpu[i, 1])))

        # Extract region
        y0, y1 = max(0, y - radius), min(img_gpu.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(img_gpu.shape[1], x + radius + 1)

        if CUPY_AVAILABLE and ctx.is_gpu:
            region = img_gpu[y0:y1, x0:x1]
            border = xp.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            background = float(xp.median(border)) if border.size > 0 else 0.0
            signal = xp.maximum(region - background, 0)

            area[i] = float(xp.count_nonzero(signal > 0))
            volume[i] = float(xp.sum(signal))

            # Compute eccentricity
            yy, xx = xp.indices(region.shape, dtype=xp.float64)
            weight = signal.ravel()
            if weight.sum() <= 0 or len(weight) < 2:
                eccentricity[i] = 0.0
                continue

            coords_flat = xp.column_stack([xx.ravel(), yy.ravel()])
            mean = xp.average(coords_flat, axis=0, weights=weight)
            centered = coords_flat - mean
            covariance = (centered * weight[:, None]).T @ centered / float(weight.sum())

            # Eigenvalue decomposition
            eigenvalues = xp.linalg.eigvalsh(covariance)
            eigenvalues = xp.clip(eigenvalues, 0, None)
            major, minor = float(eigenvalues[-1]), float(eigenvalues[0])

            if major > 0:
                eccentricity[i] = float(xp.sqrt(max(0.0, 1.0 - minor / major)))
            else:
                eccentricity[i] = 0.0
        else:
            # CPU fallback
            region = img_gpu[y0:y1, x0:x1]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            background = float(np.median(border)) if border.size > 0 else 0.0
            signal = np.maximum(region - background, 0)

            area[i] = float(np.count_nonzero(signal > 0))
            volume[i] = float(np.sum(signal))

            # Compute eccentricity
            yy, xx = np.indices(region.shape, dtype=np.float64)
            weight = signal.ravel()
            if weight.sum() <= 0 or len(weight) < 2:
                eccentricity[i] = 0.0
                continue

            coords_flat = np.column_stack([xx.ravel(), yy.ravel()])
            mean = np.average(coords_flat, axis=0, weights=weight)
            centered = coords_flat - mean
            covariance = (centered * weight[:, None]).T @ centered / float(weight.sum())

            eigenvalues = np.linalg.eigvalsh(covariance)
            eigenvalues = np.clip(eigenvalues, 0, None)
            major, minor = float(eigenvalues[-1]), float(eigenvalues[0])

            if major > 0:
                eccentricity[i] = float(np.sqrt(max(0.0, 1.0 - minor / major)))
            else:
                eccentricity[i] = 0.0

    return {
        'area': area,
        'volume': volume,
        'eccentricity': eccentricity,
    }


# Export public API
__all__ = [
    'local_maxima_gpu',
    'prominence_gpu',
    'min_distance_suppression_gpu',
    'detect_particles_gpu',
    'statistics_gpu',
]
