"""Deterministic CPU particle detection and detection statistics.

The NumPy implementation is the reference behavior for NL-16. Coordinates are
always returned as ``(x, y)`` even though NumPy indexes images as ``(y, x)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter
from skimage.measure import profile_line

from nanolocz.core.types import DetectionResult


def _validate_image(img: Any) -> np.ndarray:
    image = np.asarray(img, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got {image.ndim}D")
    if image.size == 0:
        raise ValueError("image must not be empty")
    return np.nan_to_num(image, nan=-np.inf)


def _validate_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    value = np.asarray(mask)
    if value.shape != shape:
        raise ValueError(f"mask shape {value.shape} does not match image shape {shape}")
    if value.dtype != bool:
        raise ValueError("mask must have boolean dtype")
    return value


def _prominence(image: np.ndarray, peaks: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """Estimate prominence along the line to the nearest higher peak."""
    values = np.zeros(len(peaks), dtype=np.float64)
    for index, peak in enumerate(peaks):
        higher = np.flatnonzero(heights > heights[index])
        if higher.size:
            distances = np.sum((peaks[higher] - peak) ** 2, axis=1)
            # lexsort makes equal-distance choices deterministic.
            nearest = higher[np.lexsort((highter := higher, distances))[0]]
            profile = profile_line(image, peak[::-1], peaks[nearest][::-1], mode="nearest")
            values[index] = heights[index] - float(np.min(profile))
        else:
            finite = image[np.isfinite(image)]
            values[index] = heights[index] - float(np.min(finite)) if finite.size else 0.0
    return np.maximum(values, 0.0)


def _select_min_distance(peaks: np.ndarray, heights: np.ndarray, min_distance: float) -> np.ndarray:
    """Greedily retain strongest peaks separated by ``min_distance`` pixels."""
    if min_distance <= 0 or len(peaks) < 2:
        return np.arange(len(peaks))
    order = np.lexsort((peaks[:, 0], peaks[:, 1], -heights))
    kept: list[int] = []
    minimum_squared = float(min_distance) ** 2
    for candidate in order:
        if all(np.sum((peaks[candidate] - peaks[index]) ** 2) >= minimum_squared for index in kept):
            kept.append(int(candidate))
    return np.asarray(sorted(kept), dtype=int)


def fast_peaks2d(
    img: Any,
    thresh: float,
    kernel_size: int,
    min_prom: float | None = None,
    use_gpu: bool = False,
    *,
    mask: Any = None,
    min_distance: float = 0.0,
) -> np.ndarray:
    """Find local maxima, returning columns ``x, y, height, prominence``.

    ``use_gpu`` is accepted for API compatibility; this CPU reference path is
    intentionally deterministic and is the implementation used for parity.
    """
    del use_gpu
    image = _validate_image(img)
    allowed = _validate_mask(mask, image.shape)
    if not np.isscalar(thresh) or not np.isfinite(thresh):
        raise ValueError("thresh must be a finite scalar")
    if not isinstance(kernel_size, (int, np.integer)) or kernel_size < 1:
        raise ValueError("kernel_size must be a positive integer")
    if min_prom is not None and (not np.isscalar(min_prom) or not np.isfinite(min_prom)):
        raise ValueError("min_prom must be a finite scalar")

    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    candidate_mask = allowed & np.isfinite(image) & (image >= float(thresh))
    local_maxima = candidate_mask & (maximum_filter(image, size=size, mode="nearest") == image)
    y, x = np.where(local_maxima)
    if not len(x):
        return np.empty((0, 4), dtype=np.float64)

    # y then x is the stable image-order contract.
    order = np.lexsort((x, y))
    peaks = np.column_stack((x[order], y[order])).astype(np.float64)
    heights = image[y[order], x[order]]
    prominences = _prominence(image, peaks, heights)
    keep = _select_min_distance(peaks, heights, float(min_distance))
    if min_prom is not None:
        keep = keep[prominences[keep] >= float(min_prom)]
    return np.column_stack((peaks[keep], heights[keep], prominences[keep]))


def _statistics(image: np.ndarray, coordinates: np.ndarray, radius: int = 2) -> dict[str, np.ndarray]:
    stats = {name: np.empty(len(coordinates), dtype=np.float64) for name in ("area", "volume", "eccentricity")}
    for index, (x_value, y_value) in enumerate(coordinates):
        x, y = int(round(x_value)), int(round(y_value))
        y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
        region = image[y0:y1, x0:x1]
        border = np.concatenate((region[0, :], region[-1, :], region[:, 0], region[:, -1]))
        background = float(np.median(border)) if border.size else 0.0
        signal = np.maximum(region - background, 0.0)
        stats["area"][index] = float(np.count_nonzero(signal > 0))
        stats["volume"][index] = float(np.sum(signal))
        yy, xx = np.indices(region.shape, dtype=np.float64)
        weight = signal.ravel()
        if weight.sum() <= 0 or len(weight) < 2:
            stats["eccentricity"][index] = 0.0
            continue
        coords = np.column_stack((xx.ravel(), yy.ravel()))
        mean = np.average(coords, axis=0, weights=weight)
        centered = coords - mean
        covariance = (centered * weight[:, None]).T @ centered / float(weight.sum())
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        major, minor = float(eigenvalues[-1]), float(eigenvalues[0])
        stats["eccentricity"][index] = np.sqrt(max(0.0, 1.0 - minor / major)) if major else 0.0
    return stats


def detect_particles(
    img: Any,
    method: str = "direct",
    ref_img: Any = None,
    thresh: float | None = None,
    kernel_size: int = 5,
    min_prom: float | None = None,
    rotation_angles: Any = None,
    use_gpu: bool = False,
    *,
    mask: Any = None,
    min_distance: float = 0.0,
    threshold: float | None = None,  # Backward compatibility alias for thresh
) -> DetectionResult:
    """Detect particles and return typed coordinates, mask, and statistics.
    
    Parameters
    ----------
    img : array_like
        Input image data
    method : str
        Detection method: 'direct' or 'crosscorr'
    ref_img : array_like, optional
        Reference image for cross-correlation method
    thresh : float, optional
        Detection threshold (default: auto-calculated)
    threshold : float, optional
        Alias for thresh for backward compatibility with tests
    kernel_size : int
        Size of local maxima kernel
    min_prom : float, optional
        Minimum prominence for peak selection
    rotation_angles : array_like, optional
        Angles to test for cross-correlation method
    use_gpu : bool
        GPU acceleration flag (currently CPU-only reference)
    mask : array_like, optional
        Boolean mask for valid regions
    min_distance : float
        Minimum distance between detected peaks
    """
    # Backward compatibility: threshold kwarg aliases thresh
    if threshold is not None:
        if thresh is not None:
            raise ValueError("Cannot specify both 'thresh' and 'threshold'")
        thresh = threshold
    
    image = _validate_image(img)
    allowed = _validate_mask(mask, image.shape)
    if method == "direct":
        threshold = float(np.mean(image[np.isfinite(image)]) + 2 * np.std(image[np.isfinite(image)])) if thresh is None else thresh
        peaks = fast_peaks2d(image, threshold, kernel_size, min_prom, use_gpu, mask=allowed, min_distance=min_distance)
        angle = None
    elif method == "crosscorr":
        if ref_img is None:
            raise ValueError("ref_img required for crosscorr method")
        reference = _validate_image(ref_img)
        deviation = float(np.std(reference))
        if deviation == 0:
            raise ValueError("ref_img must have non-zero variance")
        from scipy.ndimage import rotate
        from scipy.signal import correlate2d
        angles = [0.0] if rotation_angles is None else [float(value) for value in rotation_angles]
        best: tuple[np.ndarray, float, float] | None = None
        for candidate_angle in angles:
            rotated = rotate(reference, candidate_angle, reshape=False) if candidate_angle else reference
            normalized = (rotated - np.mean(rotated)) / float(np.std(rotated))
            correlation = correlate2d(image, normalized, mode="same", boundary="symm")
            threshold = float(np.mean(correlation) + 3 * np.std(correlation)) if thresh is None else thresh
            candidate = fast_peaks2d(correlation, threshold, kernel_size, min_prom, False, mask=allowed, min_distance=min_distance)
            score = float(np.max(candidate[:, 2])) if len(candidate) else -np.inf
            if best is None or score > best[1]:
                best = (candidate, score, candidate_angle)
        peaks, _, angle = best if best is not None else (np.empty((0, 4)), -np.inf, 0.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    coordinates = peaks[:, :2].astype(np.float64, copy=False)
    intensities = peaks[:, 2].astype(np.float64, copy=False)
    scores = peaks[:, 3].astype(np.float64, copy=False)
    output_mask = np.zeros(image.shape, dtype=bool)
    if len(coordinates):
        output_mask[coordinates[:, 1].astype(int), coordinates[:, 0].astype(int)] = True
    return DetectionResult(
        coordinates=coordinates,
        intensities=intensities,
        scores=scores,
        mask=output_mask,
        statistics=_statistics(image, coordinates),
        angle=angle,
    )


# Kept private-name compatibility for callers of the prototype.
def _calculate_prominence(img: Any, peaks: np.ndarray, heights: np.ndarray) -> np.ndarray:
    return _prominence(_validate_image(img), np.asarray(peaks, dtype=np.float64), np.asarray(heights, dtype=np.float64))
