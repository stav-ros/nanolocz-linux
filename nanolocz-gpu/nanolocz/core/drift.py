"""Per-frame drift estimation and correction for AFM movies.

This module provides algorithms for estimating and correcting sample drift
in AFM image sequences using cross-correlation and particle-based methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import shift
from scipy.signal import correlate2d

from nanolocz.core.detection import detect_particles
from nanolocz.core.types import Frame


@dataclass
class DriftResult:
    """Result from drift estimation.
    
    Attributes:
        shifts: Per-frame shifts relative to reference, shape (N, 2) with [dy, dx]
        cumulative_drift: Cumulative drift trajectory, shape (N, 2) with [dy, dx]
        method: Estimation method used ('xcorr' or 'particles')
        reference_frame: Index of reference frame
        per_frame_quality: Quality metric per frame (correlation peak or match count)
    """
    shifts: np.ndarray  # Shape (N, 2), columns [dy, dx]
    cumulative_drift: np.ndarray  # Shape (N, 2), columns [dy, dx]
    method: str
    reference_frame: int = 0
    per_frame_quality: np.ndarray | None = None
    
    def __post_init__(self):
        """Validate drift result."""
        if self.shifts.ndim != 2 or self.shifts.shape[1] != 2:
            raise ValueError(f"shifts must be shape (N, 2), got {self.shifts.shape}")
        if len(self.cumulative_drift) != len(self.shifts):
            raise ValueError("cumulative_drift length must match shifts")
        if self.method not in ('xcorr', 'particles'):
            raise ValueError(f"method must be 'xcorr' or 'particles', got {self.method}")


def _movie_to_array(movie: list[Frame] | np.ndarray) -> np.ndarray:
    """Convert movie to 3D numpy array."""
    if isinstance(movie, np.ndarray):
        if movie.ndim != 3:
            raise ValueError(f"Movie array must be 3D, got {movie.ndim}D")
        return movie
    
    if isinstance(movie, list):
        if len(movie) == 0:
            raise ValueError("Movie list cannot be empty")
        frames = []
        for item in movie:
            if isinstance(item, Frame):
                frames.append(item.data)
            elif isinstance(item, np.ndarray):
                frames.append(item)
            else:
                raise TypeError(f"Expected Frame or ndarray, got {type(item)}")
        return np.stack(frames, axis=0)
    
    raise TypeError(f"Expected list[Frame] or ndarray, got {type(movie)}")


def _phase_correlation_shift(
    image1: np.ndarray,
    image2: np.ndarray,
    upsample_factor: int = 10,
) -> tuple[float, float, float]:
    """Estimate sub-pixel shift between two images using phase correlation.
    
    Args:
        image1: Reference image
        image2: Moving image
        upsample_factor: Factor for sub-pixel upsampling
        
    Returns:
        Tuple of (shift_y, shift_x, correlation_peak)
    """
    # Normalize images
    img1 = image1 - np.mean(image1)
    img2 = image2 - np.mean(image2)
    
    # Compute cross-correlation via FFT
    fft1 = np.fft.fft2(img1)
    fft2 = np.fft.fft2(img2)
    cross_power_spectrum = fft1 * np.conj(fft2)
    
    # Normalize to get phase correlation
    magnitude = np.abs(cross_power_spectrum)
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    phase_correlation = np.fft.ifft2(cross_power_spectrum / magnitude)
    
    # Find peak location
    phase_corr_real = np.real(phase_correlation)
    max_idx = np.unravel_index(np.argmax(phase_corr_real), phase_corr_real.shape)
    
    # Handle wrap-around for shifts larger than half image size
    shift_y = max_idx[0] if max_idx[0] <= image1.shape[0] // 2 else max_idx[0] - image1.shape[0]
    shift_x = max_idx[1] if max_idx[1] <= image1.shape[1] // 2 else max_idx[1] - image1.shape[1]
    
    # Sub-pixel refinement via upsampling (simplified DFT upsampling)
    if upsample_factor > 1:
        # Extract neighborhood around peak
        neighborhood_size = 10
        y_range = slice(max(0, max_idx[0] - neighborhood_size), 
                       min(image1.shape[0], max_idx[0] + neighborhood_size + 1))
        x_range = slice(max(0, max_idx[1] - neighborhood_size),
                       min(image1.shape[1], max_idx[1] + neighborhood_size + 1))
        
        local_region = phase_corr_real[y_range, x_range]
        local_max = np.unravel_index(np.argmax(local_region), local_region.shape)
        
        # Fit parabolic surface for sub-pixel precision
        if local_region.shape[0] >= 3 and local_region.shape[1] >= 3:
            cy, cx = local_max
            if 0 < cy < local_region.shape[0] - 1 and 0 < cx < local_region.shape[1] - 1:
                # Parabolic fit in y direction
                y_vals = local_region[cy-1:cy+2, cx]
                dy = 0.5 * (y_vals[0] - y_vals[2]) / (y_vals[0] - 2*y_vals[1] + y_vals[2] + 1e-10)
                
                # Parabolic fit in x direction  
                x_vals = local_region[cy, cx-1:cx+2]
                dx = 0.5 * (x_vals[0] - x_vals[2]) / (x_vals[0] - 2*x_vals[1] + x_vals[2] + 1e-10)
                
                shift_y += dy
                shift_x += dx
    
    # Get correlation peak value as quality metric
    peak_value = float(phase_corr_real[max_idx])
    
    return float(shift_y), float(shift_x), peak_value


def estimate_drift_xcorr(
    movie: list[Frame] | np.ndarray,
    reference: int | np.ndarray = 0,
    upsample_factor: int = 10,
) -> DriftResult:
    """Estimate drift using cross-correlation (phase correlation).
    
    This method uses FFT-based phase correlation to estimate frame-to-frame
    shifts with sub-pixel precision. It is robust to noise and works well
    for images with sufficient texture.
    
    Args:
        movie: Input movie as list[Frame] or 3D array (frames, height, width)
        reference: Reference frame index or array (default: 0 = first frame)
        upsample_factor: Factor for sub-pixel refinement (default: 10)
        
    Returns:
        DriftResult with shifts, cumulative_drift, and quality metrics
        
    Raises:
        ValueError: If movie is empty or has invalid dimensions
        TypeError: If movie type is not supported
    """
    movie_array = _movie_to_array(movie)
    n_frames = movie_array.shape[0]
    
    if n_frames < 2:
        # Single frame: no drift
        return DriftResult(
            shifts=np.zeros((n_frames, 2), dtype=np.float64),
            cumulative_drift=np.zeros((n_frames, 2), dtype=np.float64),
            method='xcorr',
            reference_frame=0 if isinstance(reference, int) else 0,
        )
    
    # Get reference frame
    if isinstance(reference, int):
        ref_frame = movie_array[reference]
        ref_idx = reference
    else:
        ref_frame = np.asarray(reference, dtype=np.float64)
        if ref_frame.ndim != 2:
            raise ValueError(f"Reference must be 2D, got {ref_frame.ndim}D")
        ref_idx = -1  # External reference
    
    # Estimate shifts relative to reference
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    quality = np.zeros(n_frames, dtype=np.float64)
    
    # First frame (reference) has zero shift
    if isinstance(reference, int) and reference == 0:
        quality[0] = 1.0  # Perfect correlation with itself
    
    for i in range(1 if isinstance(reference, int) and reference == 0 else 0, n_frames):
        if i == ref_idx:
            continue
            
        dy, dx, peak = _phase_correlation_shift(
            ref_frame, movie_array[i], upsample_factor=upsample_factor
        )
        shifts[i] = [dy, dx]
        quality[i] = peak
    
    # Compute cumulative drift (integrate shifts)
    cumulative_drift = np.cumsum(shifts, axis=0)
    
    return DriftResult(
        shifts=shifts,
        cumulative_drift=cumulative_drift,
        method='xcorr',
        reference_frame=ref_idx if ref_idx >= 0 else 0,
        per_frame_quality=quality,
    )


def _match_particles_greedy(
    coords1: np.ndarray,
    coords2: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match particles between two frames using greedy nearest-neighbor.
    
    Args:
        coords1: Coordinates in frame 1, shape (N1, 2)
        coords2: Coordinates in frame 2, shape (N2, 2)
        max_distance: Maximum matching distance
        
    Returns:
        Tuple of (matched_indices_1, matched_indices_2)
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    # Compute distance matrix
    diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    
    matched1 = []
    matched2 = []
    used2 = set()
    
    # Sort by minimum distance for deterministic matching
    for i in range(len(coords1)):
        min_dist = np.min(distances[i])
        if min_dist > max_distance:
            continue
            
        j = np.argmin(distances[i])
        if j in used2:
            # Try next closest
            sorted_j = np.argsort(distances[i])
            found = False
            for alt_j in sorted_j:
                if alt_j not in used2 and distances[i, alt_j] <= max_distance:
                    j = alt_j
                    found = True
                    break
            if not found:
                continue
        
        matched1.append(i)
        matched2.append(j)
        used2.add(j)
    
    return np.array(matched1, dtype=np.int64), np.array(matched2, dtype=np.int64)


def estimate_drift_particles(
    movie: list[Frame] | np.ndarray,
    detection_params: dict[str, Any] | None = None,
    match_radius: float = 10.0,
) -> DriftResult:
    """Estimate drift using particle detection and matching.
    
    This method detects particles in each frame and matches them between
    consecutive frames. The median displacement of matched particles
    provides a robust drift estimate.
    
    Args:
        movie: Input movie as list[Frame] or 3D array
        detection_params: Parameters for detect_particles() (default: auto threshold)
        match_radius: Maximum distance for particle matching (default: 10.0 pixels)
        
    Returns:
        DriftResult with shifts, cumulative_drift, and match counts
        
    Raises:
        ValueError: If movie is empty or no particles detected
    """
    movie_array = _movie_to_array(movie)
    n_frames = movie_array.shape[0]
    
    if detection_params is None:
        detection_params = {'thresh': None, 'kernel_size': 5, 'min_prom': None}
    
    # Detect particles in each frame
    all_coords = []
    for i in range(n_frames):
        result = detect_particles(movie_array[i], **detection_params)
        all_coords.append(result.coordinates)
    
    # Estimate frame-to-frame shifts
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    match_counts = np.zeros(n_frames, dtype=np.int64)
    match_counts[0] = len(all_coords[0]) if len(all_coords) > 0 else 0
    
    for i in range(1, n_frames):
        coords_prev = all_coords[i - 1]
        coords_curr = all_coords[i]
        
        if len(coords_prev) == 0 or len(coords_curr) == 0:
            # No particles to match: use previous shift or zero
            shifts[i] = shifts[i - 1] if i > 1 else [0, 0]
            match_counts[i] = 0
            continue
        
        # Match particles between frames
        idx1, idx2 = _match_particles_greedy(coords_prev, coords_curr, match_radius)
        
        if len(idx1) < 2:
            # Not enough matches: use previous shift
            shifts[i] = shifts[i - 1] if i > 1 else [0, 0]
            match_counts[i] = len(idx1)
            continue
        
        # Compute median displacement
        displacements = coords_curr[idx2] - coords_prev[idx1]
        median_shift = np.median(displacements, axis=0)  # [dx, dy]
        
        # Store as [dy, dx] to match convention
        shifts[i] = [median_shift[1], median_shift[0]]
        match_counts[i] = len(idx1)
    
    # Compute cumulative drift
    cumulative_drift = np.cumsum(shifts, axis=0)
    
    return DriftResult(
        shifts=shifts,
        cumulative_drift=cumulative_drift,
        method='particles',
        reference_frame=0,
        per_frame_quality=match_counts.astype(np.float64),
    )


def correct_drift(
    movie: list[Frame] | np.ndarray,
    drift: np.ndarray,
    mode: str = "constant",
) -> list[Frame] | np.ndarray:
    """Apply drift correction to a movie.
    
    Args:
        movie: Input movie as list[Frame] or 3D array
        drift: Drift trajectory to correct, shape (N, 2) with [dy, dx]
               (negative of estimated drift to undo it)
        mode: Interpolation mode for shift ('constant', 'reflect', 'wrap', 'nearest')
        
    Returns:
        Drift-corrected movie in same format as input
        
    Raises:
        ValueError: If drift shape doesn't match movie
    """
    movie_array = _movie_to_array(movie)
    n_frames = movie_array.shape[0]
    
    if len(drift) != n_frames:
        raise ValueError(f"Drift length {len(drift)} doesn't match movie length {n_frames}")
    
    if drift.ndim != 2 or drift.shape[1] != 2:
        raise ValueError(f"Drift must be shape (N, 2), got {drift.shape}")
    
    # Apply shift correction to each frame
    corrected = np.zeros_like(movie_array)
    
    for i in range(n_frames):
        # Shift by negative of drift to correct
        shift_vector = -drift[i]  # [dy, dx]
        corrected[i] = shift(
            movie_array[i],
            shift=shift_vector,
            mode=mode,
            cval=np.mean(movie_array[i]),
            order=1,  # Bilinear interpolation
        )
    
    # Return in same format as input
    if isinstance(movie, list) and len(movie) > 0 and isinstance(movie[0], Frame):
        # Reconstruct list[Frame]
        result = []
        for i, frame in enumerate(movie):
            if isinstance(frame, Frame):
                # Create new Frame with corrected data
                corrected_frame = Frame(
                    data=corrected[i],
                    meta=frame.meta,
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                )
                result.append(corrected_frame)
            else:
                result.append(corrected[i])
        return result
    
    return corrected


# GPU acceleration stubs (to be implemented when CuPy available)
def estimate_drift_xcorr_gpu(
    movie: Any,
    reference: int | Any = 0,
    upsample_factor: int = 10,
) -> DriftResult:
    """GPU-accelerated drift estimation using cross-correlation.
    
    Currently falls back to CPU implementation. Will be enhanced with
    CuPy FFT when GPU backend is fully integrated.
    """
    # For now, fall back to CPU
    return estimate_drift_xcorr(movie, reference=reference, upsample_factor=upsample_factor)


__all__ = [
    'DriftResult',
    'estimate_drift_xcorr',
    'estimate_drift_particles',
    'correct_drift',
    'estimate_drift_xcorr_gpu',
]
