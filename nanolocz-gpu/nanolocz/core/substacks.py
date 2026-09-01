"""Particle substack extraction from AFM movies.

This module provides functions for extracting particle-centered substacks
from drift-corrected AFM movies for downstream classification and averaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.ndimage import shift, map_coordinates

from nanolocz.core.types import Frame, Localizations, ParticleStack


def _apply_mask_to_data(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a 2D or per-frame mask to extraction data which may be 3D or 4D.
    
    Parameters
    ----------
    data : np.ndarray
        Data with shape either (n_particles, H, W) or (n_particles, n_frames, H, W)
    mask : np.ndarray
        Either 2D (H, W) or 3D (n_frames, H, W) mask
        
    Returns
    -------
    np.ndarray
        Masked data (in-place multiplication)
        
    Raises
    ------
    ValueError
        If mask dimensions are incompatible with data
    """
    if mask is None:
        return data

    # Ensure mask is numpy array
    mask = np.asarray(mask)

    if data.ndim == 3:  # (n_particles, H, W)
        if mask.ndim == 2:
            data *= mask[None, :, :]
        elif mask.ndim == 3:
            # mask per frame but data has no per-frame axis
            if mask.shape[0] == 1:
                data *= mask[0][None, :, :]
            else:
                raise ValueError(
                    "Provided mask has a frame axis but extracted data is 2D per particle; "
                    "provide 2D mask."
                )
        else:
            raise ValueError("mask must be 2D or 3D")
    elif data.ndim == 4:  # (n_particles, n_frames, H, W)
        if mask.ndim == 2:
            data *= mask[None, None, :, :]
        elif mask.ndim == 3:
            # mask expected shape (n_frames, H, W)
            if mask.shape[0] != data.shape[1]:
                raise ValueError(
                    f"Frame dimension of mask ({mask.shape[0]}) does not match "
                    f"extracted data frames ({data.shape[1]})"
                )
            data *= mask[None, :, :, :]
        else:
            raise ValueError("mask must be 2D or 3D")
    else:
        raise ValueError(f"Unexpected data ndim: expected 3 or 4, got {data.ndim}")

    return data


def _movie_to_array(movie: list[Frame] | np.ndarray) -> np.ndarray:
    """Convert movie to 3D numpy array (frames, height, width)."""
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


def _localizations_to_dict(localizations: Localizations) -> dict[int, list[tuple[float, float]]]:
    """Convert Localizations to dict mapping frame_index -> list of (x, y) coordinates."""
    result: dict[int, list[tuple[float, float]]] = {}
    for i, (xy, frame_idx) in enumerate(zip(localizations.xy, localizations.frame_index)):
        if frame_idx not in result:
            result[frame_idx] = []
        result[frame_idx].append(xy)
    return result


def _get_unique_particles(localizations: Localizations) -> list[tuple[int, list[tuple[float, float, int]]]]:
    """Get unique particle IDs with their detections across frames.
    
    Returns list of (particle_id, [(x, y, frame_idx), ...]) tuples.
    Simple tracking: particles are identified by proximity across frames.
    """
    # Group by frame
    frame_detections = _localizations_to_dict(localizations)
    
    # Simple nearest-neighbor tracking
    tracks: dict[int, list[tuple[float, float, int]]] = {}
    next_track_id = 0
    
    for frame_idx in sorted(frame_detections.keys()):
        detections = frame_detections[frame_idx]
        
        if frame_idx == 0:
            # First frame: create new tracks
            for xy in detections:
                tracks[next_track_id] = [(xy[0], xy[1], frame_idx)]
                next_track_id += 1
        else:
            # Match to existing tracks
            used = set()
            for xy in detections:
                best_track = None
                best_dist = float('inf')
                
                for track_id, track_locs in tracks.items():
                    if track_id in used:
                        continue
                    
                    # Get last position
                    last_x, last_y, _ = track_locs[-1]
                    dist = np.sqrt((xy[0] - last_x)**2 + (xy[1] - last_y)**2)
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_track = track_id
                
                # Link if within threshold (e.g., 10 pixels)
                if best_track is not None and best_dist < 10.0:
                    tracks[best_track].append((xy[0], xy[1], frame_idx))
                    used.add(best_track)
                else:
                    # Start new track
                    tracks[next_track_id] = [(xy[0], xy[1], frame_idx)]
                    next_track_id += 1
    
    # Convert to list, filtering single-frame detections
    result = [(tid, locs) for tid, locs in tracks.items() if len(locs) >= 1]
    return result


def extract_particle_substacks(
    movie: list[Frame] | np.ndarray,
    localizations: Localizations,
    patch_size: tuple[int, int] = (32, 32),
    mask: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ParticleStack:
    """Extract particle-centered substacks from a movie.
    
    Args:
        movie: Input movie as list of Frames or 3D array (frames, height, width)
        localizations: Particle localizations with xy coordinates and frame indices
        patch_size: Size of extracted patches (height, width)
        mask: Optional binary or soft mask to apply to substacks
        progress_callback: Optional callback(current, total) for progress reporting
        
    Returns:
        ParticleStack with:
            - data: 4D array (particles, frames, height, width) or 3D if single detection per particle
            - centers_xy: Original (x, y) coordinates used for extraction
            - frame_index: Frame index for each substack
            - box_size: Patch size
            
    Notes:
        - Particles are tracked across frames using simple nearest-neighbor linking
        - Missing detections result in gaps (zeros) in the time series
        - Mask is applied identically to all substacks if provided
    """
    movie_arr = _movie_to_array(movie)
    n_frames, height, width = movie_arr.shape
    patch_h, patch_w = patch_size
    
    # Get particle tracks
    tracks = _get_unique_particles(localizations)
    n_particles = len(tracks)
    
    if n_particles == 0:
        # Return empty ParticleStack with proper shape
        return ParticleStack(
            data=np.zeros((0, 0, 0), dtype=np.float64),
            centers_xy=[],
            frame_index=[],
            box_size=None
        )
    
    # Determine output shape: (n_particles, max_frames_per_particle, patch_h, patch_w)
    max_frames = max(len(locs) for _, locs in tracks)
    
    # Extract substacks
    substacks = np.zeros((n_particles, max_frames, patch_h, patch_w), dtype=np.float64)
    centers_list = []
    frame_indices = []
    
    half_h, half_w = patch_h // 2, patch_w // 2
    
    for pid, (track_id, locs) in enumerate(tracks):
        if progress_callback is not None:
            progress_callback(pid, n_particles)
        
        for fidx, (x, y, frame_idx) in enumerate(locs):
            # Compute extraction bounds
            x_int, y_int = int(round(x)), int(round(y))
            x_start, x_end = x_int - half_w, x_int + half_w
            y_start, y_end = y_int - half_h, y_int + half_h
            
            # Handle boundary conditions with padding
            frame_data = movie_arr[frame_idx]
            
            # Check if center is within valid region
            if (x_start < 0 or x_end > width or y_start < 0 or y_end > height):
                # Skip or pad - here we skip by leaving zeros
                continue
            
            # Extract patch
            patch = frame_data[y_start:y_end, x_start:x_end].copy()
            
            # Apply mask if provided using robust helper
            if mask is not None:
                # Create a temporary 4D array for consistent masking
                temp_data = patch[None, None, :, :]  # (1, 1, H, W)
                if mask.shape == (patch_h, patch_w):
                    # 2D mask
                    masked = _apply_mask_to_data(temp_data, mask)
                    patch = masked[0, 0]
                else:
                    # Assume mask is full-frame, extract same region
                    mask_patch = mask[y_start:y_end, x_start:x_end]
                    masked = _apply_mask_to_data(temp_data, mask_patch)
                    patch = masked[0, 0]
            
            substacks[pid, fidx, :, :] = patch
            centers_list.append((x, y))
            frame_indices.append(frame_idx)
    
    # Ensure 4D output for storage compatibility: (n_particles, max_frames, patch_h, patch_w)
    # Do NOT squeeze even if single frame - keep 4D for save_particle_stacks compatibility
    
    return ParticleStack(
        data=substacks,
        centers_xy=centers_list,
        frame_index=frame_indices,
        box_size=patch_h if patch_h == patch_w else None
    )


def extract_drift_corrected_substacks(
    movie: list[Frame] | np.ndarray,
    localizations: Localizations,
    drift: np.ndarray,
    patch_size: tuple[int, int] = (32, 32),
    interpolation_order: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ParticleStack:
    """Extract drift-corrected particle substacks.
    
    Applies inverse drift shifts during extraction to align substacks without
    resampling the entire movie.
    
    Args:
        movie: Input movie as list of Frames or 3D array (frames, height, width)
        localizations: Particle localizations with xy coordinates and frame indices
        drift: Drift trajectory from estimate_drift_xcorr() or estimate_drift_particles(),
               shape (n_frames, 2) with columns [dy, dx]
        patch_size: Size of extracted patches (height, width)
        interpolation_order: Order for spline interpolation (0=nearest, 1=bilinear, 3=cubic)
        progress_callback: Optional callback(current, total) for progress reporting
        
    Returns:
        ParticleStack with drift-corrected, aligned substacks
        
    Notes:
        - Drift correction is applied by shifting extraction coordinates
        - Sub-pixel shifts use spline interpolation
        - Avoids full-movie resampling for memory efficiency
    """
    movie_arr = _movie_to_array(movie)
    n_frames, height, width = movie_arr.shape
    
    if drift.shape[0] != n_frames:
        raise ValueError(f"Drift must have {n_frames} frames, got {drift.shape[0]}")
    
    patch_h, patch_w = patch_size
    
    # Get particle tracks
    tracks = _get_unique_particles(localizations)
    n_particles = len(tracks)
    
    if n_particles == 0:
        return ParticleStack(
            data=np.array([]),
            centers_xy=[],
            frame_index=[],
            box_size=None
        )
    
    max_frames = max(len(locs) for _, locs in tracks)
    
    substacks = np.zeros((n_particles, max_frames, patch_h, patch_w), dtype=np.float64)
    centers_list = []
    frame_indices = []
    
    half_h, half_w = patch_h // 2, patch_w // 2
    
    for pid, (track_id, locs) in enumerate(tracks):
        if progress_callback is not None:
            progress_callback(pid, n_particles)
        
        for fidx, (x, y, frame_idx) in enumerate(locs):
            # Apply inverse drift correction
            drift_dy, drift_dx = drift[frame_idx]
            x_corrected = x - drift_dx
            y_corrected = y - drift_dy
            
            # Extract with sub-pixel precision using interpolation
            x_center = x_corrected
            y_center = y_corrected
            
            # Create coordinate grids for the patch
            y_coords = np.arange(y_center - half_h, y_center + half_h)
            x_coords = np.arange(x_center - half_w, x_center + half_w)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Check bounds
            if (np.min(xx) < 0 or np.max(xx) >= width or 
                np.min(yy) < 0 or np.max(yy) >= height):
                # Out of bounds - leave as zeros
                centers_list.append((x, y))
                frame_indices.append(frame_idx)
                continue
            
            # Interpolate at sub-pixel positions
            if interpolation_order == 0:
                # Nearest neighbor
                yy_round = np.round(yy).astype(int)
                xx_round = np.round(xx).astype(int)
                patch = movie_arr[frame_idx][yy_round, xx_round]
            else:
                # Spline interpolation
                coordinates = np.vstack([yy.ravel(), xx.ravel()])
                patch_flat = map_coordinates(
                    movie_arr[frame_idx],
                    coordinates,
                    order=interpolation_order,
                    mode='constant',
                    cval=0.0
                )
                patch = patch_flat.reshape(patch_h, patch_w)
            
            substacks[pid, fidx, :, :] = patch
            centers_list.append((x, y))
            frame_indices.append(frame_idx)
    
    if max_frames == 1:
        substacks = substacks[:, 0, :, :]
    
    return ParticleStack(
        data=substacks,
        centers_xy=centers_list,
        frame_index=frame_indices,
        box_size=patch_h if patch_h == patch_w else None
    )


def create_gaussian_mask(
    patch_size: tuple[int, int],
    sigma: float | None = None,
) -> np.ndarray:
    """Create a Gaussian soft mask for substack extraction.
    
    Args:
        patch_size: Size of mask (height, width)
        sigma: Standard deviation of Gaussian; defaults to patch_size/6
        
    Returns:
        2D Gaussian mask with values in [0, 1]
    """
    patch_h, patch_w = patch_size
    
    if sigma is None:
        sigma = min(patch_h, patch_w) / 6.0
    
    y = np.arange(patch_h) - patch_h / 2
    x = np.arange(patch_w) - patch_w / 2
    yy, xx = np.meshgrid(y, x, indexing='ij')
    
    r_squared = xx**2 + yy**2
    mask = np.exp(-r_squared / (2 * sigma**2))
    
    return mask


def batch_extract_substacks(
    movie: list[Frame] | np.ndarray,
    localizations: Localizations,
    patch_size: tuple[int, int] = (32, 32),
    batch_size: int = 100,
    drift: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ParticleStack]:
    """Extract substacks in batches for memory efficiency.
    
    Args:
        movie: Input movie
        localizations: Particle localizations
        patch_size: Patch size (height, width)
        batch_size: Number of particles per batch
        drift: Optional drift trajectory for correction
        mask: Optional mask to apply
        progress_callback: Progress callback
        
    Yields:
        ParticleStack objects for each batch
    """
    tracks = _get_unique_particles(localizations)
    n_particles = len(tracks)
    
    for start_idx in range(0, n_particles, batch_size):
        end_idx = min(start_idx + batch_size, n_particles)
        batch_tracks = tracks[start_idx:end_idx]
        
        # Create subset of localizations for this batch
        batch_xy = []
        batch_frame_idx = []
        
        for _, (_, locs) in enumerate(batch_tracks):
            for x, y, frame_idx in locs:
                batch_xy.append((x, y))
                batch_frame_idx.append(frame_idx)
        
        batch_localizations = Localizations(
            xy=batch_xy,
            frame_index=batch_frame_idx
        )
        
        if drift is not None:
            substack = extract_drift_corrected_substacks(
                movie, batch_localizations, drift, patch_size,
                progress_callback=progress_callback
            )
        else:
            substack = extract_particle_substacks(
                movie, batch_localizations, patch_size, mask,
                progress_callback=progress_callback
            )
        
        yield substack
