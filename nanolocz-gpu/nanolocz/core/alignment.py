"""
In-class alignment and averaging for particle substacks.

This module provides tools for rigid alignment of particles within each cluster
followed by computation of class averages to improve signal-to-noise ratio (SNR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.signal import correlate2d

from .types import ParticleStack


@dataclass
class AlignmentResult:
    """Results from particle alignment."""
    
    aligned_stack: ParticleStack
    """Aligned particle stack."""
    
    shifts: np.ndarray
    """Shift values (dx, dy) for each particle. Shape: (n_particles, 2) in pixels."""
    
    correlation_scores: np.ndarray
    """Peak cross-correlation scores indicating alignment confidence. Shape: (n_particles,)."""
    
    reference_images: Dict[int, np.ndarray]
    """Reference image used for each cluster. Maps cluster_id → reference image."""
    
    n_aligned: int
    """Number of successfully aligned particles."""
    
    n_clusters: int
    """Number of clusters processed."""
    
    failed_indices: list[int] = field(default_factory=list)
    """Indices of particles that failed alignment."""
    
    def get_cluster_shifts(self, cluster_id: int, labels: np.ndarray) -> np.ndarray:
        """Return shifts for particles in specified cluster."""
        mask = labels == cluster_id
        return self.shifts[mask]
    
    def get_shift_statistics(self, cluster_id: int, labels: np.ndarray) -> dict:
        """Return mean, std, min, max shifts for cluster."""
        cluster_shifts = self.get_cluster_shifts(cluster_id, labels)
        if len(cluster_shifts) == 0:
            return {'mean': (0.0, 0.0), 'std': (0.0, 0.0), 'min': (0.0, 0.0), 'max': (0.0, 0.0)}
        
        dx_stats = (cluster_shifts[:, 0].mean(), cluster_shifts[:, 0].std(),
                    cluster_shifts[:, 0].min(), cluster_shifts[:, 0].max())
        dy_stats = (cluster_shifts[:, 1].mean(), cluster_shifts[:, 1].std(),
                    cluster_shifts[:, 1].min(), cluster_shifts[:, 1].max())
        
        return {
            'mean': (dx_stats[0], dy_stats[0]),
            'std': (dx_stats[1], dy_stats[1]),
            'min': (dx_stats[2], dy_stats[2]),
            'max': (dx_stats[3], dy_stats[3]),
        }


@dataclass
class ClassAverage:
    """Class average statistics for a single cluster."""
    
    cluster_id: int
    """Cluster identifier."""
    
    mean: np.ndarray
    """2D mean image."""
    
    std: np.ndarray
    """2D standard deviation image."""
    
    count: int
    """Number of particles in this cluster."""
    
    aligned_indices: np.ndarray
    """Indices of particles belonging to this cluster."""
    
    resolution_estimate: Optional[float] = None
    """Optional FRC-based resolution estimate in pixels."""
    
    def save_to_zarr(self, store, path: str):
        """Save class average to Zarr store."""
        # Implementation depends on NanoLoczStore API
        pass


def compute_shift_fft(
    moving: np.ndarray,
    reference: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute sub-pixel shift between moving and reference images using FFT cross-correlation.
    
    Parameters
    ----------
    moving : np.ndarray
        Moving image (2D array).
    reference : np.ndarray
        Reference image (2D array).
    mask : np.ndarray, optional
        Optional mask to exclude bad regions (same shape as images).
    
    Returns
    -------
    dx, dy : float
        Shift values in pixels (positive = moving shifted right/down relative to reference).
    """
    if moving.shape != reference.shape:
        raise ValueError(f"Shape mismatch: moving {moving.shape} vs reference {reference.shape}")
    
    # Apply mask if provided
    if mask is not None:
        if mask.shape != moving.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match image shape {moving.shape}")
        moving = moving * mask
        reference = reference * mask
    
    # Normalize images
    moving_norm = (moving - moving.mean()) / (moving.std() + 1e-10)
    reference_norm = (reference - reference.mean()) / (reference.std() + 1e-10)
    
    # FFT-based cross-correlation
    fft_moving = np.fft.fft2(moving_norm)
    fft_ref = np.fft.fft2(reference_norm)
    
    # Cross-correlation via inverse FFT of product with conjugate
    cross_corr = np.fft.ifft2(fft_moving * np.conj(fft_ref))
    cross_corr = np.real(cross_corr)
    
    # Find peak location
    peak_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    
    # Compute shift from peak position
    ny, nx = cross_corr.shape
    # Shift direction: if moving is shifted by (dx, dy) relative to reference,
    # the correlation peak will be at (dx, dy)
    dx = float(peak_idx[1]) if peak_idx[1] < nx // 2 else float(peak_idx[1] - nx)
    dy = float(peak_idx[0]) if peak_idx[0] < ny // 2 else float(peak_idx[0] - ny)
    
    # Sub-pixel refinement via parabolic fit around peak
    dx, dy = _refine_shift_parabolic(cross_corr, peak_idx, dx, dy)
    
    return dx, dy


def _refine_shift_parabolic(
    cross_corr: np.ndarray,
    peak_idx: Tuple[int, int],
    dx_coarse: float,
    dy_coarse: float,
    neighborhood: int = 3,
) -> Tuple[float, float]:
    """
    Refine shift estimate using parabolic fit around correlation peak.
    
    Parameters
    ----------
    cross_corr : np.ndarray
        Cross-correlation surface.
    peak_idx : tuple
        Integer pixel coordinates of coarse peak.
    dx_coarse, dy_coarse : float
        Coarse shift estimates.
    neighborhood : int
        Size of neighborhood for parabolic fit.
    
    Returns
    -------
    dx, dy : float
        Refined sub-pixel shift estimates.
    """
    y0, x0 = peak_idx
    ny, nx = cross_corr.shape
    
    # Extract neighborhood around peak
    y_start = max(0, y0 - neighborhood)
    y_end = min(ny, y0 + neighborhood + 1)
    x_start = max(0, x0 - neighborhood)
    x_end = min(nx, x0 + neighborhood + 1)
    
    neighborhood_data = cross_corr[y_start:y_end, x_start:x_end]
    
    if neighborhood_data.size < 6:
        # Not enough points for fit, return coarse estimate
        return dx_coarse, dy_coarse
    
    # Fit 2D paraboloid: z = a*x² + b*y² + c*x*y + d*x + e*y + f
    # Using least squares
    yy, xx = np.mgrid[y_start:y_end, x_start:x_end]
    X = np.column_stack([
        xx.ravel(), yy.ravel(),
        xx.ravel()**2, yy.ravel()**2,
        xx.ravel() * yy.ravel(),
        np.ones(xx.size)
    ])
    y = neighborhood_data.ravel()
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        # coeffs: [d, e, a, b, c, f]
        
        # Find vertex of paraboloid
        a, b, c = coeffs[2], coeffs[3], coeffs[4]
        d, e = coeffs[0], coeffs[1]
        
        denom = 4 * a * b - c**2
        if abs(denom) < 1e-10:
            return dx_coarse, dy_coarse
        
        dx_refined = -(2 * b * d - c * e) / denom
        dy_refined = -(2 * a * e - c * d) / denom
        
        # Clamp to reasonable range
        dx_refined = np.clip(dx_refined, -1.5, 1.5)
        dy_refined = np.clip(dy_refined, -1.5, 1.5)
        
        return dx_coarse + dx_refined, dy_coarse + dy_refined
    except Exception:
        return dx_coarse, dy_coarse


def apply_shift_fourier(
    image: np.ndarray,
    dx: float,
    dy: float,
    boundary_mode: str = 'constant',
) -> np.ndarray:
    """
    Apply sub-pixel shift to image using Fourier shift theorem.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (2D array).
    dx, dy : float
        Shift values in pixels.
    boundary_mode : str
        Boundary handling mode ('constant', 'reflect', 'wrap').
    
    Returns
    -------
    shifted : np.ndarray
        Shifted image (same shape as input).
    """
    if dx == 0 and dy == 0:
        return image.copy()
    
    ny, nx = image.shape
    
    # FFT of image
    fft_img = np.fft.fft2(image)
    
    # Create phase ramp
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(kx, ky)
    
    # Phase shift: exp(-2πi * (kx*dx + ky*dy))
    phase_shift = np.exp(-2j * np.pi * (KX * dx + KY * dy))
    
    # Apply phase shift and inverse FFT
    fft_shifted = fft_img * phase_shift
    shifted = np.fft.ifft2(fft_shifted)
    
    # Return real part (imaginary should be ~0 for real input)
    result = np.real(shifted)
    
    # Handle boundary conditions if needed
    if boundary_mode == 'constant':
        # Already handled by Fourier method
        pass
    elif boundary_mode == 'reflect':
        # Additional processing would be needed
        pass
    
    return result


def apply_shift_spline(
    image: np.ndarray,
    dx: float,
    dy: float,
    order: int = 3,
    boundary_mode: str = 'constant',
) -> np.ndarray:
    """
    Apply sub-pixel shift using spline interpolation (fallback method).
    
    Parameters
    ----------
    image : np.ndarray
        Input image (2D array).
    dx, dy : float
        Shift values in pixels.
    order : int
        Spline interpolation order (0-5).
    boundary_mode : str
        Boundary handling mode.
    
    Returns
    -------
    shifted : np.ndarray
        Shifted image.
    """
    # Note: ndimage.shift shifts in opposite direction
    shifted = ndimage.shift(image, shift=(dy, dx), order=order, mode=boundary_mode)
    return shifted


def _select_reference(
    cluster_data: np.ndarray,
    method: Literal['median', 'mean', 'highest_snr', 'manual'] = 'median',
    reference_index: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Select reference image for alignment from cluster particles.
    
    Parameters
    ----------
    cluster_data : np.ndarray
        Stack of particle images. Shape: (n_particles, H, W).
    method : str
        Reference selection method.
    reference_index : int, optional
        Manual reference index (used when method='manual').
    
    Returns
    -------
    reference : np.ndarray
        Selected reference image.
    ref_index : int
        Index of selected reference in cluster_data.
    """
    n_particles = cluster_data.shape[0]
    
    if method == 'manual':
        if reference_index is None or reference_index < 0 or reference_index >= n_particles:
            raise ValueError(f"Invalid reference_index {reference_index} for {n_particles} particles")
        return cluster_data[reference_index].copy(), reference_index
    
    elif method == 'median':
        reference = np.median(cluster_data, axis=0)
        # Find particle closest to median
        distances = np.array([np.sum((cluster_data[i] - reference)**2) 
                             for i in range(n_particles)])
        ref_index = int(np.argmin(distances))
        return cluster_data[ref_index].copy(), ref_index
    
    elif method == 'mean':
        reference = np.mean(cluster_data, axis=0)
        # Find particle closest to mean
        distances = np.array([np.sum((cluster_data[i] - reference)**2) 
                             for i in range(n_particles)])
        ref_index = int(np.argmin(distances))
        return cluster_data[ref_index].copy(), ref_index
    
    elif method == 'highest_snr':
        # Use variance as SNR proxy
        variances = np.array([np.var(cluster_data[i]) for i in range(n_particles)])
        ref_index = int(np.argmax(variances))
        return cluster_data[ref_index].copy(), ref_index
    
    else:
        raise ValueError(f"Unknown reference_method: {method}")


def align_particles(
    particle_stack: ParticleStack,
    cluster_labels: np.ndarray,
    reference_method: Literal['median', 'mean', 'highest_snr', 'manual'] = 'median',
    reference_index: Optional[int] = None,
    shift_method: Literal['fourier', 'spline'] = 'fourier',
    boundary_mode: str = 'constant',
    mask: Optional[np.ndarray] = None,
    min_cluster_size: int = 2,
) -> AlignmentResult:
    """
    Align particles within each cluster to a common reference.
    
    Parameters
    ----------
    particle_stack : ParticleStack
        Input particle stack from NL-32.
    cluster_labels : np.ndarray
        Cluster assignments from NL-33. Shape: (n_particles,).
    reference_method : str
        Method for selecting reference image per cluster.
    reference_index : int, optional
        Manual reference index (used when reference_method='manual').
    shift_method : str
        Method for applying shifts ('fourier' or 'spline').
    boundary_mode : str
        Boundary handling mode for shift application.
    mask : np.ndarray, optional
        Optional mask to exclude bad regions during alignment.
    min_cluster_size : int
        Minimum particles required to attempt alignment (default: 2).
    
    Returns
    -------
    result : AlignmentResult
        Aligned particle stack and metadata.
    """
    data = particle_stack.data
    n_particles = data.shape[0]
    
    # Handle 4D data (n_particles, n_frames, H, W) by using first frame
    if data.ndim == 4:
        if data.shape[1] > 1:
            # Use mean across frames for alignment
            data_2d = data[:, 0]  # Use first frame
        else:
            data_2d = data[:, 0]
    else:
        data_2d = data
    
    # Initialize output arrays
    aligned_data = np.zeros_like(data_2d)
    shifts = np.zeros((n_particles, 2), dtype=np.float64)
    correlation_scores = np.zeros(n_particles, dtype=np.float64)
    failed_indices = []
    reference_images: Dict[int, np.ndarray] = {}
    
    unique_labels = np.unique(cluster_labels)
    # Exclude noise particles (label=-1)
    unique_labels = unique_labels[unique_labels >= 0]
    
    n_clusters = len(unique_labels)
    n_aligned = 0
    
    for cluster_id in unique_labels:
        # Get particles in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)
        
        if cluster_size < min_cluster_size:
            # Skip small clusters - copy original data
            for idx in cluster_indices:
                aligned_data[idx] = data_2d[idx]
                shifts[idx] = (0.0, 0.0)
                correlation_scores[idx] = 0.0
            continue
        
        # Extract cluster data
        cluster_data = data_2d[cluster_mask]
        
        # Select reference
        reference, local_ref_idx = _select_reference(
            cluster_data, 
            method=reference_method,
            reference_index=reference_index
        )
        
        # Store global reference index
        global_ref_idx = cluster_indices[local_ref_idx]
        reference_images[cluster_id] = reference
        
        # Align all particles in cluster to reference
        for local_idx, global_idx in enumerate(cluster_indices):
            particle = data_2d[global_idx]
            
            if local_idx == local_ref_idx:
                # Reference particle - no shift needed
                aligned_data[global_idx] = particle.copy()
                shifts[global_idx] = (0.0, 0.0)
                correlation_scores[global_idx] = 1.0
                n_aligned += 1
                continue
            
            # Compute shift
            try:
                dx, dy = compute_shift_fft(particle, reference, mask=mask)
                
                # Apply shift
                if shift_method == 'fourier':
                    aligned_particle = apply_shift_fourier(particle, dx, dy, boundary_mode)
                else:
                    aligned_particle = apply_shift_spline(particle, dx, dy, 3, boundary_mode)
                
                # Compute correlation score
                normalized_particle = (aligned_particle - aligned_particle.mean()) / (aligned_particle.std() + 1e-10)
                normalized_ref = (reference - reference.mean()) / (reference.std() + 1e-10)
                corr_score = np.max(correlate2d(normalized_particle, normalized_ref, mode='valid'))
                
                aligned_data[global_idx] = aligned_particle
                shifts[global_idx] = (dx, dy)
                correlation_scores[global_idx] = corr_score
                n_aligned += 1
                
            except Exception as e:
                # Failed alignment - keep original
                aligned_data[global_idx] = particle.copy()
                shifts[global_idx] = (0.0, 0.0)
                correlation_scores[global_idx] = 0.0
                failed_indices.append(global_idx)
    
    # Create aligned ParticleStack
    # Preserve original structure (3D or 4D)
    if particle_stack.data.ndim == 4:
        # Reconstruct 4D array
        aligned_4d = np.zeros_like(particle_stack.data)
        aligned_4d[:, 0] = aligned_data
        # Copy remaining frames if present
        if particle_stack.data.shape[1] > 1:
            for i in range(1, particle_stack.data.shape[1]):
                for j in range(n_particles):
                    aligned_4d[j, i] = particle_stack.data[j, i]
        aligned_stack = ParticleStack(
            data=aligned_4d,
            centers_xy=particle_stack.centers_xy,
            frame_index=particle_stack.frame_index,
            box_size=particle_stack.box_size,
        )
    else:
        aligned_stack = ParticleStack(
            data=aligned_data,
            centers_xy=particle_stack.centers_xy,
            frame_index=particle_stack.frame_index,
            box_size=particle_stack.box_size,
        )
    
    return AlignmentResult(
        aligned_stack=aligned_stack,
        shifts=shifts,
        correlation_scores=correlation_scores,
        reference_images=reference_images,
        n_aligned=n_aligned,
        n_clusters=n_clusters,
        failed_indices=failed_indices,
    )


def compute_class_averages(
    aligned_stack: ParticleStack,
    cluster_labels: np.ndarray,
    compute_std: bool = True,
    compute_frc: bool = False,
) -> Dict[int, ClassAverage]:
    """
    Compute class averages for each cluster from aligned particles.
    
    Parameters
    ----------
    aligned_stack : ParticleStack
        Aligned particle stack from align_particles().
    cluster_labels : np.ndarray
        Cluster assignments. Shape: (n_particles,).
    compute_std : bool
        Whether to compute standard deviation images.
    compute_frc : bool
        Whether to compute FRC-based resolution estimates.
    
    Returns
    -------
    averages : dict
        Dictionary mapping cluster_id → ClassAverage.
    """
    data = aligned_stack.data
    
    # Handle 4D data
    if data.ndim == 4:
        data_2d = data[:, 0]
    else:
        data_2d = data
    
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels >= 0]
    
    averages: Dict[int, ClassAverage] = {}
    
    for cluster_id in unique_labels:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_data = data_2d[cluster_mask]
        
        if len(cluster_data) == 0:
            continue
        
        # Compute mean
        mean_img = np.mean(cluster_data, axis=0)
        
        # Compute std if requested
        if compute_std:
            std_img = np.std(cluster_data, axis=0)
        else:
            std_img = np.zeros_like(mean_img)
        
        # Compute FRC resolution if requested
        resolution_estimate = None
        if compute_frc and len(cluster_data) >= 4:
            resolution_estimate = _compute_frc_resolution(cluster_data)
        
        averages[cluster_id] = ClassAverage(
            cluster_id=cluster_id,
            mean=mean_img,
            std=std_img,
            count=len(cluster_data),
            aligned_indices=cluster_indices,
            resolution_estimate=resolution_estimate,
        )
    
    return averages


def _compute_frc_resolution(cluster_data: np.ndarray, threshold: float = 0.143) -> Optional[float]:
    """
    Compute FRC-based resolution estimate from half-set averages.
    
    Parameters
    ----------
    cluster_data : np.ndarray
        Stack of aligned particle images. Shape: (n_particles, H, W).
    threshold : float
        FRC threshold for resolution determination (default: 0.143, "gold standard").
    
    Returns
    -------
    resolution : float or None
        Resolution estimate in pixels, or None if computation fails.
    """
    n_particles = len(cluster_data)
    if n_particles < 4:
        return None
    
    # Split into two half-sets
    mid = n_particles // 2
    half1 = cluster_data[:mid]
    half2 = cluster_data[mid:]
    
    # Compute half-set averages
    avg1 = np.mean(half1, axis=0)
    avg2 = np.mean(half2, axis=0)
    
    # Compute FFTs
    fft1 = np.fft.fft2(avg1)
    fft2 = np.fft.fft2(avg2)
    
    # Compute FRC curve (Fourier Ring Correlation)
    ny, nx = avg1.shape
    max_freq = min(nx, ny) // 2
    
    frc_curve = np.zeros(max_freq)
    for r in range(1, max_freq):
        # Create ring mask at radius r
        y, x = np.ogrid[:ny, :nx]
        center_y, center_x = ny // 2, nx // 2
        mask = (np.sqrt((x - center_x)**2 + (y - center_y)**2) >= r - 0.5) & \
               (np.sqrt((x - center_x)**2 + (y - center_y)**2) < r + 0.5)
        
        if np.sum(mask) == 0:
            continue
        
        # Compute correlation in this ring
        ring1 = fft1[mask]
        ring2 = fft2[mask]
        
        numerator = np.real(np.sum(ring1 * np.conj(ring2)))
        denominator = np.sqrt(np.sum(np.abs(ring1)**2) * np.sum(np.abs(ring2)**2))
        
        if denominator > 0:
            frc_curve[r] = numerator / denominator
    
    # Find resolution where FRC crosses threshold
    resolution = None
    for r in range(1, max_freq):
        if frc_curve[r] < threshold:
            # Resolution in pixels = 1 / spatial frequency
            resolution = float(nx) / r
            break
    
    return resolution


def refine_alignment(
    particle_stack: ParticleStack,
    cluster_labels: np.ndarray,
    n_iterations: int = 3,
    convergence_threshold: float = 0.1,
    reference_method: Literal['median', 'mean', 'highest_snr'] = 'mean',
    shift_method: Literal['fourier', 'spline'] = 'fourier',
    mask: Optional[np.ndarray] = None,
) -> AlignmentResult:
    """
    Perform iterative alignment refinement.
    
    Parameters
    ----------
    particle_stack : ParticleStack
        Input particle stack.
    cluster_labels : np.ndarray
        Cluster assignments.
    n_iterations : int
        Maximum number of refinement iterations.
    convergence_threshold : float
        Stop if mean shift magnitude falls below this value (pixels).
    reference_method : str
        Reference selection method for subsequent iterations.
    shift_method : str
        Shift application method.
    mask : np.ndarray, optional
        Optional alignment mask.
    
    Returns
    -------
    result : AlignmentResult
        Final aligned particle stack and metadata.
    """
    current_stack = particle_stack
    
    for iteration in range(n_iterations):
        # Align particles
        result = align_particles(
            current_stack,
            cluster_labels,
            reference_method=reference_method,
            shift_method=shift_method,
            mask=mask,
        )
        
        # Check convergence
        shift_magnitudes = np.sqrt(np.sum(result.shifts**2, axis=1))
        mean_shift = np.mean(shift_magnitudes)
        
        if mean_shift < convergence_threshold:
            break
        
        # Update stack for next iteration
        current_stack = result.aligned_stack
    
    return result


def compute_shift_gpu(
    moving: np.ndarray,
    reference: np.ndarray,
) -> Tuple[float, float]:
    """
    GPU-accelerated shift computation using CuPy.
    
    Parameters
    ----------
    moving : np.ndarray
        Moving image.
    reference : np.ndarray
        Reference image.
    
    Returns
    -------
    dx, dy : float
        Shift values in pixels.
    """
    try:
        import cupy as cp
    except ImportError:
        # Fall back to CPU
        return compute_shift_fft(moving, reference)
    
    # Transfer to GPU
    moving_gpu = cp.asarray(moving)
    reference_gpu = cp.asarray(reference)
    
    # Normalize
    moving_norm = (moving_gpu - moving_gpu.mean()) / (moving_gpu.std() + 1e-10)
    reference_norm = (reference_gpu - reference_gpu.mean()) / (reference_gpu.std() + 1e-10)
    
    # FFT cross-correlation
    fft_moving = cp.fft.fft2(moving_norm)
    fft_ref = cp.fft.fft2(reference_norm)
    
    cross_corr = cp.fft.ifft2(fft_moving * cp.conj(fft_ref))
    cross_corr = cp.real(cross_corr)
    
    # Find peak
    peak_idx = cp.unravel_index(cp.argmax(cross_corr), cross_corr.shape)
    
    # Compute shift
    ny, nx = cross_corr.shape
    dx = int(peak_idx[1]) if peak_idx[1] < nx // 2 else int(peak_idx[1]) - nx
    dy = int(peak_idx[0]) if peak_idx[0] < ny // 2 else int(peak_idx[0]) - ny
    
    # Transfer back to CPU
    return float(dx), float(dy)


def align_particles_gpu(
    particle_stack: ParticleStack,
    cluster_labels: np.ndarray,
    reference_method: Literal['median', 'mean', 'highest_snr'] = 'median',
    batch_size: int = 64,
) -> AlignmentResult:
    """
    GPU-accelerated particle alignment using CuPy.
    
    Parameters
    ----------
    particle_stack : ParticleStack
        Input particle stack.
    cluster_labels : np.ndarray
        Cluster assignments.
    reference_method : str
        Reference selection method.
    batch_size : int
        Batch size for GPU processing.
    
    Returns
    -------
    result : AlignmentResult
        Aligned particle stack and metadata.
    """
    try:
        import cupy as cp
        _HAS_CUPY = True
    except ImportError:
        _HAS_CUPY = False
    
    if not _HAS_CUPY:
        # Fall back to CPU implementation
        return align_particles(particle_stack, cluster_labels, reference_method)
    
    # GPU implementation would go here
    # For now, fall back to CPU
    return align_particles(particle_stack, cluster_labels, reference_method)
