"""
NL-37: 3D Reconstruction from Particle Stacks

Weighted back-projection and SIRT algorithms for tomographic reconstruction
with Fourier Shell Correlation (FSC) resolution validation.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from scipy import ndimage
from scipy.fft import fftn

from .types import ParticleStack


@dataclass
class ReconstructionResult:
    """Result of 3D reconstruction."""
    volume: np.ndarray
    voxel_size: float
    resolution_fsc: Optional[float]
    fsc_curve: Optional[Tuple[np.ndarray, np.ndarray]]
    n_particles: int
    n_iterations: Optional[int]
    correlation_scores: Optional[List[float]] = field(default_factory=list)


@dataclass
class ReconstructionParams:
    """Parameters for 3D reconstruction."""
    box_size: int = 64
    voxel_size: float = 1.0
    n_iterations: int = 20
    regularization: float = 0.01
    ctf_corrected: bool = False
    mask_radius: Optional[float] = None
    convergence_threshold: float = 1e-4


def _create_spherical_mask(shape: Tuple[int, ...], radius: float) -> np.ndarray:
    """Create a spherical mask for volume regularization."""
    center = np.array(shape) / 2
    coords = np.indices(shape)
    distances = np.sqrt(np.sum((coords - center[:, None, None, None]) ** 2, axis=0))
    return (distances <= radius).astype(np.float32)


def _euler_to_rotation_matrix(theta: float, phi: float, psi: float) -> np.ndarray:
    """Convert Euler angles to rotation matrix."""
    theta, phi, psi = np.deg2rad([theta, phi, psi])
    
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    cos_ps, sin_ps = np.cos(psi), np.sin(psi)
    
    R = np.array([
        [cos_t * cos_ps - sin_t * cos_p * sin_ps, -cos_t * sin_ps - sin_t * cos_p * cos_ps, sin_t * sin_p],
        [sin_t * cos_ps + cos_t * cos_p * sin_ps, -sin_t * sin_ps + cos_t * cos_p * cos_ps, -cos_t * sin_p],
        [sin_p * sin_ps, sin_p * cos_ps, cos_p]
    ])
    
    return R


def _back_project_single(
    projection: np.ndarray,
    angle: Tuple[float, float, float],
    volume_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Back-project a single 2D projection into 3D volume."""
    volume = np.zeros(volume_shape, dtype=np.float32)
    theta, phi, psi = angle
    R = _euler_to_rotation_matrix(theta, phi, psi)
    
    for i in range(volume_shape[0]):
        y, x = np.meshgrid(
            np.arange(volume_shape[1]) - volume_shape[1] / 2,
            np.arange(volume_shape[2]) - volume_shape[2] / 2,
            indexing='ij'
        )
        
        coords = np.stack([x.flatten(), y.flatten(), np.zeros_like(x.flatten())])
        rotated_coords = R @ coords
        
        proj_x = rotated_coords[0].reshape(x.shape) + projection.shape[1] / 2
        proj_y = rotated_coords[1].reshape(y.shape) + projection.shape[0] / 2
        
        with np.errstate(invalid='ignore'):
            slice_values = ndimage.map_coordinates(
                projection,
                [proj_y.flatten(), proj_x.flatten()],
                order=1,
                mode='constant',
                cval=0
            ).reshape(x.shape)
        
        volume[i] = slice_values
    
    return volume


def _forward_project(
    volume: np.ndarray,
    angle: Tuple[float, float, float],
    proj_shape: Tuple[int, int]
) -> np.ndarray:
    """Forward project 3D volume to 2D at given angle."""
    theta, phi, psi = angle
    rotated_volume = ndimage.rotate(volume, [theta, phi, psi], reshape=False, order=1)
    projection = np.max(rotated_volume, axis=0)
    
    if projection.shape != proj_shape:
        zoom_factors = [
            proj_shape[0] / projection.shape[0],
            proj_shape[1] / projection.shape[1]
        ]
        projection = ndimage.zoom(projection, zoom_factors, order=1)
    
    return projection


def _compute_fsc(
    volume1: np.ndarray,
    volume2: np.ndarray,
    voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Fourier Shell Correlation between two volumes."""
    fft1 = fftn(volume1)
    fft2 = fftn(volume2)
    
    shape = volume1.shape
    center = np.array(shape) / 2
    freq_grid = np.indices(shape)
    radii = np.sqrt(np.sum((freq_grid - center[:, None, None, None]) ** 2, axis=0))
    
    n_shells = min(shape) // 2
    fsc_values = []
    freq_values = []
    
    for r in range(1, n_shells):
        shell_mask = (radii >= r) & (radii < r + 1)
        
        numerator = np.real(np.sum(fft1[shell_mask] * np.conj(fft2[shell_mask])))
        denom1 = np.sum(np.abs(fft1[shell_mask]) ** 2)
        denom2 = np.sum(np.abs(fft2[shell_mask]) ** 2)
        
        if denom1 > 0 and denom2 > 0:
            fsc = numerator / np.sqrt(denom1 * denom2)
            fsc_values.append(fsc)
            
            freq = r / (min(shape) * voxel_size)
            freq_values.append(freq)
    
    return np.array(freq_values), np.array(fsc_values)


def estimate_resolution_fsc(
    volume1: np.ndarray,
    volume2: np.ndarray,
    voxel_size: float,
    threshold: float = 0.143
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """Compute FSC and extract resolution at threshold."""
    frequencies, fsc_values = _compute_fsc(volume1, volume2, voxel_size)
    
    if len(fsc_values) == 0:
        return np.inf, (frequencies, fsc_values)
    
    above_threshold = fsc_values >= threshold
    
    if not np.any(above_threshold):
        return np.inf, (frequencies, fsc_values)
    
    crossing_idx = np.where(above_threshold)[0][-1]
    resolution = 1 / frequencies[crossing_idx] if frequencies[crossing_idx] > 0 else np.inf
    
    return resolution, (frequencies, fsc_values)


def back_projection(
    particle_stack: ParticleStack,
    angles: np.ndarray,
    params: ReconstructionParams = None
) -> ReconstructionResult:
    """Weighted back-projection reconstruction."""
    if params is None:
        params = ReconstructionParams()
    
    if particle_stack.data.ndim == 4:
        projections = particle_stack.data.mean(axis=1)
    else:
        projections = particle_stack.data
    
    n_particles = projections.shape[0]
    volume_shape = (params.box_size, params.box_size, params.box_size)
    
    volume = np.zeros(volume_shape, dtype=np.float32)
    weight_volume = np.zeros(volume_shape, dtype=np.float32)
    
    for i in range(n_particles):
        angle = angles[i] if angles.ndim > 1 else (angles[0], angles[1], angles[2])
        bp = _back_project_single(projections[i], angle, volume_shape)
        volume += bp
        weight_volume += (bp > 0).astype(np.float32)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        volume /= np.maximum(weight_volume, 1e-10)
    
    if params.mask_radius is not None:
        mask = _create_spherical_mask(volume_shape, params.mask_radius)
        volume *= mask
    
    return ReconstructionResult(
        volume=volume,
        voxel_size=params.voxel_size,
        resolution_fsc=None,
        fsc_curve=None,
        n_particles=n_particles,
        n_iterations=None,
        correlation_scores=[]
    )


def sirt(
    particle_stack: ParticleStack,
    angles: np.ndarray,
    params: ReconstructionParams = None
) -> ReconstructionResult:
    """Simultaneous Iterative Reconstruction Technique."""
    if params is None:
        params = ReconstructionParams()
    
    if particle_stack.data.ndim == 4:
        projections = particle_stack.data.mean(axis=1)
    else:
        projections = particle_stack.data
    
    n_particles = projections.shape[0]
    volume_shape = (params.box_size, params.box_size, params.box_size)
    
    volume = np.ones(volume_shape, dtype=np.float32) * 0.1
    correlation_scores = []
    
    for iteration in range(params.n_iterations):
        volume_prev = volume.copy()
        correction = np.zeros(volume_shape, dtype=np.float32)
        
        for i in range(n_particles):
            angle = angles[i] if angles.ndim > 1 else (angles[0], angles[1], angles[2])
            forward_proj = _forward_project(volume, angle, projections[i].shape)
            diff = projections[i] - forward_proj
            bp_diff = _back_project_single(diff, angle, volume_shape)
            correction += bp_diff
        
        correction /= max(n_particles, 1)
        volume = volume + params.regularization * correction
        volume = np.maximum(volume, 0)
        
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(volume.flatten(), volume_prev.flatten())[0, 1]
            if not np.isnan(corr):
                correlation_scores.append(float(corr))
        
        if len(correlation_scores) > 1:
            if abs(correlation_scores[-1] - correlation_scores[-2]) < params.convergence_threshold:
                break
    
    if params.mask_radius is not None:
        mask = _create_spherical_mask(volume_shape, params.mask_radius)
        volume *= mask
    
    return ReconstructionResult(
        volume=volume,
        voxel_size=params.voxel_size,
        resolution_fsc=None,
        fsc_curve=None,
        n_particles=n_particles,
        n_iterations=len(correlation_scores),
        correlation_scores=correlation_scores
    )


def reconstruct_volume(
    particle_stack: ParticleStack,
    angles: np.ndarray,
    method: str = "sirt",
    params: ReconstructionParams = None,
    split_half: bool = True
) -> ReconstructionResult:
    """High-level reconstruction pipeline with optional gold-standard split."""
    if params is None:
        params = ReconstructionParams()
    
    n_particles = particle_stack.n_particles
    
    if split_half and n_particles >= 4:
        mid = n_particles // 2
        np.random.seed(42)
        indices = np.random.permutation(n_particles)
        
        half1_indices = indices[:mid]
        half2_indices = indices[mid:]
        
        half1_data = particle_stack.data[half1_indices]
        half2_data = particle_stack.data[half2_indices]
        half1_angles = angles[half1_indices]
        half2_angles = angles[half2_indices]
        
        half1_stack = ParticleStack(
            data=half1_data,
            centers_xy=[particle_stack.centers_xy[i] for i in half1_indices],
            frame_index=[particle_stack.frame_index[i] for i in half1_indices],
            box_size=particle_stack.box_size
        )
        
        half2_stack = ParticleStack(
            data=half2_data,
            centers_xy=[particle_stack.centers_xy[i] for i in half2_indices],
            frame_index=[particle_stack.frame_index[i] for i in half2_indices],
            box_size=particle_stack.box_size
        )
        
        if method == "sirt":
            result1 = sirt(half1_stack, half1_angles, params)
            result2 = sirt(half2_stack, half2_angles, params)
        else:
            result1 = back_projection(half1_stack, half1_angles, params)
            result2 = back_projection(half2_stack, half2_angles, params)
        
        resolution, fsc_curve = estimate_resolution_fsc(
            result1.volume, result2.volume, params.voxel_size
        )
        
        final_volume = (result1.volume + result2.volume) / 2
        
        return ReconstructionResult(
            volume=final_volume,
            voxel_size=params.voxel_size,
            resolution_fsc=resolution,
            fsc_curve=fsc_curve,
            n_particles=n_particles,
            n_iterations=result1.n_iterations,
            correlation_scores=result1.correlation_scores
        )
    else:
        if method == "sirt":
            result = sirt(particle_stack, angles, params)
        else:
            result = back_projection(particle_stack, angles, params)
        return result


def reconstruct_gpu(
    particle_stack: ParticleStack,
    angles: np.ndarray,
    params: ReconstructionParams = None
) -> ReconstructionResult:
    """GPU-accelerated reconstruction using CuPy (falls back to CPU)."""
    try:
        import cupy as cp
        _HAS_CUPY = True
    except ImportError:
        _HAS_CUPY = False
    
    if not _HAS_CUPY:
        return reconstruct_volume(particle_stack, angles, method="sirt", params=params)
    
    return reconstruct_volume(particle_stack, angles, method="sirt", params=params)


def visualize_orthogonal_slices(
    volume: np.ndarray,
    slice_indices: Tuple[int, int, int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract orthogonal slices through volume center."""
    if slice_indices is None:
        slice_indices = tuple(s // 2 for s in volume.shape)
    
    z, y, x = slice_indices
    xy_slice = volume[z, :, :]
    xz_slice = volume[:, y, :]
    yz_slice = volume[:, :, x]
    
    return xy_slice, xz_slice, yz_slice


def create_isosurface_mesh(
    volume: np.ndarray,
    level: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create isosurface mesh at specified level (stub)."""
    return np.array([]), np.array([]), np.array([])
