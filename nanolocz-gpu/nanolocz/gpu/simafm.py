"""GPU-accelerated simulation AFM kernels.

This module provides CuPy-based implementations for:
- Height field computation from atomic coordinates
- Tip convolution with sample surface
- Noise injection models
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


def compute_height_field_gpu(
    atomic_coords: Any,
    atomic_radii: Any | None = None,
    output_shape: tuple[int, int] = (256, 256),
    pixel_size: float = 0.1,  # nm per pixel
    z_scale: float = 1.0,
    ctx: BackendContext | None = None,
) -> Any:
    """Compute height field from atomic coordinates (hard collision model).

    Simulates a simple hard-collision AFM where the tip follows the
    outermost surface defined by atomic spheres.

    Args:
        atomic_coords: Array of atomic coordinates (N, 3) with columns [x, y, z] in nm
        atomic_radii: Array of atomic radii (N,) in nm, or None for uniform radius
        output_shape: Output image shape (height, width) in pixels
        pixel_size: Size of each pixel in nm
        z_scale: Scaling factor for z-heights
        ctx: Backend context

    Returns:
        Height field image (H, W) in nm
    """
    if ctx is None:
        ctx = get_backend_context()

    coords = _validate_array(atomic_coords, ctx)
    
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"atomic_coords must be (N, 3), got {coords.shape}")

    xp = cp if CUPY_AVAILABLE and ctx.is_gpu else np

    # Default radii if not provided
    if atomic_radii is None:
        radii = xp.ones(len(coords), dtype=ctx.dtype) * 0.15  # ~1.5 Angstrom default
    else:
        radii = _validate_array(atomic_radii, ctx)

    # Create coordinate grid
    ny, nx = output_shape
    y_coords = xp.arange(ny, dtype=ctx.dtype) * pixel_size
    x_coords = xp.arange(nx, dtype=ctx.dtype) * pixel_size
    xx, yy = xp.meshgrid(x_coords, y_coords, indexing='xy')

    # Initialize height field
    height_field = xp.full(output_shape, -xp.inf, dtype=ctx.dtype)

    # For each atom, compute its contribution to the height field
    n_atoms = len(coords)
    for i in range(n_atoms):
        atom_x = float(coords[i, 0])
        atom_y = float(coords[i, 1])
        atom_z = float(coords[i, 2])
        atom_radius = float(radii[i])

        # Compute distance from each pixel to atom center (in xy plane)
        dx = xx - atom_x
        dy = yy - atom_y
        r_xy_sq = dx ** 2 + dy ** 2

        # Only consider pixels within the atom's projection
        mask = r_xy_sq <= atom_radius ** 2
        
        if xp.any(mask):
            # Height at each pixel: z + sqrt(r^2 - d^2)
            r_xy = xp.sqrt(r_xy_sq[mask])
            z_height = atom_z + xp.sqrt(atom_radius ** 2 - r_xy ** 2)
            
            # Update height field (take maximum)
            current_heights = height_field[mask]
            height_field[mask] = xp.maximum(current_heights, z_height)

    # Apply z-scale and handle empty regions
    height_field = xp.where(xp.isfinite(height_field), height_field * z_scale, 0.0)

    return height_field


def convolve_tip_gpu(
    sample_surface: Any,
    tip_shape: str = "sphere",
    tip_radius: float = 10.0,  # nm
    tip_angle: float = 30.0,  # degrees for cone
    ctx: BackendContext | None = None,
) -> Any:
    """Convolve sample surface with tip geometry (dilation operation).

    Simulates the finite tip size effect in AFM imaging using
    mathematical morphology (dilation).

    Args:
        sample_surface: Input height field (H, W) in nm
        tip_shape: Tip geometry: "sphere", "cone", "paraboloid", "pyramid"
        tip_radius: Tip radius in nm (for sphere/cone/paraboloid)
        tip_angle: Half-angle in degrees (for cone/pyramid)
        ctx: Backend context

    Returns:
        Convolved height field (H, W) in nm
    """
    if ctx is None:
        ctx = get_backend_context()

    surface = _validate_array(sample_surface, ctx)
    xp = cp if CUPY_AVAILABLE and ctx.is_gpu else np

    if surface.ndim != 2:
        raise ValueError(f"sample_surface must be 2D, got {surface.ndim}D")

    ny, nx = surface.shape
    
    # Create tip kernel based on shape
    kernel_size = int(2 * tip_radius / 0.1) + 1  # Assuming ~0.1 nm pixel size
    kernel_size = max(kernel_size, 5)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create coordinate grid for kernel
    cy, cx = kernel_size // 2, kernel_size // 2
    ky, kx = xp.meshgrid(
        xp.arange(kernel_size, dtype=ctx.dtype) - cy,
        xp.arange(kernel_size, dtype=ctx.dtype) - cx,
        indexing='ij'
    )

    # Generate tip height profile
    if tip_shape == "sphere":
        r_sq = kx ** 2 + ky ** 2
        tip_kernel = xp.where(
            r_sq <= tip_radius ** 2,
            tip_radius - xp.sqrt(tip_radius ** 2 - r_sq),
            -xp.inf
        )
    elif tip_shape == "cone":
        angle_rad = np.radians(tip_angle)
        r = xp.sqrt(kx ** 2 + ky ** 2)
        tip_kernel = xp.where(
            r <= tip_radius,
            r / np.tan(angle_rad),
            -xp.inf
        )
    elif tip_shape == "paraboloid":
        r_sq = kx ** 2 + ky ** 2
        tip_kernel = xp.where(
            r_sq <= tip_radius ** 2,
            r_sq / (2 * tip_radius),
            -xp.inf
        )
    elif tip_shape == "pyramid":
        angle_rad = np.radians(tip_angle)
        r = xp.maximum(xp.abs(kx), xp.abs(ky))
        tip_kernel = xp.where(
            r <= tip_radius,
            r / np.tan(angle_rad),
            -xp.inf
        )
    else:
        raise ValueError(f"Unknown tip_shape: {tip_shape}")

    # Perform dilation (maximum filter with tip kernel)
    # This is a simplified implementation; full morphological dilation
    # would require more sophisticated approach
    
    # Pad surface
    pad_y = kernel_size // 2
    pad_x = kernel_size // 2
    padded = xp.pad(surface, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')

    # Output
    result = xp.zeros_like(surface)

    # For each position, compute maximum of (surface + tip)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            tip_val = tip_kernel[dy, dx]
            if xp.isfinite(tip_val):
                shifted = padded[dy:dy + ny, dx:dx + nx]
                candidate = shifted + tip_val
                result = xp.maximum(result, candidate)

    return result


def add_thermal_noise_gpu(
    image: Any,
    amplitude: float = 0.1,  # nm
    correlation_length: float = 5.0,  # pixels
    ctx: BackendContext | None = None,
) -> Any:
    """Add thermal drift and vibration noise.

    Simulates thermal effects including slow drift and high-frequency
    vibrations.

    Args:
        image: Input height field (H, W) in nm
        amplitude: Noise amplitude in nm
        correlation_length: Spatial correlation length in pixels
        ctx: Backend context

    Returns:
        Image with thermal noise added
    """
    if ctx is None:
        ctx = get_backend_context()

    img = _validate_array(image, ctx)
    xp = cp if CUPY_AVAILABLE and ctx.is_gpu else np

    # Generate correlated noise using Gaussian filter
    random_noise = xp.random.normal(0, 1, img.shape, dtype=ctx.dtype)
    
    # Apply Gaussian smoothing for spatial correlation
    from scipy.ndimage import gaussian_filter
    sigma = max(correlation_length / 2.0, 0.5)
    
    if CUPY_AVAILABLE and ctx.is_gpu:
        # Convert to numpy for filtering, back to GPU
        noise_np = cp.asnumpy(random_noise)
        smoothed = gaussian_filter(noise_np, sigma=sigma)
        correlated_noise = cp.asarray(smoothed, dtype=ctx.dtype)
    else:
        correlated_noise = gaussian_filter(random_noise, sigma=sigma)

    return img + amplitude * correlated_noise


def add_shot_noise_gpu(
    image: Any,
    scale: float = 1.0,
    ctx: BackendContext | None = None,
) -> Any:
    """Add Poisson-distributed shot noise.

    Simulates detector shot noise following Poisson statistics.

    Args:
        image: Input height field (H, W)
        scale: Scaling factor for Poisson parameter
        ctx: Backend context

    Returns:
        Image with shot noise added
    """
    if ctx is None:
        ctx = get_backend_context()

    img = _validate_array(image, ctx)
    xp = cp if CUPY_AVAILABLE and ctx.is_gpu else np

    # Shift to positive values for Poisson
    img_shifted = img - xp.min(img) + 1
    
    # Scale and apply Poisson noise
    scaled = img_shifted * scale
    noisy = xp.random.poisson(scaled) / scale

    # Restore original offset
    return noisy + xp.min(img) - 1


def add_scan_artifacts_gpu(
    image: Any,
    line_noise_amplitude: float = 0.05,  # nm
    trace_retrace_offset: float = 0.02,  # nm
    ctx: BackendContext | None = None,
) -> Any:
    """Simulate scan line artifacts.

    Adds realistic AFM artifacts including:
    - Line-to-line noise variations
    - Trace/retrace mismatch

    Args:
        image: Input height field (H, W) in nm
        line_noise_amplitude: Amplitude of line noise in nm
        trace_retrace_offset: Offset between trace and retrace scans
        ctx: Backend context

    Returns:
        Image with scan artifacts added
    """
    if ctx is None:
        ctx = get_backend_context()

    img = _validate_array(image, ctx)
    xp = cp if CUPY_AVAILABLE and ctx.is_gpu else np

    ny, nx = img.shape
    result = img.copy()

    # Add line-to-line noise (each row has a constant offset)
    line_offsets = xp.random.normal(0, line_noise_amplitude, ny, dtype=ctx.dtype)
    result = result + line_offsets[:, None]

    # Add trace/retrace offset (alternating lines)
    trace_retrace = xp.zeros(ny, dtype=ctx.dtype)
    trace_retrace[::2] = trace_retrace_offset / 2
    trace_retrace[1::2] = -trace_retrace_offset / 2
    result = result + trace_retrace[:, None]

    return result


def simulate_afm_image_gpu(
    atomic_coords: Any,
    atomic_radii: Any | None = None,
    output_shape: tuple[int, int] = (256, 256),
    pixel_size: float = 0.1,
    tip_radius: float = 10.0,
    tip_shape: str = "sphere",
    thermal_amplitude: float = 0.1,
    scan_noise_amplitude: float = 0.05,
    add_noise: bool = True,
    ctx: BackendContext | None = None,
) -> Any:
    """Full AFM image simulation pipeline.

    Combines height field computation, tip convolution, and noise
    addition into a single convenient function.

    Args:
        atomic_coords: Atomic coordinates (N, 3) in nm
        atomic_radii: Atomic radii (N,) in nm
        output_shape: Output image shape (H, W)
        pixel_size: Pixel size in nm
        tip_radius: Tip radius in nm
        tip_shape: Tip geometry
        thermal_amplitude: Thermal noise amplitude in nm
        scan_noise_amplitude: Scan line noise amplitude in nm
        add_noise: Whether to add noise artifacts
        ctx: Backend context

    Returns:
        Simulated AFM image (H, W) in nm
    """
    if ctx is None:
        ctx = get_backend_context()

    # Step 1: Compute height field from atoms
    height_field = compute_height_field_gpu(
        atomic_coords=atomic_coords,
        atomic_radii=atomic_radii,
        output_shape=output_shape,
        pixel_size=pixel_size,
        ctx=ctx,
    )

    # Step 2: Convolve with tip
    convolved = convolve_tip_gpu(
        sample_surface=height_field,
        tip_shape=tip_shape,
        tip_radius=tip_radius,
        ctx=ctx,
    )

    # Step 3: Add noise if requested
    if add_noise:
        result = add_thermal_noise_gpu(
            convolved,
            amplitude=thermal_amplitude,
            ctx=ctx,
        )
        result = add_scan_artifacts_gpu(
            result,
            line_noise_amplitude=scan_noise_amplitude,
            ctx=ctx,
        )
    else:
        result = convolved

    return result


# Export public API
__all__ = [
    'compute_height_field_gpu',
    'convolve_tip_gpu',
    'add_thermal_noise_gpu',
    'add_shot_noise_gpu',
    'add_scan_artifacts_gpu',
    'simulate_afm_image_gpu',
]
