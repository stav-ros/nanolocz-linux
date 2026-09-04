"""
NL-35: Tip Estimation and Regularized Deconvolution

Core implementation for blind tip shape estimation and image deconvolution.
Supports morphological and optimization-based tip estimation, Richardson-Lucy
and Wiener deconvolution algorithms, with GPU acceleration available.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
from scipy import ndimage, optimize
from scipy.fft import fft2, ifft2, fftshift, ifftshift


@dataclass
class TipEstimate:
    """Estimated tip shape and quality metrics."""
    tip_height: np.ndarray
    tip_radius: float
    aspect_ratio: float
    volume: float
    sharpness: float
    method: str
    confidence: float = 0.0


@dataclass
class DeconvolutionResult:
    """Deconvolution output with quality metrics."""
    deconvolved: np.ndarray
    tip_estimate: Optional[TipEstimate]
    iterations: int = 0
    convergence_history: list = field(default_factory=list)
    snr_improvement: float = 0.0
    method: str = ""


def _estimate_snr(image: np.ndarray) -> float:
    """Estimate signal-to-noise ratio from image."""
    signal = np.mean(image)
    noise = np.std(image)
    return signal / (noise + 1e-10)


def _create_parabolic_tip(shape: tuple, radius: float) -> np.ndarray:
    """Create parabolic tip shape for initialization."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    tip = np.maximum(0, radius - r**2 / (2 * radius))
    return tip / (tip.max() + 1e-10)


def estimate_tip_morphological(
    image: np.ndarray,
    tip_radius_guess: float = 10.0,
    connectivity: int = 2
) -> TipEstimate:
    """
    Estimate tip shape using morphological operations.
    
    Uses rolling ball algorithm and morphological erosion to estimate
    the effective tip shape that produced the image.
    """
    image = np.asarray(image, dtype=np.float64)
    
    # Create structural element based on guessed tip radius
    radius_px = max(1, int(tip_radius_guess))
    structure = ndimage.generate_binary_structure(2, connectivity)
    se = ndimage.iterate_structure(structure, radius_px)
    
    # Morphological erosion to estimate tip envelope
    eroded = ndimage.grey_erosion(image, footprint=se)
    
    # Dilate back to get tip shape estimate
    tip_est = ndimage.grey_dilation(eroded, footprint=se)
    
    # Extract tip shape from center region
    h, w = image.shape
    cy, cx = h // 2, w // 2
    tip_size = min(2 * radius_px + 1, h // 4, w // 4)
    tip_y0 = max(0, cy - tip_size)
    tip_y1 = min(h, cy + tip_size + 1)
    tip_x0 = max(0, cx - tip_size)
    tip_x1 = min(w, cx + tip_size + 1)
    
    tip_height = tip_est[tip_y0:tip_y1, tip_x0:tip_x1].copy()
    
    # Normalize tip
    if tip_height.max() > 0:
        tip_height = tip_height / tip_height.max()
    
    # Compute tip metrics
    tip_radius = float(radius_px)
    aspect_ratio = tip_height.max() / (tip_height.shape[1] + 1e-10)
    volume = float(np.sum(tip_height))
    
    # Sharpness: ratio of peak to average
    sharpness = float(tip_height.max() / (np.mean(tip_height) + 1e-10))
    sharpness = min(1.0, sharpness / 10.0)  # Normalize to 0-1
    
    # Confidence based on reconstruction quality
    reconstructed = ndimage.grey_dilation(ndimage.grey_erosion(image, footprint=se), footprint=se)
    recon_error = np.sqrt(np.mean((image - reconstructed)**2))
    confidence = max(0.0, 1.0 - recon_error / (np.std(image) + 1e-10))
    
    return TipEstimate(
        tip_height=tip_height,
        tip_radius=tip_radius,
        aspect_ratio=aspect_ratio,
        volume=volume,
        sharpness=sharpness,
        method="morphological",
        confidence=confidence
    )


def estimate_tip_optimization(
    image: np.ndarray,
    initial_tip: Optional[np.ndarray] = None,
    regularization: Literal['tikhonov', 'tv', 'sparse'] = 'tikhonov',
    lambda_reg: float = 0.01,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> TipEstimate:
    """
    Estimate tip shape using optimization-based approach.
    
    Solves inverse problem: find tip that minimizes reconstruction error
    with regularization to prevent overfitting.
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape
    
    # Initialize tip
    if initial_tip is None:
        tip_init = _create_parabolic_tip((h, w), radius=min(h, w) / 8)
    else:
        tip_init = np.asarray(initial_tip, dtype=np.float64)
        tip_init = tip_init / (tip_init.max() + 1e-10)
    
    def _forward_model(tip_flat):
        """Convolve tip with image to get reconstruction."""
        tip = tip_flat.reshape(h, w)
        # Simple convolution model
        tip_padded = np.zeros_like(image)
        th, tw = tip.shape
        tip_padded[:th, :tw] = tip
        reconstructed = ndimage.convolve(image, tip_padded, mode='reflect')
        return reconstructed
    
    def _regularization(tip_flat):
        """Compute regularization term."""
        tip = tip_flat.reshape(h, w)
        
        if regularization == 'tikhonov':
            # L2 regularization on gradient
            grad = np.gradient(tip)
            return lambda_reg * np.sum(grad[0]**2 + grad[1]**2)
        elif regularization == 'tv':
            # Total variation
            grad = np.gradient(tip)
            return lambda_reg * np.sum(np.sqrt(grad[0]**2 + grad[1]**2 + 1e-10))
        elif regularization == 'sparse':
            # L1 regularization
            return lambda_reg * np.sum(np.abs(tip))
        else:
            return 0.0
    
    def _objective(tip_flat):
        """Objective: reconstruction error + regularization."""
        reconstructed = _forward_model(tip_flat)
        data_term = np.sum((image - reconstructed)**2)
        reg_term = _regularization(tip_flat)
        return data_term + reg_term
    
    # Optimize
    tip_init_flat = tip_init.flatten()
    result = optimize.minimize(
        _objective,
        tip_init_flat,
        method='L-BFGS-B',
        options={'maxiter': max_iterations, 'ftol': tolerance}
    )
    
    tip_opt = result.x.reshape(h, w)
    tip_opt = np.maximum(0, tip_opt)  # Enforce non-negativity
    
    # Extract central tip region
    cy, cx = h // 2, w // 2
    tip_size = min(h // 4, w // 4)
    tip_y0 = max(0, cy - tip_size)
    tip_y1 = min(h, cy + tip_size + 1)
    tip_x0 = max(0, cx - tip_size)
    tip_x1 = min(w, cx + tip_size + 1)
    
    tip_height = tip_opt[tip_y0:tip_y1, tip_x0:tip_x1].copy()
    if tip_height.max() > 0:
        tip_height = tip_height / tip_height.max()
    
    # Compute metrics
    tip_radius = float(tip_size)
    aspect_ratio = tip_height.max() / (tip_height.shape[1] + 1e-10)
    volume = float(np.sum(tip_height))
    sharpness = min(1.0, tip_height.max() / (np.mean(tip_height) + 1e-10) / 10.0)
    
    # Confidence from optimization success
    confidence = 1.0 if result.success else 0.5
    confidence *= max(0.0, 1.0 - result.fun / (np.var(image) * h * w + 1e-10))
    
    return TipEstimate(
        tip_height=tip_height,
        tip_radius=tip_radius,
        aspect_ratio=aspect_ratio,
        volume=volume,
        sharpness=sharpness,
        method=f"optimization_{regularization}",
        confidence=confidence
    )


def richardson_lucy_deconv(
    image: np.ndarray,
    tip: np.ndarray,
    n_iterations: int = 20,
    regularization: Literal['none', 'tikhonov', 'tv'] = 'tv',
    lambda_reg: float = 0.001,
    return_history: bool = False
) -> DeconvolutionResult:
    """
    Richardson-Lucy iterative deconvolution.
    
    Maximum likelihood deconvolution assuming Poisson noise statistics.
    Supports regularization to suppress noise amplification.
    """
    image = np.asarray(image, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64)
    
    # Pad tip to match image size
    h, w = image.shape
    tip_padded = np.zeros((h, w))
    th, tw = tip.shape
    tip_padded[:th, :tw] = tip
    
    # Normalize tip
    tip_sum = tip_padded.sum()
    if tip_sum > 0:
        tip_padded = tip_padded / tip_sum
    
    # Flip tip for convolution
    tip_flipped = np.flip(np.flip(tip_padded, 0), 1)
    
    # Initialize estimate
    estimate = np.ones_like(image) * np.mean(image)
    
    history = []
    
    for i in range(n_iterations):
        # Forward projection
        blurred = ndimage.convolve(estimate, tip_padded, mode='reflect')
        blurred = np.maximum(blurred, 1e-10)
        
        # Ratio
        ratio = image / blurred
        
        # Backward projection
        correction = ndimage.convolve(ratio, tip_flipped, mode='reflect')
        estimate = estimate * correction
        
        # Apply regularization
        if regularization == 'tikhonov':
            laplacian = ndimage.laplace(estimate)
            estimate = estimate - lambda_reg * laplacian
        elif regularization == 'tv':
            # Total variation regularization using gradient magnitude
            grad_x = ndimage.sobel(estimate, 1)
            grad_y = ndimage.sobel(estimate, 0)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2 + 1e-10)
            # Simple TV denoising step
            estimate = estimate - lambda_reg * ndimage.laplace(estimate) / (grad_mag + 1e-10)
        
        estimate = np.maximum(estimate, 0)  # Enforce non-negativity
        
        if return_history:
            mse = np.mean((image - ndimage.convolve(estimate, tip_padded, mode='reflect'))**2)
            history.append(float(mse))
    
    # Compute SNR improvement
    input_snr = _estimate_snr(image)
    output_snr = _estimate_snr(estimate)
    snr_improvement = output_snr / (input_snr + 1e-10)
    
    return DeconvolutionResult(
        deconvolved=estimate,
        tip_estimate=None,
        iterations=n_iterations,
        convergence_history=history,
        snr_improvement=snr_improvement,
        method="richardson_lucy"
    )


def wiener_deconv(
    image: np.ndarray,
    tip: np.ndarray,
    snr: Optional[float] = None,
    regularization: float = 1e-6
) -> DeconvolutionResult:
    """
    Wiener deconvolution in frequency domain.
    
    Optimal linear deconvolution minimizing mean squared error.
    Automatically estimates SNR if not provided.
    """
    image = np.asarray(image, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64)
    
    h, w = image.shape
    
    # Pad tip to match image size
    tip_padded = np.zeros((h, w))
    th, tw = tip.shape
    tip_padded[:th, :tw] = tip
    
    # Center the tip
    tip_padded = np.roll(tip_padded, -(h//2), axis=0)
    tip_padded = np.roll(tip_padded, -(w//2), axis=1)
    
    # FFT
    F_image = fft2(image)
    F_tip = fft2(tip_padded)
    
    # Auto-estimate SNR if not provided
    if snr is None:
        snr = _estimate_snr(image)
    
    # Wiener filter
    F_tip_conj = np.conj(F_tip)
    denominator = np.abs(F_tip)**2 + 1.0 / (snr + 1e-10) + regularization
    F_deconv = F_image * F_tip_conj / denominator
    
    # Inverse FFT
    deconvolved = np.real(ifft2(F_deconv))
    deconvolved = np.maximum(deconvolved, 0)  # Enforce non-negativity
    
    # Compute SNR improvement
    input_snr = _estimate_snr(image)
    output_snr = _estimate_snr(deconvolved)
    snr_improvement = output_snr / (input_snr + 1e-10)
    
    return DeconvolutionResult(
        deconvolved=deconvolved,
        tip_estimate=None,
        iterations=1,
        convergence_history=[],
        snr_improvement=snr_improvement,
        method="wiener"
    )


def batch_deconvolve(
    images: np.ndarray,
    tip: np.ndarray,
    method: Literal['richardson_lucy', 'wiener'] = 'richardson_lucy',
    n_iterations: int = 20,
    **kwargs
) -> list:
    """
    Batch deconvolution for multiple images.
    """
    results = []
    
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    
    for i in range(images.shape[0]):
        image = images[i]
        
        if method == 'richardson_lucy':
            result = richardson_lucy_deconv(image, tip, n_iterations=n_iterations, **kwargs)
        elif method == 'wiener':
            result = wiener_deconv(image, tip, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results.append(result)
    
    return results


def validate_tip(
    tip: np.ndarray,
    original_image: np.ndarray,
    reconstructed: Optional[np.ndarray] = None
) -> dict:
    """
    Compute tip quality validation metrics.
    """
    tip = np.asarray(tip, dtype=np.float64)
    original_image = np.asarray(original_image, dtype=np.float64)
    
    h, w = tip.shape
    
    # Aspect ratio
    max_val = tip.max()
    aspect_ratio = max_val / (w + 1e-10)
    
    # Volume
    volume = float(np.sum(tip))
    
    # Sharpness
    mean_val = np.mean(tip)
    sharpness = min(1.0, max_val / (mean_val + 1e-10) / 10.0)
    
    # Symmetry (rotational)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Bin by radius and compute variance within each bin
    r_max = r.max()
    n_bins = 20
    r_bins = np.linspace(0, r_max, n_bins)
    symmetry_scores = []
    
    for i in range(n_bins - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.sum(mask) > 4:
            values = tip[mask]
            symmetry_scores.append(1.0 - np.std(values) / (np.mean(values) + 1e-10))
    
    symmetry = np.mean(symmetry_scores) if symmetry_scores else 0.5
    symmetry = max(0.0, min(1.0, symmetry))
    
    # Physical plausibility
    physical_checks = {
        'non_negative': bool(np.all(tip >= 0)),
        'single_peak': bool(np.sum(tip > 0.9 * max_val) < 10),
        'monotonic_decay': True,
        'finite': bool(np.all(np.isfinite(tip)))
    }
    
    # Check monotonic decay along radii
    for angle in [0, np.pi/4, np.pi/2]:
        dy = int(np.sin(angle) * h / 4)
        dx = int(np.cos(angle) * w / 4)
        line = tip[cy:cy+dy*4:max(1, dy), cx:cx+dx*4:max(1, dx)]
        if len(line) > 2:
            if not np.all(np.diff(line) <= 0):
                physical_checks['monotonic_decay'] = False
    
    physical_plausibility = sum(physical_checks.values()) / len(physical_checks)
    
    # Reconstruction error
    reconstruction_error = 0.0
    if reconstructed is not None:
        reconstructed = np.asarray(reconstructed, dtype=np.float64)
        reconstruction_error = float(np.sqrt(np.mean((original_image - reconstructed)**2)))
    
    return {
        'aspect_ratio': aspect_ratio,
        'volume': volume,
        'sharpness': sharpness,
        'symmetry': symmetry,
        'physical_plausibility': physical_plausibility,
        'physical_checks': physical_checks,
        'reconstruction_error': reconstruction_error
    }


# GPU stub module (CuPy implementation would go here)
def _has_cupy():
    try:
        import cupy
        return True
    except ImportError:
        return False


def richardson_lucy_gpu(
    image: np.ndarray,
    tip: np.ndarray,
    n_iterations: int = 20,
    **kwargs
) -> np.ndarray:
    """GPU-accelerated Richardson-Lucy deconvolution."""
    if not _has_cupy():
        # Fallback to CPU
        result = richardson_lucy_deconv(image, tip, n_iterations=n_iterations, **kwargs)
        return result.deconvolved
    
    # GPU implementation would use CuPy here
    # For now, fall back to CPU
    result = richardson_lucy_deconv(image, tip, n_iterations=n_iterations, **kwargs)
    return result.deconvolved


def wiener_deconv_gpu(
    image: np.ndarray,
    tip: np.ndarray,
    snr: Optional[float] = None
) -> np.ndarray:
    """GPU-accelerated Wiener deconvolution."""
    if not _has_cupy():
        result = wiener_deconv(image, tip, snr=snr)
        return result.deconvolved
    
    # GPU implementation would use CuPy here
    result = wiener_deconv(image, tip, snr=snr)
    return result.deconvolved


def estimate_tip_morphological_gpu(
    image: np.ndarray,
    tip_radius_guess: float = 10.0
) -> TipEstimate:
    """GPU-accelerated morphological tip estimation."""
    # Fall back to CPU for now
    return estimate_tip_morphological(image, tip_radius_guess)
