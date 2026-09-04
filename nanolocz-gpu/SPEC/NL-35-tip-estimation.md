# NL-35: Tip Estimation and Regularized Deconvolution

## Goals
- Estimate AFM tip shape from particle images or class averages
- Perform regularized deconvolution to recover true sample height
- Support multiple tip estimation methods (morphological, optimization-based)
- Implement Richardson-Lucy and Wiener deconvolution algorithms
- Validate tip quality with quantitative metrics
- GPU acceleration for iterative deconvolution

## Dependencies
- **NL-24**: Simulation AFM kernels (tip convolution forward model)
- **NL-34**: In-class alignment and averaging (input: aligned class averages)
- **NL-20**: NumPy/CuPy backend switch (GPU acceleration infrastructure)

## Acceptance Criteria
1. Blind tip estimation from single particles or class averages
2. Morphological tip estimation using erosion/dilation operations
3. Optimization-based tip estimation with regularization
4. Richardson-Lucy iterative deconvolution implementation
5. Wiener deconvolution with automatic noise estimation
6. Tip validation metrics (aspect ratio, volume, sharpness)
7. Batch processing for multiple particles
8. GPU-accelerated deconvolution (10x speedup for 10+ iterations)
9. All CPU tests pass with float64 reference accuracy
10. Comprehensive test suite with synthetic and realistic data

## API Design

### Core Module: `nanolocz/core/tip_estimation.py`

```python
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np

@dataclass
class TipEstimate:
    """Estimated tip shape and quality metrics."""
    tip_height: np.ndarray  # 2D tip shape array
    tip_radius: float  # Estimated tip radius of curvature
    aspect_ratio: float  # Height/width ratio
    volume: float  # Tip volume
    sharpness: float  # Tip sharpness metric (0-1)
    method: str  # Estimation method used
    confidence: float  # Confidence score (0-1)

@dataclass
class DeconvolutionResult:
    """Deconvolution output with quality metrics."""
    deconvolved: np.ndarray  # Deconvolved height image
    tip_estimate: TipEstimate  # Used tip estimate
    iterations: int  # Number of iterations (for iterative methods)
    convergence_history: list  # Loss/convergence history
    snr_improvement: float  # SNR improvement factor
    method: str  # Deconvolution method used

def estimate_tip_morphological(
    image: np.ndarray,
    tip_radius_guess: float = 10.0,
    connectivity: int = 2
) -> TipEstimate:
    """
    Estimate tip shape using morphological operations.
    
    Uses rolling ball algorithm and morphological erosion to estimate
    the effective tip shape that produced the image.
    
    Parameters
    ----------
    image : np.ndarray
        Input AFM height image (single particle or class average)
    tip_radius_guess : float
        Initial guess for tip radius of curvature in pixels
    connectivity : int
        Structural element connectivity (1, 2, or 3)
    
    Returns
    -------
    TipEstimate
        Estimated tip shape with quality metrics
    """
    ...

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
    
    Parameters
    ----------
    image : np.ndarray
        Input AFM height image
    initial_tip : np.ndarray, optional
        Initial tip guess (default: parabolic tip)
    regularization : str
        Regularization type: 'tikhonov' (L2), 'tv' (total variation), 'sparse' (L1)
    lambda_reg : float
        Regularization strength
    max_iterations : int
        Maximum optimization iterations
    tolerance : float
        Convergence tolerance
    
    Returns
    -------
    TipEstimate
        Optimized tip shape with quality metrics
    """
    ...

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
    
    Parameters
    ----------
    image : np.ndarray
        Blurred AFM image to deconvolve
    tip : np.ndarray
        Point spread function (estimated tip shape)
    n_iterations : int
        Number of RL iterations
    regularization : str
        Regularization type applied at each iteration
    lambda_reg : float
        Regularization strength
    return_history : bool
        Return convergence history
    
    Returns
    -------
    DeconvolutionResult
        Deconvolved image with metadata
    """
    ...

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
    
    Parameters
    ----------
    image : np.ndarray
        Blurred AFM image to deconvolve
    tip : np.ndarray
        Point spread function (estimated tip shape)
    snr : float, optional
        Signal-to-noise ratio (auto-estimated if None)
    regularization : float
        Small constant to prevent division by zero
    
    Returns
    -------
    DeconvolutionResult
        Deconvolved image with metadata
    """
    ...

def batch_deconvolve(
    images: np.ndarray,
    tip: np.ndarray,
    method: Literal['richardson_lucy', 'wiener'] = 'richardson_lucy',
    n_iterations: int = 20,
    **kwargs
) -> list[DeconvolutionResult]:
    """
    Batch deconvolution for multiple images.
    
    Parameters
    ----------
    images : np.ndarray
        Stack of images to deconvolve, shape (n_images, H, W)
    tip : np.ndarray
        Common tip shape for all images
    method : str
        Deconvolution method
    n_iterations : int
        Iterations for Richardson-Lucy (ignored for Wiener)
    **kwargs
        Additional arguments passed to deconvolution function
    
    Returns
    -------
    list[DeconvolutionResult]
        List of deconvolution results
    """
    ...

def validate_tip(
    tip: np.ndarray,
    original_image: np.ndarray,
    reconstructed: Optional[np.ndarray] = None
) -> dict:
    """
    Compute tip quality validation metrics.
    
    Parameters
    ----------
    tip : np.ndarray
        Estimated tip shape
    original_image : np.ndarray
        Original AFM image
    reconstructed : np.ndarray, optional
        Reconstructed image after deconvolution
    
    Returns
    -------
    dict
        Validation metrics including:
        - aspect_ratio: tip height/width ratio
        - volume: tip volume in cubic pixels
        - sharpness: edge sharpness metric (0-1)
        - symmetry: rotational symmetry score (0-1)
        - physical_plausibility: checks for non-physical features
        - reconstruction_error: RMS error if reconstructed provided
    """
    ...
```

### GPU Module: `nanolocz/gpu/tip_estimation.py`

```python
def richardson_lucy_gpu(
    image: np.ndarray,
    tip: np.ndarray,
    n_iterations: int = 20,
    **kwargs
) -> np.ndarray:
    """GPU-accelerated Richardson-Lucy deconvolution."""
    ...

def wiener_deconv_gpu(
    image: np.ndarray,
    tip: np.ndarray,
    snr: Optional[float] = None
) -> np.ndarray:
    """GPU-accelerated Wiener deconvolution."""
    ...

def estimate_tip_morphological_gpu(
    image: np.ndarray,
    tip_radius_guess: float = 10.0
) -> np.ndarray:
    """GPU-accelerated morphological tip estimation."""
    ...
```

## Test Plan

### File: `tests/test_tip_estimation_nl35.py`

```python
import pytest
import numpy as np
from nanolocz.core.tip_estimation import (
    estimate_tip_morphological,
    estimate_tip_optimization,
    richardson_lucy_deconv,
    wiener_deconv,
    batch_deconvolve,
    validate_tip,
    TipEstimate,
    DeconvolutionResult
)

class TestMorphologicalTipEstimation:
    def test_parabolic_tip_recovery(self):
        """Test recovery of known parabolic tip shape."""
        ...
    
    def test_conical_tip_recovery(self):
        """Test recovery of conical tip shape."""
        ...
    
    def test_tip_radius_sensitivity(self):
        """Test sensitivity to initial radius guess."""
        ...
    
    def test_noisy_image_robustness(self):
        """Test robustness to image noise."""
        ...

class TestOptimizationTipEstimation:
    def test_tikhonov_regularization(self):
        """Test L2-regularized tip estimation."""
        ...
    
    def test_total_variation_regularization(self):
        """Test TV-regularized tip estimation."""
        ...
    
    def test_sparse_regularization(self):
        """Test L1-regularized tip estimation."""
        ...
    
    def test_convergence_behavior(self):
        """Test optimization convergence."""
        ...
    
    def test_initialization_sensitivity(self):
        """Test sensitivity to initial tip guess."""
        ...

class TestRichardsonLucyDeconv:
    def test_basic_deconvolution(self):
        """Test basic RL deconvolution on synthetic data."""
        ...
    
    def test_iteration_convergence(self):
        """Test convergence with increasing iterations."""
        ...
    
    def test_regularization_effect(self):
        """Test effect of different regularization types."""
        ...
    
    def test_noise_amplification_suppression(self):
        """Test that regularization suppresses noise amplification."""
        ...
    
    def test_conservation_of_mass(self):
        """Test that total signal is approximately conserved."""
        ...

class TestWienerDeconv:
    def test_basic_wiener_deconv(self):
        """Test basic Wiener deconvolution."""
        ...
    
    def test_auto_snr_estimation(self):
        """Test automatic SNR estimation."""
        ...
    
    def test_known_snr_input(self):
        """Test with known SNR value."""
        ...
    
    def test_frequency_domain_correctness(self):
        """Verify frequency domain filtering is correct."""
        ...

class TestBatchDeconvolution:
    def test_batch_consistency(self):
        """Test batch results match individual processing."""
        ...
    
    def test_batch_efficiency(self):
        """Test batch processing is efficient."""
        ...
    
    def test_variable_image_count(self):
        """Test with different batch sizes."""
        ...

class TestTipValidation:
    def test_aspect_ratio_computation(self):
        """Test aspect ratio metric calculation."""
        ...
    
    def test_volume_computation(self):
        """Test tip volume calculation."""
        ...
    
    def test_sharpness_metric(self):
        """Test sharpness metric range and behavior."""
        ...
    
    def test_symmetry_metric(self):
        """Test rotational symmetry scoring."""
        ...
    
    def test_physical_plausibility_checks(self):
        """Test detection of non-physical tip features."""
        ...

class TestIntegration:
    def test_end_to_end_tip_estimation_deconv(self):
        """Test full pipeline: estimate tip → deconvolve → validate."""
        ...
    
    def test_with_class_averages_from_nl34(self):
        """Test integration with NL-34 class averages."""
        ...
    
    def test_with_simulated_afm_from_nl24(self):
        """Test with NL-24 simulated AFM images with known tip."""
        ...
    
    def test_tip_ground_truth_comparison(self):
        """Compare estimated tip to ground truth from simulation."""
        ...

class TestTipEstimationGPU:
    @pytest.mark.skipif(not cupy_available, reason="CuPy not available")
    def test_gpu_richardson_lucy_parity(self):
        """Test GPU RL matches CPU within tolerance."""
        ...
    
    @pytest.mark.skipif(not cupy_available, reason="CuPy not available")
    def test_gpu_wiener_parity(self):
        """Test GPU Wiener matches CPU within tolerance."""
        ...
    
    @pytest.mark.skipif(not cupy_available, reason="CuPy not available")
    def test_gpu_speedup(self):
        """Test GPU provides expected speedup."""
        ...
```

## Implementation Notes

### Tip Estimation Methods

1. **Morphological Approach**:
   - Use rolling ball algorithm with varying radii
   - Apply morphological erosion with structural elements
   - Fit parabolic/conical model to eroded surface
   - Fast but approximate

2. **Optimization Approach**:
   - Forward model: image = convolve(tip, true_surface) + noise
   - Minimize: ||image - convolve(tip, estimate)||² + λ·R(tip)
   - R(tip): regularization term (Tikhonov, TV, or sparse)
   - Slower but more accurate

### Deconvolution Algorithms

1. **Richardson-Lucy**:
   - Iterative: f_{k+1} = f_k · (h ⊗ (g / (h * f_k)))
   - Assumes Poisson noise
   - Monotonically increases likelihood
   - Requires regularization for stability

2. **Wiener**:
   - Closed-form: F{f} = F{h}* / (|F{h}|² + 1/SNR) · F{g}
   - Assumes Gaussian noise
   - Single step (no iteration)
   - Optimal for stationary signals

### GPU Acceleration Strategy

- FFT-based convolution/deconvolution on GPU
- CuPy's fft module for GPU FFTs
- Batch processing for multiple images
- Memory management for large stacks

## Deliverables Checklist

- [ ] `SPEC/NL-35-tip-estimation.md` (this file)
- [ ] `nanolocz/core/tip_estimation.py` — Core implementation
- [ ] `nanolocz/gpu/tip_estimation.py` — GPU kernels
- [ ] `tests/test_tip_estimation_nl35.py` — Test suite
- [ ] `SESSIONS/YYYY-MM-DD-NL-35.md` — Session handoff
- [ ] Update `STATUS.md` — Mark NL-35 as done
- [ ] Update `SPEC/tasks.md` — Update task list

## Success Metrics

- Tip radius estimation error < 15% on synthetic data
- Deconvolution SNR improvement > 3 dB for moderate noise
- Richardson-Lucy converges within 50 iterations
- GPU speedup > 10x for 10+ iterations on 256×256 images
- All validation metrics in physically plausible ranges
- Integration with NL-34 class averages works seamlessly
