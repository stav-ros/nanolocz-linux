# NL-37: 3D Reconstruction from Particle Stacks

## Status
- **State**: ready to start
- **Phase**: Phase 3 — LAFM+ Core Pipeline
- **Dependencies**: NL-34 (In-class alignment) ✓, NL-36 (Dynamics traces) ✓
- **Blocked by**: None

## Goals
1. Implement weighted back-projection algorithm for tomographic reconstruction
2. Implement SIRT (Simultaneous Iterative Reconstruction Technique) for higher quality
3. Add Fourier Shell Correlation (FSC) for resolution validation
4. Support conformation-specific reconstruction using NL-36 state labels
5. Provide GPU acceleration path using CuPy/ASTRA toolbox

## Acceptance Criteria
- [ ] `back_projection()` function with optional CTF correction
- [ ] `sirt()` iterative reconstruction with convergence tracking
- [ ] `estimate_resolution_fsc()` with gold-standard half-map approach
- [ ] `reconstruct_volume()` high-level pipeline integrating alignment metadata
- [ ] GPU-accelerated version `reconstruct_gpu()` using CuPy
- [ ] Visualization utilities for orthogonal slices and isosurfaces
- [ ] Tests with synthetic data validating reconstruction geometry
- [ ] Integration test with NL-34 aligned particles
- [ ] Resolution validation test showing expected FSC curves

## API Design

```python
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

@dataclass
class ReconstructionResult:
    volume: np.ndarray
    voxel_size: float
    resolution_fsc: Optional[float]
    fsc_curve: Optional[Tuple[np.ndarray, np.ndarray]]
    n_particles: int
    n_iterations: Optional[int]
    correlation_scores: Optional[List[float]]

@dataclass
class ReconstructionParams:
    box_size: int = 64
    voxel_size: float = 1.0
    n_iterations: int = 20
    regularization: float = 0.01
    ctf_corrected: bool = False
    mask_radius: Optional[float] = None

def back_projection(
    particle_stack,
    angles: np.ndarray,
    params: ReconstructionParams = None
) -> ReconstructionResult:
    """Weighted back-projection reconstruction."""
    ...

def sirt(
    particle_stack,
    angles: np.ndarray,
    params: ReconstructionParams = None
) -> ReconstructionResult:
    """Simultaneous Iterative Reconstruction Technique."""
    ...

def estimate_resolution_fsc(
    volume1: np.ndarray,
    volume2: np.ndarray,
    voxel_size: float,
    threshold: float = 0.143
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """Compute FSC and extract resolution at threshold."""
    ...

def reconstruct_volume(
    particle_stack,
    angles: np.ndarray,
    method: str = "sirt",
    params: ReconstructionParams = None,
    split_half: bool = True
) -> ReconstructionResult:
    """High-level reconstruction pipeline with optional gold-standard split."""
    ...

def reconstruct_gpu(
    particle_stack,
    angles: np.ndarray,
    params: ReconstructionParams = None
) -> ReconstructionResult:
    """GPU-accelerated reconstruction using CuPy/ASTRA."""
    ...
```

## Test Plan

### Unit Tests
1. **test_back_projection_geometry** — Verify reconstruction of known phantom
2. **test_sirt_convergence** — Check SIRT reduces error over iterations
3. **test_fsc_resolution** — Validate FSC curve and threshold detection
4. **test_reconstruction_noise_handling** — Robustness to noisy projections
5. **test_ctf_correction** — Optional CTF handling in back-projection

### Integration Tests
6. **test_reconstruct_aligned_particles** — Use NL-34 output as input
7. **test_state_specific_reconstruction** — Use NL-36 labels for conformation separation
8. **test_end_to_end_reconstruction** — Full pipeline from substacks to volume

### GPU Tests (skip if CuPy unavailable)
9. **test_reconstruct_gpu** — GPU reconstruction parity with CPU
10. **test_sirt_gpu** — GPU SIRT performance and accuracy

## Implementation Notes
- Use scipy.ndimage for interpolation and geometric transforms
- ASTRA toolbox integration for GPU acceleration (optional dependency)
- Gold-standard approach: split particles into two halves, reconstruct independently
- FSC threshold: 0.143 for cryo-EM standard, 0.5 for AFM applications
- Support for arbitrary projection angles (not limited to regular grids)
- Memory-efficient implementation for large volumes

## Success Metrics
- Synthetic phantom reconstruction achieves >0.9 correlation with ground truth
- FSC resolution estimate matches known features in synthetic data
- SIRT converges within specified iterations (correlation plateaus)
- GPU acceleration provides >5x speedup for typical datasets (64³ volume, 100+ particles)
