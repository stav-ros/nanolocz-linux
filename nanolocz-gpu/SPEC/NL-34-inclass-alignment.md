# NL-34 — In-class alignment and averaging

## Goal
Implement rigid alignment of particles within each HDBSCAN cluster followed by computation of class averages to improve signal-to-noise ratio (SNR) and prepare for tip estimation.

## Motivation
After classifying particles into conformational groups (NL-33), particles within each cluster still exhibit translational misalignment due to:
- Detection centering errors (±1-2 pixels)
- Residual drift not corrected by global drift estimation (NL-30)
- Local stage variations during acquisition

Aligning particles within each class and computing averages will:
- Improve SNR by ~√N where N is cluster size
- Reveal finer structural details obscured by noise
- Enable accurate tip shape estimation (NL-35)
- Provide representative images for each conformational state
- Support validation of clustering quality

## Dependencies
- **NL-32**: Particle substack extraction (input data)
- **NL-33**: PCA to HDBSCAN grouping (cluster labels)
- **NL-03**: Typed contracts (`ParticleStack`, `Localizations`)
- **NL-10**: Zarr schema for storing aligned stacks and averages

## Acceptance Criteria

### 1. Rigid translational alignment
Implement `align_particles()` that:
- Takes `ParticleStack` data and cluster labels from NL-33
- For each cluster with ≥2 particles:
  - Selects reference particle (median or highest SNR)
  - Computes cross-correlation shifts for all particles vs reference
  - Applies sub-pixel shifts using Fourier shift theorem or spline interpolation
  - Returns aligned particle stack
- Handles clusters of varying sizes
- Preserves original box size (no cropping)
- Validates alignment quality via correlation scores

### 2. Reference selection strategies
Support multiple reference selection methods:
- **median**: Use median image of cluster (robust to outliers)
- **mean**: Use mean image of cluster
- **highest_snr**: Use particle with highest variance/SNR
- **manual**: Allow user-specified reference index
- Default: 'median' for robustness

### 3. Shift computation
Implement cross-correlation based shift estimation:
- FFT-based cross-correlation for efficiency
- Sub-pixel precision via parabolic fit or phase correlation
- Return shift values (dx, dy) for each particle
- Support optional mask to exclude bad regions
- Handle edge effects appropriately

### 4. Shift application
Implement sub-pixel image shifting:
- Fourier shift method (preferred for accuracy)
- Spline interpolation fallback (scipy.ndimage.shift)
- Preserve intensity conservation
- Handle boundary conditions (constant, reflect, wrap)
- Default: Fourier shift with constant boundary

### 5. Class average computation
Implement `compute_class_averages()` that:
- Takes aligned particle stacks and cluster labels
- Computes mean and standard deviation images per cluster
- Returns dictionary mapping cluster_id → {mean, std, count}
- Optionally computes weighted average using alignment confidence
- Saves representative particles per cluster

### 6. Iterative refinement (optional)
Provide `refine_alignment()` for multi-round alignment:
- Align → average → realign to updated reference
- Configurable number of iterations (default: 2-3)
- Convergence criterion based on shift magnitude
- Prevents overfitting via regularization

### 7. Alignment quality metrics
Compute and return quality metrics:
- Cross-correlation peak height (alignment confidence)
- Shift magnitude distribution per cluster
- Resolution estimate via FRC between half-set averages
- Outlier detection (particles with poor alignment)

### 8. GPU acceleration
Provide GPU-accelerated alignment:
- `align_particles_gpu()` using CuPy FFT
- Batch processing for large clusters
- Maintain parity with CPU within tolerance
- Fall back to CPU when CuPy unavailable

### 9. Integration
- Export from `nanolocz.core.alignment` module
- Type-annotated functions with proper error handling
- Compatible with `ClassificationResult` from NL-33
- Results storable in Zarr format (NL-10)
- Visualization utilities for alignment verification

## Deliverables

### Specification
- `SPEC/NL-34-inclass-alignment.md` (this file)

### Implementation
- `nanolocz/core/alignment.py` — CPU reference implementations
- `nanolocz/gpu/alignment.py` — GPU-accelerated versions

### Tests
- `tests/test_alignment_nl34.py` with:
  - TestTranslationalAlignment (8+ tests)
  - TestReferenceSelection (5+ tests)
  - TestShiftComputation (6+ tests)
  - TestShiftApplication (6+ tests)
  - TestClassAverages (5+ tests)
  - TestIterativeRefinement (4+ tests)
  - TestAlignmentQualityMetrics (5+ tests)
  - TestAlignmentGPU (6+ tests, skipped if CuPy unavailable)
  - TestAlignmentIntegration (6+ tests)

### Documentation
- `SESSIONS/YYYY-MM-DD-NL-34.md` session handoff
- Update `STATUS.md` with completion evidence

## API Design

```python
from nanolocz.core.alignment import (
    align_particles,
    compute_class_averages,
    refine_alignment,
    compute_shift_fft,
    apply_shift_fourier,
    AlignmentResult,
    ClassAverage,
)
from nanolocz.core.classification import ClassificationResult
from nanolocz.core.types import ParticleStack

# Basic alignment pipeline
aligned_result = align_particles(
    particle_stack: ParticleStack,
    cluster_labels: np.ndarray,  # From NL-33
    reference_method: str = 'median',  # or 'mean', 'highest_snr', 'manual'
    reference_index: int | None = None,  # For manual reference
    shift_method: str = 'fourier',  # or 'spline'
    boundary_mode: str = 'constant',
    mask: np.ndarray | None = None,  # Optional mask
)

# Access aligned data
aligned_stack = aligned_result.aligned_stack  # ParticleStack
shifts = aligned_result.shifts  # Shape: (n_particles, 2)
scores = aligned_result.correlation_scores  # Alignment confidence
n_aligned = aligned_result.n_aligned

# Compute class averages
averages = compute_class_averages(
    aligned_stack=aligned_stack,
    cluster_labels=cluster_labels,
    compute_std=True,
)

# Access averages
for cluster_id, avg in averages.items():
    mean_img = avg.mean  # 2D array
    std_img = avg.std    # 2D array
    count = avg.count    # Number of particles
    print(f"Cluster {cluster_id}: {count} particles, mean shape {mean_img.shape}")

# Iterative refinement
refined_result = refine_alignment(
    particle_stack=particle_stack,
    cluster_labels=cluster_labels,
    n_iterations: int = 3,
    convergence_threshold: float = 0.1,  # pixels
)

# Manual two-step process
shifts = compute_shift_fft(particle_stack.data, reference_image)
aligned_data = apply_shift_fourier(particle_stack.data, shifts)
```

## Data Classes

```python
from dataclasses import dataclass, field
import numpy as np
from typing import Dict

@dataclass
class AlignmentResult:
    """Results from particle alignment."""
    aligned_stack: ParticleStack  # Aligned particle stack
    shifts: np.ndarray  # Shape: (n_particles, 2), (dx, dy) in pixels
    correlation_scores: np.ndarray  # Shape: (n_particles,), peak correlation
    reference_images: Dict[int, np.ndarray]  # cluster_id → reference used
    n_aligned: int  # Number of successfully aligned particles
    n_clusters: int  # Number of clusters processed
    failed_indices: list[int] = field(default_factory=list)  # Failed alignments
    
    def get_cluster_shifts(self, cluster_id: int) -> np.ndarray:
        """Return shifts for particles in specified cluster."""
        ...
    
    def get_shift_statistics(self, cluster_id: int) -> dict:
        """Return mean, std, min, max shifts for cluster."""
        ...

@dataclass
class ClassAverage:
    """Class average statistics for a single cluster."""
    cluster_id: int
    mean: np.ndarray  # 2D mean image
    std: np.ndarray   # 2D standard deviation image
    count: int        # Number of particles
    aligned_indices: np.ndarray  # Indices of particles in this cluster
    resolution_estimate: float | None = None  # Optional FRC resolution
    
    def save_to_zarr(self, store, path: str):
        """Save class average to Zarr store."""
        ...
```

## Tolerance Policy
- Shift computation: rtol=1e-4, atol=1e-6 pixels (sub-pixel accuracy)
- Alignment reconstruction: rtol=1e-5, atol=1e-8 (Fourier method)
- Class average intensity: rtol=1e-6, atol=1e-10 (conservation)
- GPU parity: rtol=1e-3, atol=1e-5 (GPU float32 mode)
- Cross-correlation peak: rtol=1e-5, atol=1e-8

## Notes
- FFT-based shift computation is O(N log N) vs O(N²) for spatial correlation
- Fourier shift theorem: shift in real space = phase ramp in Fourier space
- Sub-pixel shifts via Fourier method avoid interpolation artifacts
- Reference selection impacts convergence speed but not final result (with iteration)
- Clusters with <2 particles are skipped (cannot align single particle)
- Poor alignment may indicate: wrong cluster assignment, damaged particle, or aggregation
- Consider implementing rotational alignment for symmetric particles (future extension)
- Mask support allows excluding scan-line artifacts or bad regions from alignment
- FRC (Fourier Ring Correlation) between half-set averages provides resolution estimate
- Alignment quality metrics help identify problematic clusters for manual inspection

## Test Data Requirements
- Synthetic particle stacks with known shifts applied
- Real AFM data with expected conformational states
- Edge cases: single-particle clusters, very large clusters (>100 particles)
- Particles with varying SNR levels
- Test data with ground-truth shifts for validation
- Large dataset for performance testing (>500 particles across multiple clusters)

## Performance Targets
- Alignment of 100 particles (32×32 box): <1 second (CPU)
- Alignment of 1000 particles (32×32 box): <5 seconds (CPU)
- GPU acceleration should provide 5-10× speedup for large datasets
- Memory usage: ≤2× input stack size during alignment
