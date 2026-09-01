# NL-31 — Directional deskar filter

## Goal
Implement a directional filter to remove scan-line artifacts (scars) and periodic noise patterns from AFM images while preserving particle features.

## Motivation
AFM images often contain directional artifacts from the scanning process:
- Scan lines (horizontal or vertical stripes)
- Periodic noise patterns
- Tip-induced streaks aligned with scan direction
- "Scars" from tip crashes or contamination

These artifacts interfere with:
- Particle detection (false positives along lines)
- Accurate height measurements
- Classification and averaging (NL-33, NL-34)
- Dynamics analysis (NL-36)

A directional deskar filter selectively removes these artifacts while preserving isotropic particle features.

## Dependencies
- **NL-15**: Filters, masks, scar removal (provides base filtering infrastructure)
- **NL-20**: NumPy/CuPy backend (for GPU acceleration option)

## Acceptance Criteria

### 1. Directional frequency filtering
Implement `directional_deskar()` that:
- Computes 2D FFT of input image
- Identifies and masks frequency components aligned with scan direction
- Applies notch or band-stop filtering for periodic artifacts
- Returns filtered image via inverse FFT
- Preserves low-frequency sample topography

### 2. Scan-line artifact removal
Support dedicated scan-line removal:
- Detect dominant scan direction (horizontal/vertical/custom angle)
- Remove line-by-line offsets using robust statistics
- Optional: adaptive thresholding for scar detection
- Preserve edges perpendicular to scan direction

### 3. Anisotropic diffusion (optional enhancement)
Implement edge-preserving smoothing:
- Diffusion tensor aligned with scan direction
- Strong smoothing along scan lines
- Minimal smoothing across scan lines (preserves particle edges)
- Iterative scheme with convergence criteria

### 4. Parameter configuration
Support configurable parameters:
- `scan_angle`: Direction of scan lines (degrees, default 0° = horizontal)
- `frequency_cutoff`: FFT frequency threshold for artifact removal
- `notch_width`: Width of frequency mask around artifact peaks
- `strength`: Filter intensity (0.0 to 1.0)
- `preserve_low_freq`: Boolean to keep low-frequency topography

### 5. GPU acceleration
Provide `directional_deskar_gpu()` using CuPy:
- FFT-based filtering on GPU
- Parallel processing for batch operations
- Maintain parity with CPU implementation within tolerance
- Fall back to CPU when CuPy unavailable

### 6. Integration
- Export from `nanolocz.core.filters` module (or new `nanolocz.core.deskar` module)
- Type-annotated functions with proper error handling
- Compatible with `Frame` type from NL-03
- Works with movie batches (list of Frame)

## Deliverables

### Specification
- `SPEC/NL-31-directional-deskar.md` (this file)

### Implementation
- `nanolocz/core/deskar.py` — CPU reference implementation
- `nanolocz/gpu/deskar.py` — GPU-accelerated version (optional, can be combined with NL-20 extension)

### Tests
- `tests/test_deskar_nl31.py` with:
  - TestDirectionalDeskarFFT (6+ tests): FFT-based filtering
  - TestScanLineRemoval (5+ tests): Line artifact removal
  - TestAnisotropicDiffusion (4+ tests): Edge-preserving smoothing
  - TestParameterSensitivity (4+ tests): Parameter effects
  - TestDeskarGPU (6+ tests, skipped if CuPy unavailable)
  - TestDeskarIntegration (4+ tests): End-to-end workflow

### Documentation
- `SESSIONS/YYYY-MM-DD-NL-31.md` session handoff
- Update `STATUS.md` with completion evidence

## API Design

```python
from nanolocz.core.deskar import (
    directional_deskar,
    remove_scan_lines,
    anisotropic_diffusion,
)
from nanolocz.core.types import Frame

# Basic directional deskar filtering
filtered_frame = directional_deskar(
    frame: Frame | np.ndarray,
    scan_angle: float = 0.0,  # Horizontal scan lines
    frequency_cutoff: float = 0.1,  # Normalized frequency (0-0.5)
    notch_width: float = 0.02,  # Width of frequency mask
    strength: float = 0.8,  # Filter strength
    preserve_low_freq: bool = True,
) -> Frame

# Dedicated scan-line removal
cleaned_frame = remove_scan_lines(
    frame: Frame | np.ndarray,
    direction: str = 'horizontal',  # 'horizontal', 'vertical', or angle
    method: str = 'median',  # 'median', 'mean', 'robust'
    threshold: float = 2.0,  # Sigma threshold for scar detection
) -> Frame

# Anisotropic diffusion for edge preservation
diffused_frame = anisotropic_diffusion(
    frame: Frame | np.ndarray,
    n_iterations: int = 10,
    kappa: float = 50.0,  # Conductance parameter
    gamma: float = 0.1,  # Step size
    scan_angle: float = 0.0,  # Preferred diffusion direction
) -> Frame

# Batch processing for movies
from nanolocz.core.deskar import process_movie_deskar

cleaned_movie = process_movie_deskar(
    movie: list[Frame],
    scan_angle: float = 0.0,
    frequency_cutoff: float = 0.1,
    progress_callback: callable | None = None,
) -> list[Frame]
```

## Tolerance Policy
- Filtered pixel values: rtol=1e-5, atol=1e-8 (CPU float64)
- FFT reconstruction error: rtol=1e-6, atol=1e-10 (lossless without masking)
- GPU parity: rtol=1e-3, atol=1e-5 (GPU float32 mode)
- Artifact suppression: >90% reduction in targeted frequency bands

## Algorithm Details

### FFT-based Directional Filtering
1. Compute 2D FFT of input image
2. Rotate frequency domain by -scan_angle
3. Create mask for frequencies aligned with scan direction (vertical lines in rotated freq domain)
4. Apply notch/band-stop filter at identified artifact frequencies
5. Rotate back by +scan_angle
6. Compute inverse FFT

### Scan-Line Removal
1. For each line (row/column depending on direction):
   - Compute robust statistic (median or mean)
   - Detect outliers using threshold
   - Replace outliers with interpolated values
2. Optional: high-pass filter along scan direction only

### Anisotropic Diffusion
1. Compute gradient magnitude and direction
2. Construct diffusion tensor with preferred orientation
3. Update image: I(t+1) = I(t) + γ * div(D * ∇I)
4. Repeat for n_iterations

## Notes
- Directional deskar is a preprocessing step before particle detection (NL-16/NL-22)
- Can be applied after leveling (NL-14/NL-21) but before detection
- FFT approach is efficient for periodic artifacts; spatial approach better for isolated scars
- GPU acceleration particularly beneficial for large images and batch processing
- Integration with NL-15 filters allows combined preprocessing pipelines

## Test Data Requirements
- Synthetic images with known scan-line artifacts
- BeadCloud simulations with added periodic noise
- Real AFM data with visible scan artifacts
- Ground truth for validation (artifact-free regions or simulations)
