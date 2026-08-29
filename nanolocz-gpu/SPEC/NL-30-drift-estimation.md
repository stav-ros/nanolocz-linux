# NL-30 — Per-frame drift estimation

## Goal
Estimate and correct sample drift in AFM movie sequences using cross-correlation and particle-based methods.

## Motivation
AFM images often exhibit slow drift during acquisition due to thermal expansion, piezo creep, or mechanical instability. Accurate drift estimation is essential for:
- Aligning frames before averaging
- Building consistent particle trajectories
- Enabling super-resolution reconstruction (LAFM+)

## Dependencies
- **NL-22**: Detection and statistics kernels (for particle-based drift)
- **NL-15**: Filters (for image preprocessing)
- **NL-03**: Typed contracts (`Frame`, `Meta`)

## Acceptance Criteria

### 1. Cross-correlation drift estimation
Implement `estimate_drift_xcorr()` that:
- Takes a movie (list of `Frame` objects or 3D array)
- Uses phase correlation (FFT-based) to estimate frame-to-frame shifts
- Returns cumulative drift trajectory as `(N, 2)` array with columns `[dx, dy]`
- Supports optional reference frame (default: first frame)
- Handles sub-pixel precision via upsampling

### 2. Particle-based drift estimation
Implement `estimate_drift_particles()` that:
- Detects particles in each frame using `detect_particles()` from NL-16/NL-22
- Matches particles between consecutive frames
- Computes median displacement as drift estimate
- Returns cumulative drift trajectory
- Handles missing particles gracefully

### 3. Drift correction
Implement `correct_drift()` that:
- Takes a movie and drift trajectory
- Applies shift correction to each frame
- Returns drift-corrected movie as list of `Frame` objects
- Preserves metadata and timestamps
- Uses interpolation for sub-pixel shifts

### 4. GPU acceleration
- Provide `estimate_drift_xcorr_gpu()` using CuPy FFT
- Maintain parity with CPU implementation within tolerance
- Fall back to CPU when CuPy unavailable

### 5. Integration
- Export from `nanolocz.core.drift` module
- Type-annotated functions with proper error handling
- Compatible with `Frame` and `Movie` types from NL-03/NL-10

## Deliverables

### Specification
- `SPEC/NL-30-drift-estimation.md` (this file)

### Implementation
- `nanolocz/core/drift.py` — CPU reference implementations
- `nanolocz/gpu/drift.py` — GPU-accelerated versions

### Tests
- `tests/test_drift_nl30.py` with:
  - TestDriftXCorr (8+ tests)
  - TestDriftParticles (6+ tests)
  - TestDriftCorrection (6+ tests)
  - TestDriftGPU (8+ tests, skipped if CuPy unavailable)
  - TestDriftIntegration (4+ tests)

### Documentation
- `SESSIONS/YYYY-MM-DD-NL-30.md` session handoff
- Update `STATUS.md` with completion evidence

## API Design

```python
from nanolocz.core.drift import (
    estimate_drift_xcorr,
    estimate_drift_particles,
    correct_drift,
    DriftResult,
)
from nanolocz.core.types import Frame

# Cross-correlation method
drift_result = estimate_drift_xcorr(
    movie: list[Frame] | np.ndarray,
    reference: int | np.ndarray = 0,
    upsample_factor: int = 10,
)
# Returns: DriftResult with shifts, cumulative_drift, per_frame_correlation

# Particle-based method
drift_result = estimate_drift_particles(
    movie: list[Frame] | np.ndarray,
    detection_params: dict | None = None,
    match_radius: float = 10.0,
)

# Apply correction
corrected_movie = correct_drift(
    movie: list[Frame] | np.ndarray,
    drift: np.ndarray,
    mode: str = "constant",  # 'constant', 'reflect', 'wrap'
)
```

## Tolerance Policy
- Cross-correlation shifts: rtol=1e-4, atol=1e-2 (sub-pixel accuracy)
- Particle-based shifts: rtol=1e-3, atol=0.5 pixels
- Drift correction residual: rtol=1e-3, atol=0.1 pixels

## Notes
- Phase correlation uses FFT-based method from `scipy.signal.correlate` or `cupyx.scipy.signal`
- Particle matching uses greedy nearest-neighbor within match_radius
- Cumulative drift is computed by integrating frame-to-frame shifts
- Edge handling: use constant padding (mean value) for shift corrections
