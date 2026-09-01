# NL-32 — Particle substack extraction

## Goal
Extract and align particle-centered substacks from drift-corrected AFM movies for downstream classification and averaging.

## Motivation
After drift correction (NL-30) and particle detection (NL-16/NL-22), individual particles need to be extracted as 3D substacks (x, y, time) centered on each detected particle. These substacks enable:
- Single-particle conformational analysis
- Classification by structural state (PCA + HDBSCAN, NL-33)
- In-class averaging for SNR improvement (NL-34)
- Dynamics tracking and transition detection (NL-36)

## Dependencies
- **NL-16**: Detection and statistics (particle coordinates, masks)
- **NL-30**: Drift estimation and correction (aligned frames)
- **NL-03**: Typed contracts (`Frame`, `Localizations`, `ParticleStack`)
- **NL-10**: Zarr schema for particle stack storage

## Acceptance Criteria

### 1. Substack extraction function
Implement `extract_particle_substacks()` that:
- Takes a movie (list of `Frame` or 3D array) and particle localizations
- Extracts fixed-size patches centered on each particle in each frame where detected
- Returns `ParticleStack` object with:
  - `data`: 4D array (particles, height, width, frames) or 3D if single-frame
  - `particle_ids`: Array mapping substacks to particle track IDs
  - `frame_indices`: Frame index for each substack slice
  - `centers`: Original (x, y) coordinates used for extraction
- Handles missing detections gracefully (particles not visible in all frames)
- Supports variable patch sizes via configuration

### 2. Drift-aware extraction
Implement `extract_drift_corrected_substacks()` that:
- Integrates drift correction into substack extraction
- Applies inverse drift shift during extraction (no resampling of full movie)
- Uses interpolation for sub-pixel center positions
- Returns aligned substacks ready for averaging

### 3. Mask-based extraction
Support optional mask application:
- Binary masks from NL-16 detection to extract only particle regions
- Soft masks (Gaussian-weighted) for smoother boundaries
- Option to return masked vs unmasked substacks

### 4. Batch processing
Implement efficient batch extraction:
- Process multiple particles in single operation
- Support progress callback for large datasets
- Memory-efficient streaming for very large movies

### 5. GPU acceleration
Provide `extract_particle_substacks_gpu()` using CuPy:
- Parallel extraction across particles
- GPU-accelerated interpolation for drift-corrected centers
- Maintain parity with CPU implementation within tolerance
- Fall back to CPU when CuPy unavailable

### 6. Integration
- Export from `nanolocz.core.substacks` module
- Type-annotated functions with proper error handling
- Compatible with `ParticleStack` type from NL-03/NL-10
- Integration with Zarr storage (NL-10) for saving extracted substacks

## Deliverables

### Specification
- `SPEC/NL-32-particle-substacks.md` (this file)

### Implementation
- `nanolocz/core/substacks.py` — CPU reference implementations
- `nanolocz/gpu/substacks.py` — GPU-accelerated versions

### Tests
- `tests/test_substacks_nl32.py` with:
  - TestParticleSubstackExtraction (8+ tests)
  - TestDriftCorrectedSubstacks (6+ tests)
  - TestMaskedExtraction (4+ tests)
  - TestBatchProcessing (4+ tests)
  - TestSubstacksGPU (8+ tests, skipped if CuPy unavailable)
  - TestSubstacksIntegration (6+ tests)

### Documentation
- `SESSIONS/YYYY-MM-DD-NL-32.md` session handoff
- Update `STATUS.md` with completion evidence

## API Design

```python
from nanolocz.core.substacks import (
    extract_particle_substacks,
    extract_drift_corrected_substacks,
    ParticleStack,
)
from nanolocz.core.types import Frame, Localizations, ParticleStack

# Basic extraction
substacks = extract_particle_substacks(
    movie: list[Frame] | np.ndarray,
    localizations: Localizations,
    patch_size: tuple[int, int] = (32, 32),
    mask: np.ndarray | None = None,
)
# Returns: ParticleStack with data, particle_ids, frame_indices, centers

# Drift-corrected extraction
substacks = extract_drift_corrected_substacks(
    movie: list[Frame] | np.ndarray,
    localizations: Localizations,
    drift: np.ndarray,  # From estimate_drift_xcorr() or estimate_drift_particles()
    patch_size: tuple[int, int] = (32, 32),
    interpolation_order: int = 1,  # Bilinear
)

# Save to Zarr
from nanolocz.io.store import NanoLoczStore
with NanoLoczStore.open('output.zarr', mode='w') as store:
    store.save_particle_stacks(substacks, name='particle_substacks')
```

## Tolerance Policy
- Substack pixel values: rtol=1e-5, atol=1e-8 (CPU float64)
- Interpolation accuracy: rtol=1e-4, atol=1e-6 (sub-pixel shifts)
- GPU parity: rtol=1e-3, atol=1e-5 (GPU float32 mode)

## Notes
- Substack extraction is memory-intensive; consider chunking for large datasets
- Drift-corrected extraction avoids full-movie resampling by applying shifts during patch extraction
- ParticleStack format must support variable-length time series per particle
- Integration with NL-33 (PCA/HDBSCAN) requires consistent ordering and metadata
