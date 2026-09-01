# Project status

Last updated: 2026-08-29
Current phase: Phase 3 — LAFM+ (NL-33 COMPLETE)
Current card: NL-34
Status: in_progress — NL-33 PCA to HDBSCAN grouping complete with 31 tests passing; starting NL-34 in-class alignment and averaging

## Progress

| Card | Area | State | Evidence |
|---|---|---|---|
| NL-01 | Fork audit and toolbox map | done | `ADR/0000-toolbox-map.md`; upstream revision `e41575c` audited |
| NL-02 | Golden parity harness | done | `SPEC/parity.md`; checksum loader, centralized tolerances, 7 CPU tests, CI workflow |
| NL-03 | Typed core contracts | done | `nanolocz/core/types.py`; 43 type contract tests green; protocols with @runtime_checkable; SESSIONS/2026-08-28-NL-03.md |
| NL-10 | Zarr schema and opener interface | **done** | `SPEC/NL-10-zarr-schema.md`; `nanolocz/io/store.py`; 29/29 I/O tests green; complete round-trip validation; SESSIONS/2026-08-28-NL-10.md |
| NL-11 | .gwy and .h5-jpk Openers | **done** | Gwyddion (.gwy) and JPK (.h5-jpk) readers implemented; unified opener interface; metadata extraction; 99/99 tests green; SESSIONS/2026-08-28-NL-11.md |
| NL-12 | .spm, .jpk, and .ibw openers | **done** | `nanolocz/formats/spm_reader.py`, `jpk_reader.py`, `ibw_reader.py`; 4/4 tests green; ASCII SPM, HDF5 JPK, optional IBW support; SESSIONS/2026-08-29-NL-12-13.md |
| NL-13 | .asd opener including trace-only files | **done** | `nanolocz/formats/asd_reader.py`; 3/3 tests green; image and trace-only ASD support; SESSIONS/2026-08-29-NL-12-13.md |
| NL-14 | Line, plane, and weighted multi-plane levelling | **done** | `nanolocz/core/leveling.py`; line/plane/weighted leveling; batch movie processing; 142/142 tests green; SESSIONS/2026-08-28-NL-14-15.md |
| NL-15 | Filters, masks, scar removal, and profiles | **done** | `nanolocz/core/filters.py`; Gaussian/median/uniform filters; gradient/Laplacian; masks; scar removal; morphological ops; 142/142 tests green; SESSIONS/2026-08-28-NL-14-15.md |
| NL-16 | Detection, statistics, and masks | done | `nanolocz/core/detection.py`; 9 focused tests; 165/165 full Python tests green; typed `DetectionResult`; masks, prominence, min-distance, and statistics |
| NL-17 | Deterministic single-particle tracking | done | `nanolocz/core/tracking.py`; `tests/test_tracking_nl17.py`; deterministic gap reconnection and input-order IDs; acceptance evidence recorded |
| NL-20 | NumPy/CuPy backend switch and precision policy | done | `nanolocz/gpu/backend.py`; `tests/test_backend_nl20.py`; Backend/BackendContext/PrecisionMode/TolerancePolicy; float64 CPU reference; GPU tolerance rules; 35+ backend consistency tests; SESSIONS/2026-08-29-NL-20.md |
| NL-21 | Batched levelling kernel | done | `nanolocz/gpu/leveling.py`; `tests/test_leveling_nl21.py`; batch_line_level_gpu, batch_plane_level_gpu; 25+ tests; SESSIONS/2026-08-29-NL-21.md |
| NL-22 | Detection and statistics kernels | done | `nanolocz/gpu/detection.py`; `tests/test_detection_gpu_nl22.py`; local_maxima_gpu, prominence_gpu, min_distance_suppression_gpu, detect_particles_gpu, statistics_gpu; 19 tests; SESSIONS/2026-08-29-NL-22.md |
| NL-23 | LAFM splat kernel and FRC | done | `nanolocz/gpu/lafm.py`; `tests/test_lafm_gpu_nl23.py`; splat_gaussian_gpu, compute_frc_gpu, frc_resolution, batch_splat_gpu; 25 tests; SESSIONS/2026-08-29-NL-23.md |
| NL-24 | Simulation AFM kernels | done | `nanolocz/gpu/simafm.py`; `tests/test_simafm_gpu_nl24.py`; compute_height_field_gpu, convolve_tip_gpu, noise/artifact functions, simulate_afm_image_gpu; 26 tests; SESSIONS/2026-08-29-NL-24.md |
| NL-30 | Per-frame drift estimation | done | `nanolocz/core/drift.py`; `tests/test_drift_nl30.py`; estimate_drift_xcorr, estimate_drift_particles, correct_drift; 28 passed + 8 skipped (CuPy); SESSIONS/2026-08-29-NL-30.md |
| NL-31 | Directional deskar filter | **done** | `nanolocz/core/deskar.py`; `tests/test_deskar_nl31.py`; directional_deskar, remove_scan_lines, anisotropic_diffusion, process_movie_deskar; 23 passed + 6 skipped (CuPy); SPEC/NL-31-directional-deskar.md; SESSIONS/2026-08-29-NL-31.md |
| NL-32 | Particle substack extraction | **done** | `nanolocz/core/substacks.py`; `tests/test_substacks_nl32.py`; extract_particle_substacks, extract_drift_corrected_substacks, batch_extract_substacks, create_gaussian_mask, apply_binary_mask; 28 passed + 8 skipped (CuPy); SPEC/NL-32-particle-substacks.md; SESSIONS/2026-08-29-NL-32.md |
| NL-33 | PCA to HDBSCAN grouping | **done** | `nanolocz/core/classification.py`; `nanolocz/gpu/classification.py`; `tests/test_classification_nl33.py`; classify_particles, reduce_dimensions_pca, cluster_hdbscan, ClassificationResult, plot_scree, plot_clusters_2d; 31 passed + 1 skipped (CuPy); SPEC/NL-33-pca-hdbscan.md; SESSIONS/2026-08-29-NL-33.md |
| NL-12–NL-13 | Additional file openers | done | See NL-12 and NL-13 rows above |
| NL-34 | In-class alignment and averaging | in_progress | SPEC/NL-34-inclass-alignment.md pending; depends on NL-33 ✓ |
| NL-40–NL-43 | Interface and shipping | not_started | blocked by core/GPU work |
| NL-51–NL-54 | Simulation bridge extensions | not_started | NL-50 minimal PDB/simulation/fitting workflow is complete; future cards cover hard-collision parity, CUDA synthesis, masked NCC, and viewer validation |

## Phase 1 reconciliation evidence

- Canonical package: `nanolocz/`; obsolete `src/nanolocz/` removed.
- Single valid `pyproject.toml` with top-level package discovery and `[test]` extra.
- Parity fixtures and tolerances are exposed from `nanolocz.parity`.
- Validation baseline: `219 passed, 10 skipped` (CuPy unavailable); editable install succeeds; project self-check passes.

## Phase 2 completion summary

### NL-10 Deliverables
- Zarr schema specification in `SPEC/NL-10-zarr-schema.md`
- Complete `NanoLoczStore` implementation in `nanolocz/io/store.py`
- Full I/O support for movies, localizations, tracks, and particle stacks
- Round-trip validation with 29/29 tests passing
- Context manager support and proper resource cleanup
- Schema versioning and validation
- Compression with Blosc Zstandard (zstd) level 3
- Tuple-to-list conversion for Zarr JSON compatibility
- Proper error handling for missing data (KeyError for missing tracks/localizations)

### NL-10 Test Coverage
- Store creation and initialization
- Movie save/load with metadata preservation
- Localizations round-trip with optional fields
- Tracks save/load with empty track handling
- Particle stacks storage and retrieval
- Compressed storage validation
- Multiple save overwrites
- Complete round-trip integration test
- Opener interface tests (Zarr, HDF5, TIFF)
- Schema validation and version compatibility

### NL-14 Deliverables (Leveling)
- Line leveling: subtract median/mean per line
- Plane leveling: fit and subtract 2D polynomial surface
- Weighted multi-plane leveling: robust fitting with outlier rejection
- Batch movie processing: apply leveling to all frames
- Integration with Frame and Meta types from NL-03

### NL-15 Deliverables (Filters, Masks, Profiles)
- Filters: Gaussian, median, uniform filtering with configurable kernels
- Derivatives: gradient magnitude, Laplacian computation
- Masks: threshold, percentile, adaptive, custom masks
- Scar removal: line artifact detection and inpainting
- Morphological operations: erosion, dilation, opening, closing
- Profile extraction: line profiles, radial averages

## Current action

NL-34 is now in progress. Specification document SPEC/NL-34-inclass-alignment.md needs to be created. Implementation of in-class alignment and averaging for particle substacks is ready to begin.

**Current status**:
- Phase 1 (CPU Core): COMPLETE (NL-01 through NL-17)
- Phase 2 (GPU Foundation): COMPLETE (NL-20, NL-21, NL-22, NL-23, NL-24, NL-30)
- Phase 3 (LAFM+): IN PROGRESS with NL-34 (NL-31, NL-32, NL-33 complete)

**Active card**:
- NL-34: In-class alignment and averaging — SPEC pending; depends on NL-33 ✓, NL-32 ✓

**Next cards after NL-34**:
- NL-35: Tip estimation and regularized deconvolution (depends on NL-34, NL-24)
- NL-36: Dynamics traces, transitions, and dwell times (depends on NL-33, NL-17)
- NL-37: Napari replay hooks (depends on NL-34, NL-36)

The verified workflows now support:
```
AFM file (.gwy, .h5-jpk, .spm, .jpk, .ibw, .asd, .tiff) 
    → open_nanolocz() 
    → Frame + Meta 
    → preprocessing (leveling, filters) 
    → detection 
    → tracking 
    → export (.zarr)
```
AND
```
BeadCloud → height field → tip convolution → noise/artifacts 
    → simulated AFM image → LAFM reconstruction → FRC resolution
```
AND
```
PDB → centered molecule → conical-tip AFM simulation → coarse tip/translation fit
```

## GPU Kernel Trilogy Summary (NL-22, NL-23, NL-24)

### NL-22: Detection and Statistics Kernels
**Module**: `nanolocz/gpu/detection.py`
**Tests**: `tests/test_detection_gpu_nl22.py` (19 tests)
**Functions**:
- `local_maxima_gpu()` - Find local maxima using maximum filter
- `prominence_gpu()` - Calculate peak prominence for each maximum
- `min_distance_suppression_gpu()` - Suppress peaks closer than min_distance
- `detect_particles_gpu()` - Full particle detection pipeline
- `statistics_gpu()` - Compute area, volume, eccentricity for detected particles

### NL-23: LAFM Splatting and FRC
**Module**: `nanolocz/gpu/lafm.py`
**Tests**: `tests/test_lafm_gpu_nl23.py` (25 tests)
**Functions**:
- `splat_gaussian_gpu()` - Splat 2D Gaussians at localization positions
- `splat_localizations_gpu()` - Convenience wrapper for localization splatting
- `compute_frc_gpu()` - Compute Fourier Ring Correlation between two half-maps
- `frc_resolution()` - Extract resolution at FRC=0.5 or 0.143 threshold
- `batch_splat_gpu()` - Batch processing for multiple localization sets

### NL-24: AFM Simulation Kernels
**Module**: `nanolocz/gpu/simafm.py`
**Tests**: `tests/test_simafm_gpu_nl24.py` (26 tests)
**Functions**:
- `compute_height_field_gpu()` - Generate height field from bead cloud model
- `convolve_tip_gpu()` - Simulate tip-sample convolution with parabolic/elliptical tip
- `add_thermal_noise_gpu()` - Add thermal drift artifacts
- `add_shot_noise_gpu()` - Add Poisson shot noise
- `add_scan_artifacts_gpu()` - Add scan line artifacts
- `simulate_afm_image_gpu()` - Complete AFM simulation pipeline

All kernels:
- Use BackendContext for CPU/GPU abstraction
- Support float32/float64 precision modes
- Include comprehensive tolerance policies
- Have CPU fallback when CuPy unavailable
- Maintain parity with CPU reference implementations

The minimal structure workflow is now:
PDB → centered molecule → conical-tip AFM simulation → coarse tip/translation fit

## Self-check contract

A task may move to `done` only when:

- its acceptance criteria in `SPEC/tasks.md` are satisfied;
- the targeted test command is green;
- `python tools/project_check.py` is green;
- the evidence is recorded in this file; and
- a session handoff exists under `SESSIONS/`.

Allowed states: `not_started`, `in_progress`, `blocked`, `done`.

### NL-20 Deliverables (NumPy/CuPy Backend Switch and Precision Policy)

#### Core Implementation (`nanolocz/gpu/backend.py`)
- **Backend enum**: CPU, CUDA, AUTO selection with auto-detection
- **PrecisionMode enum**: REFERENCE (float64), MIXED (float32 GPU/float64 CPU), HIGH (float64 everywhere)
- **BackendConfig**: Configuration dataclass with backend, precision, device_id, memory_limit, allow_downcast
- **BackendContext**: Execution context managing array creation, transfer, and computation
  - `xp` property for numpy/cupy module access
  - Array allocation methods: `zeros`, `ones`, `empty`, `allocate`, `array`
  - Transfer methods: `to_cpu`, `to_backend`
  - Utility methods: `copy`, `astype`, `get_stream`
- **TolerancePolicy**: Dataclass for numerical comparison tolerances
  - `CPU_REFERENCE_TOLERANCE`: rtol=1e-10, atol=1e-12 (strictest, for golden fixtures)
  - `CPU_STANDARD_TOLERANCE`: rtol=1e-5, atol=1e-8 (standard CPU float64)
  - `GPU_FLOAT32_TOLERANCE`: rtol=1e-3, atol=1e-5 (GPU single precision)
  - `GPU_FLOAT64_TOLERANCE`: rtol=1e-7, atol=1e-10 (GPU double precision)
  - `CROSS_BACKEND_TOLERANCE`: rtol=1e-4, atol=1e-6 (CPU vs GPU comparison)
- **assert_close**: Unified numerical comparison function supporting numpy/cupy arrays
- **validate_backend_result**: Result validation helper (shape, dtype, finite checks)
- **Convenience functions**: `get_backend_context`, `create_reference_context`, `create_gpu_context`

#### Test Coverage (`tests/test_backend_nl20.py`)
- TestBackendConfig: 7 tests for configuration and resolution
- TestTolerancePolicy: 10 tests for tolerance definitions and selection
- TestBackendContext: 12 tests for context creation and array operations
- TestAssertClose: 7 tests for numerical comparison
- TestValidateBackendResult: 5 tests for result validation
- TestBackendIntegration: 4 tests for integration scenarios
- TestBackendEdgeCases: 5 tests for edge cases
- Total: 50+ test cases (some skipped when CuPy unavailable)

#### Float64 CPU Reference Behavior
- `create_reference_context()` returns CPU backend with REFERENCE precision
- All reference computations use float64 dtype
- Strictest tolerance (1e-10/1e-12) for golden fixture generation
- Suitable for high-accuracy scientific validation

#### GPU Precision/Tolerance Rules
- MIXED mode: float32 on GPU, float64 on CPU (default for performance)
- HIGH mode: float64 everywhere (for accuracy-critical paths)
- Tolerance automatically selected based on backend + precision combination
- Cross-backend tolerance for CPU/GPU result comparison

#### Integration with Existing Code
- Updated `nanolocz/gpu/__init__.py` to export new backend API
- Legacy `utils.py` functions preserved for backward compatibility
- Protocols defined for type checking (BackendArray, GPUArray)

