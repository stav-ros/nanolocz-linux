# Project status

Last updated: 2026-08-29
Current phase: Phase 2 — GPU foundation (COMPLETE)
Current card: NL-22
Status: in_progress — NL-21 batched line levelling is complete; beginning detection/statistics kernel work

## Progress

| Card | Area | State | Evidence |
|---|---|---|---|
| NL-01 | Fork audit and toolbox map | done | `ADR/0000-toolbox-map.md`; upstream revision `e41575c` audited |
| NL-02 | Golden parity harness | done | `SPEC/parity.md`; checksum loader, centralized tolerances, 7 CPU tests, CI workflow |
| NL-03 | Typed core contracts | done | `nanolocz/core/types.py`; 43 type contract tests green; protocols with @runtime_checkable; SESSIONS/2026-08-28-NL-03.md |
| NL-10 | Zarr schema and opener interface | **done** | `SPEC/NL-10-zarr-schema.md`; `nanolocz/io/store.py`; 29/29 I/O tests green; complete round-trip validation; SESSIONS/2026-08-28-NL-10.md |
| NL-11 | .gwy and .h5-jpk Openers | **done** | Gwyddion (.gwy) and JPK (.h5-jpk) readers implemented; unified opener interface; metadata extraction; 99/99 tests green; SESSIONS/2026-08-28-NL-11.md |
| NL-14 | Line, plane, and weighted multi-plane levelling | **done** | `nanolocz/core/leveling.py`; line/plane/weighted leveling; batch movie processing; 142/142 tests green; SESSIONS/2026-08-28-NL-14-15.md |
| NL-15 | Filters, masks, scar removal, and profiles | **done** | `nanolocz/core/filters.py`; Gaussian/median/uniform filters; gradient/Laplacian; masks; scar removal; morphological ops; 142/142 tests green; SESSIONS/2026-08-28-NL-14-15.md |
| NL-16 | Detection, statistics, and masks | done | `nanolocz/core/detection.py`; 9 focused tests; 165/165 full Python tests green; typed `DetectionResult`; masks, prominence, min-distance, and statistics |
| NL-17 | Deterministic single-particle tracking | done | `nanolocz/core/tracking.py`; `tests/test_tracking_nl17.py`; deterministic gap reconnection and input-order IDs; acceptance handoff recorded |
| NL-20 | NumPy/CuPy backend switch and precision policy | done | `nanolocz/gpu/backend.py`; `tests/test_backend_nl20.py`; Backend/BackendContext/PrecisionMode/TolerancePolicy; float64 CPU reference; GPU tolerance rules; SESSIONS/2026-08-29-NL-20.md |
| NL-12 | `.spm`, `.jpk`, and `.ibw` openers | done | `nanolocz/formats/{spm_reader,jpk_reader,ibw_reader}.py`; 4 focused tests; unified read-only opener routes |
| NL-13 | `.asd` opener including trace-only files | done | `nanolocz/formats/asd_reader.py`; 3 focused tests; image and trace-only handling; unified read-only opener route |
| NL-21 | Batched levelling kernel | done | `nanolocz/core/leveling.py::batch_line_leveling`; 4 focused tests; 223/223 runnable tests green; project self-check passes |
| NL-22–NL-24 | GPU kernels | in_progress | NL-22 is the active card; NL-20 backend and NL-21 batched levelling are complete |
| NL-30–NL-37 | LAFM+ science | not_started | blocked by core/GPU work |
| NL-40–NL-43 | Interface and shipping | not_started | blocked by core/GPU work |
| NL-50–NL-54 | Simulation bridge | not_started | **unblocked** — can start NL-50 |

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

## Next action

Implement NL-22: detection and statistics kernels using the validated NumPy/CuPy backend policy. NL-12, NL-13, NL-16, NL-17, NL-20, and NL-21 are complete. NL-16 provides a NumPy/float64 reference implementation:
- typed `DetectionResult` with legacy result access compatibility
- deterministic local maxima and minimum-distance suppression
- line-based prominence calculation and threshold filtering
- boolean candidate masks and input mask restriction
- area, volume, and eccentricity statistics
- focused synthetic parity tests; MATLAB/golden fixture data remains unavailable in this checkout

The verified CPU workflow is now:
AFM file → normalized Frame → preprocessing → detection → tracking → export

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

