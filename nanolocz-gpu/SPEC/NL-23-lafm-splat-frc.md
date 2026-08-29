# NL-23: LAFM Splatting and FRC Kernel

## Objective
Implement GPU-accelerated LAFM (Localization AFM) splatting kernel for super-resolution reconstruction and Fourier Ring Correlation (FRC) for resolution estimation.

## Dependencies
- NL-20: NumPy/CuPy backend switch and precision policy ✓
- NL-16: Detection, statistics, and masks (CPU reference) ✓

## Acceptance Criteria
1. GPU kernel for Gaussian splatting from localizations (`splat_localizations_gpu`)
2. Support for variable localization uncertainties (sigma values)
3. Batch processing for multiple frames/timepoints
4. FRC computation between two half-map reconstructions (`compute_frc`)
5. Resolution estimation from FRC curve (1/7 threshold or 1/2 bit threshold)
6. Parity tests showing GPU results match CPU within tolerance policy

## Implementation Plan

### 1. LAFM Splatting Kernel (`nanolocz/gpu/lafm.py`)
- `splat_gaussian_gpu`: Render Gaussian peaks at localization coordinates
- `splat_localizations_gpu`: Main entry point for splatting localizations
- Support for per-localization sigma (uncertainty) or global sigma
- Optional intensity weighting from localization intensities

### 2. FRC Computation (`nanolocz/gpu/frc.py`)
- `compute_frc_gpu`: Compute Fourier Ring Correlation between two maps
- `frc_resolution`: Estimate resolution from FRC curve using standard thresholds
- Support for masking and background subtraction

### 3. Integration
- Use `BackendContext` from NL-20 for array management
- Apply tolerance policies for cross-backend validation
- Maintain API compatibility with future CPU implementation

## Files to Create
- `nanolocz/gpu/lafm.py` - LAFM splatting kernels
- `nanolocz/gpu/frc.py` - FRC computation kernels
- `tests/test_lafm_gpu_nl23.py` - LAFM/FRC GPU parity tests

## Tolerance Policy
- GPU float32: rtol=1e-3, atol=1e-5
- GPU float64: rtol=1e-7, atol=1e-10
- Cross-backend: rtol=1e-4, atol=1e-6

## Notes
- CuPy may not be available in test environment; tests should skip gracefully
- CPU reference path remains authoritative for golden fixture generation
- GPU path is for acceleration; correctness validated against CPU
- FRC thresholds: 1/7 (standard), 1/2-bit (information theoretic)
