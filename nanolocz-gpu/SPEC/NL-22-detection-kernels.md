# NL-22: Detection and Statistics Kernels

## Objective
Port detection and statistics computations to GPU using CuPy, maintaining parity with CPU reference implementation.

## Dependencies
- NL-20: NumPy/CuPy backend switch and precision policy ✓
- NL-16: Detection, statistics, and masks (CPU reference) ✓

## Acceptance Criteria
1. GPU kernel for local maxima detection (`fast_peaks2d_gpu`)
2. GPU kernel for prominence calculation
3. GPU kernel for minimum-distance suppression
4. GPU kernel for detection statistics (area, volume, eccentricity)
5. Parity tests showing GPU results match CPU within tolerance policy
6. Batch processing support for multiple frames

## Implementation Plan

### 1. GPU Detection Kernel (`nanolocz/gpu/detection.py`)
- `local_maxima_gpu`: Find local maxima using maximum_filter on GPU
- `prominence_gpu`: Calculate prominence along line to nearest higher peak
- `min_distance_suppression_gpu`: Greedy selection with minimum distance constraint
- `detect_particles_gpu`: Main entry point matching CPU API

### 2. GPU Statistics Kernel (`nanolocz/gpu/statistics.py`)
- `region_statistics_gpu`: Compute area, volume, eccentricity per detection
- Batch processing for efficiency

### 3. Integration
- Use `BackendContext` from NL-20 for array management
- Apply tolerance policies for cross-backend validation
- Maintain API compatibility with CPU implementation

## Files to Create
- `nanolocz/gpu/detection.py` - GPU detection kernels
- `nanolocz/gpu/statistics.py` - GPU statistics kernels  
- `tests/test_detection_gpu_nl22.py` - GPU detection parity tests

## Tolerance Policy
- GPU float32: rtol=1e-3, atol=1e-5
- GPU float64: rtol=1e-7, atol=1e-10
- Cross-backend: rtol=1e-4, atol=1e-6

## Notes
- CuPy may not be available in test environment; tests should skip gracefully
- CPU reference path remains authoritative for golden fixture generation
- GPU path is for acceleration; correctness validated against CPU
