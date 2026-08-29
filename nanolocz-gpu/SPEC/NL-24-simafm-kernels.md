# NL-24: AFM Simulation Kernels

## Objective
Implement GPU-accelerated simulation AFM kernels for tip-sample interaction modeling and synthetic AFM image generation.

## Dependencies
- NL-20: NumPy/CuPy backend switch and precision policy ✓
- NL-50: PDB ingest to BeadCloud (planned)
- NL-51: CPU hard-collision height field (planned)

## Acceptance Criteria
1. GPU kernel for height field computation from bead cloud model
2. GPU kernel for tip convolution with sample surface
3. Support for various tip geometries (sphere, cone, pyramid)
4. Noise injection models (thermal, shot noise, detector noise)
5. Scan line simulation with realistic artifacts
6. Parity tests showing GPU results match CPU within tolerance policy

## Implementation Plan

### 1. Height Field Kernel (`nanolocz/gpu/simafm.py`)
- `compute_height_field_gpu`: Generate height map from atomic coordinates
- `convolve_tip_gpu`: Apply tip geometry convolution
- Support for multiple tip shapes: sphere, cone, paraboloid
- Optional tilt and scan angle support

### 2. Noise Models
- `add_thermal_noise_gpu`: Add thermal drift and vibration
- `add_shot_noise_gpu`: Add Poisson-distributed detector noise
- `add_scan_artifacts_gpu`: Simulate scan line artifacts

### 3. Integration
- Use `BackendContext` from NL-20 for array management
- Apply tolerance policies for cross-backend validation
- Interface with future BeadCloud type from NL-50

## Files to Create
- `nanolocz/gpu/simafm.py` - Simulation AFM kernels
- `tests/test_simafm_gpu_nl24.py` - Simulation AFM GPU parity tests

## Tolerance Policy
- GPU float32: rtol=1e-3, atol=1e-5
- GPU float64: rtol=1e-7, atol=1e-10
- Cross-backend: rtol=1e-4, atol=1e-6

## Notes
- CuPy may not be available in test environment; tests should skip gracefully
- Initial implementation focuses on geometric simulation (hard collision)
- Future extension: soft interaction forces (van der Waals, electrostatic)
- Compatible with BioAFMviewer validation framework
