# NanoLocz Python Port - Development Plan

## Executive Summary

**NanoLocz** is a free, open-source Atomic Force Microscopy (AFM) image analysis platform originally developed in MATLAB. This document outlines the strategy for creating a Python-based fork that:
1. **Eliminates MATLAB dependency** - Pure Python implementation
2. **Adds GPU acceleration** - CUDA/OpenCL support for compute-intensive operations
3. **Maintains cross-platform compatibility** - Linux, macOS, Windows
4. **Preserves all core functionality** - File I/O, image processing, particle detection, tracking, LAFM

**License**: GPL v3.0 (must be maintained in the fork)

## Current minimal structure-to-AFM workflow

The fork includes a deliberately small BioAFM-inspired workflow. It currently
supports PDB atom import, a coarse conical-tip AFM height simulation, a simple
tip-radius estimate from an AFM image, and rough fitting over tip candidates
and translations. It is intended for visual exploration and a starting point
for fitting, not calibrated force-interaction physics or full BioAFMviewer
compatibility.

```python
from nanolocz.simafm import (
    TipParameters,
    estimate_tip_from_afm,
    fit_structure_to_afm,
    load_pdb,
    simulate_afm,
)

structure = load_pdb("protein.pdb")
simulated = simulate_afm(
    structure,
    shape=(128, 128),
    pixel_size_nm=0.1,
    tip=TipParameters(radius_nm=2.0, cone_angle_deg=20.0),
)
tip = estimate_tip_from_afm(experimental_image, pixel_size_nm=0.1)
fit = fit_structure_to_afm(
    structure,
    experimental_image,
    pixel_size_nm=0.1,
    tip_candidates=[tip],
)
```

The implementation lives in `nanolocz/simafm/`; its focused tests are in
`tests/test_simafm_simple.py`. The roadmap cards NL-51 through NL-54 cover
future simulation parity, CUDA acceleration, and optional viewer integration.

---

## Command-Line Interface (CLI)

NanoLocz provides a comprehensive command-line interface for headless batch processing and automation.

### Installation

```bash
pip install -e .
```

### Basic Usage

```bash
# Show all available commands
nanolocz --help

# Preprocess an AFM movie
nanolocz preprocess -i movie.gwy -o processed.zarr --leveling plane --filter gaussian

# Detect particles
nanolocz detect -i processed.zarr -o detected.zarr --threshold 3.5 --min-distance 5

# Track particles across frames
nanolocz track -i detected.zarr -o tracked.zarr --max-displacement 10

# LAFM reconstruction with FRC resolution
nanolocz lafm -i tracked.zarr -o lafm.zarr --pixel-size 0.5

# Batch process multiple files
nanolocz batch -i ./data/*.gwy -o ./results/ --config examples/cli_config.json --jobs 4
```

### Available Commands

| Command | Description |
|---------|-------------|
| `preprocess` | Leveling, filtering, scar removal on AFM movies |
| `detect` | Particle detection with configurable parameters |
| `track` | Single-particle tracking with gap closing |
| `lafm` | LAFM reconstruction and FRC calculation |
| `batch` | Process multiple files with a job queue |

### Configuration Files

Save pipeline parameters as JSON for reproducible workflows:

```bash
nanolocz batch -i ./data/ -o ./results/ --config my_pipeline.json
```

Example configuration file (`examples/cli_config.json`):
```json
{
  "preprocess": {
    "leveling": "plane",
    "filter": "gaussian",
    "filter_sigma": 1.0
  },
  "detect": {
    "threshold": 3.5,
    "min_distance": 5
  },
  "track": {
    "max_displacement": 10,
    "memory": 2
  },
  "lafm": {
    "pixel_size": 0.5,
    "sigma": 1.0,
    "frc_threshold": 0.5
  }
}
```

### Common Options

- `--input` / `-i`: Input file(s) or directory
- `--output` / `-o`: Output path (.zarr by default)
- `--config` / `-c`: JSON config file
- `--verbose` / `-v`: Verbose output
- `--gpu` / `--no-gpu`: Force GPU/CPU backend
- `--precision`: float32 or float64

Use `nanolocz <command> --help` for detailed options for each command.

---

## Current Architecture Analysis

### Core Components (MATLAB)

#### 1. **File I/O Layer** (`ReadAFMFile.m`, `open_*.m` files)
- Supports formats: `.spm`, `.asd`, `.jpk`, `.h5-jpk`, `.ibw`, `.ARIS`, `.tiff`, `.nhf`, `.gwy`
- Custom binary parsers for proprietary AFM formats
- HDF5 read/write capabilities

#### 2. **Image Processing Core** (`NanoLocz-lib/`)
Key functions requiring GPU acceleration:
- `Detector.m` - Particle detection (cross-correlation, peak picking)
- `localize.m` - Sub-pixel localization (Gaussian/sphere fitting)
- `align_trans.m` / `align_rot.m` - Image alignment (FFT cross-correlation)
- `align_iterate.m` - Iterative alignment
- `filter_movie.m` - Temporal filtering
- `level.m` / `level_auto.m` / `level_weighted.m` - Image leveling
- `Fast_peaks2D.m` - Peak detection
- `track_particles.m` - Particle tracking with gap closing

#### 3. **GUI Layer** (`.mlapp` files)
- MATLAB App Designer applications
- Interactive visualization and controls
- **Strategy**: Replace with PyQt6/PySide6 or web-based interface (Dash/Streamlit)

#### 4. **Specialized Features**
- **LAFM** (Localization AFM): `LAFM_renderer.m`, `LAFM_plotter.m`
- **Simulation AFM**: `Mat_SimAFM.m` series
- **Area Analysis**: `AnalyzeAreas.m`

---

## Technology Stack Recommendations

### Core Python Libraries

```python
# Numerical Computing
numpy >= 2.0          # Array operations
scipy >= 1.14         # Signal processing, optimization, FFT

# GPU Acceleration
cupy >= 13.0          # CUDA-compatible NumPy alternative
# OR
pytorch >= 2.0        # If deep learning features added later
# OR  
numba >= 0.60         # JIT compilation for CPU/GPU

# Image Processing
scikit-image >= 0.24  # Image processing algorithms
opencv-python >= 4.10 # Computer vision (normxcorr2, etc.)
pillow >= 11.0        # Image I/O

# File I/O
h5py >= 3.12          # HDF5 support
tifffile >= 2024.8    # TIFF handling
python-gdcm           # Medical/scientific imaging formats

# Optimization & Fitting
lmfit >= 1.3          # Non-linear least squares (Gaussian fitting)
scipy.optimize        # Built-in optimization

# Parallel Processing
joblib >= 1.4         # Simple parallelization
multiprocessing       # Built-in multi-core support

# GUI (Choose one)
pyqt6 >= 6.7          # Desktop application
# OR
streamlit >= 1.40     # Web-based interface
# OR
dash >= 2.18          # Web-based with Plotly

# Visualization
matplotlib >= 3.9     # Plotting
plotly >= 5.24        # Interactive 3D visualization
pyvista >= 0.44       # 3D rendering (VTK-based)

# Data Handling
pandas >= 2.2         # Tabular data export
xarray >= 2024.10     # Multi-dimensional arrays
```

### GPU Strategy

**Primary Approach: CuPy**
- Drop-in replacement for NumPy with CUDA support
- Supports most SciPy sparse operations
- GPU-accelerated FFT, convolution, correlation
- Easy migration path from MATLAB

**Alternative: PyTorch**
- Better if adding ML/AI features later
- More flexible GPU programming
- Steeper learning curve for scientific computing

**Fallback: Numba**
- CPU JIT compilation when GPU unavailable
- Can compile to CUDA with `@cuda.jit`
- Good for custom kernels

---

## Migration Priority & Phases

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Basic file I/O and image display without MATLAB

#### Tasks:
1. ✅ Set up Python project structure
2. ✅ Implement file format readers (priority formats first)
   - TIFF, HDF5 (easy, well-documented)
   - JPK, NanoScope (.spm), IBW (most common AFM formats)
3. ✅ Basic image visualization (matplotlib/PyQt)
4. ✅ Create test suite with Example Data from original repo

#### Deliverables:
- Working file loader for 5+ formats
- Basic image viewer
- Unit tests for I/O operations

### Phase 2: Core Processing (Weeks 5-10)
**Goal**: Implement image processing pipeline with optional GPU

#### Tasks:
1. ✅ Image leveling algorithms
   - `level.m` → polynomial surface fitting
   - `level_auto.m` → automatic plane detection
   - `level_weighted.m` → weighted least squares

2. ✅ Filtering operations
   - Gaussian filtering (`imgaussfilt` → `scipy.ndimage.gaussian_filter`)
   - Temporal filtering (`filter_movie.m`)

3. ✅ FFT-based operations
   - Cross-correlation (`normxcorr2` → `skimage.registration.phase_cross_correlation`)
   - DFT registration (port `dftregistration` directly)

4. ✅ GPU acceleration layer
   - Create abstraction: `get_array_module(use_gpu=False)`
   - Wrap critical functions with CuPy equivalents

#### Deliverables:
- Complete image preprocessing pipeline
- GPU/CPU abstraction layer
- Benchmark suite comparing to MATLAB

### Phase 3: Particle Detection & Tracking (Weeks 11-16)
**Goal**: Full particle analysis workflow

#### Tasks:
1. ✅ Peak detection (`Fast_peaks2D.m`)
   - Morphological operations (`scipy.ndimage.maximum_filter`)
   - Prominence calculation

2. ✅ Particle detection (`Detector.m`)
   - Direct peak picking mode
   - Cross-correlation mode with rotation search

3. ✅ Sub-pixel localization (`localize.m`)
   - Interpolation methods (bicubic, bilinear, Lanczos)
   - 2D Gaussian fitting (`lmfit` or `scipy.optimize`)
   - Sphere fitting (port `sumith_fit`)

4. ✅ Particle tracking (`track_particles.m`)
   - Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
   - Gap-closing logic

#### Deliverables:
- End-to-end particle detection workflow
- Localization precision matching MATLAB
- Tracking accuracy validation

### Phase 4: Advanced Features (Weeks 17-22)
**Goal**: LAFM, Simulation AFM, Area Analysis

#### Tasks:
1. ✅ LAFM rendering pipeline
   - Super-resolution reconstruction
   - FRC resolution analysis (`measureFRC.m`)
   - Alignment and symmetrization tools

2. ◐ Minimal Simulation AFM
   - PDB import and coarse conical-tip height rendering
   - Estimated tip radius and rough tip/translation fitting
   - More physical hard-collision modeling and movie generation remain on the roadmap

3. ✅ Area analysis tools
   - ROI selection and measurement
   - Statistical analysis

#### Deliverables:
- LAFM workflow complete
- Simulation capabilities
- Analysis tools

### Phase 5: GUI Development (Weeks 23-30)
**Goal**: User-friendly interface

#### Tasks:
1. ✅ Design modern PyQt6 interface
   - Drag-and-drop file loading
   - Interactive image display with histogram controls
   - Tool panels for each processing step
   - 3D visualization widget

2. ✅ Implement workflow manager
   - Processing pipeline builder
   - Batch processing support
   - Progress tracking and logging

3. ✅ Export functionality
   - TIFF, PNG, AVI, GIF export
   - CSV, Excel, HDF5 data export
   - Publication-quality figures

#### Deliverables:
- Full-featured desktop application
- User documentation
- Tutorial videos

### Phase 6: Optimization & Polish (Weeks 31-36)
**Goal**: Performance tuning and production readiness

#### Tasks:
1. ✅ Profile and optimize hot paths
2. ✅ Add comprehensive GPU benchmarks
3. ✅ Create installation packages
   - pip package
   - Conda forge recipe
   - Docker container
   - Standalone executable (PyInstaller)

4. ✅ Documentation
   - API documentation (Sphinx)
   - User guide (Markdown → website)
   - Example notebooks

#### Deliverables:
- Production-ready software
- Complete documentation
- CI/CD pipeline

---

## Critical Algorithm Mappings

### MATLAB → Python Function Reference

| MATLAB Function | Python Equivalent | GPU-Accelerated |
|----------------|-------------------|-----------------|
| `imresize` | `skimage.transform.resize` / `cv2.resize` | `cupyx.scipy.ndimage.zoom` |
| `imgaussfilt` | `scipy.ndimage.gaussian_filter` | `cupyx.scipy.ndimage.gaussian_filter` |
| `normxcorr2` | `skimage.registration.phase_cross_correlation` | Custom CuPy kernel |
| `fft2`, `ifft2` | `numpy.fft.fft2` | `cupy.fft.fft2` |
| `imrotate` | `scipy.ndimage.rotate` | `cupyx.scipy.ndimage.rotate` |
| `imtranslate` | `skimage.transform.warp` | Custom CuPy kernel |
| `ordfilt2` | `scipy.ndimage.maximum_filter` | Custom CuPy kernel |
| `lsqcurvefit` | `scipy.optimize.curve_fit` / `lmfit` | `cupy` + custom loss |
| `improfile` | `skimage.measure.profile_line` | Custom CuPy kernel |
| `meshgrid` | `numpy.meshgrid` | `cupy.meshgrid` |
| `parfor` | `joblib.Parallel` / `multiprocessing` | Native CUDA parallelism |

---

## GPU Acceleration Strategy

### Functions Priority for GPU Porting

**Tier 1 (Highest Impact):**
1. **Cross-correlation** (`Detector.m`, `align_trans.m`)
   - `normxcorr2` on large images
   - FFT-based correlation benefits greatly from GPU

2. **FFT operations** (alignment, registration)
   - `dftregistration` already uses FFT
   - CuPy FFT is 10-50x faster for large arrays

3. **Image resizing** (`localize.m`, `Detector.m`)
   - `imresize` called repeatedly in loops
   - GPU texture interpolation very efficient

**Tier 2 (Medium Impact):**
4. **Gaussian filtering** (preprocessing)
5. **Peak detection** (morphological operations)
6. **Iterative alignment loops**

**Tier 3 (Lower Priority):**
7. File I/O (cannot GPU accelerate)
8. Simple arithmetic operations
9. GUI rendering

### Implementation Pattern

```python
# gpu_utils.py
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def get_array_module(use_gpu=False):
    """Get numpy or cupy module based on availability and preference."""
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np

def to_gpu(array, use_gpu=False):
    """Transfer array to GPU if requested and available."""
    if use_gpu and CUPY_AVAILABLE:
        return cp.asarray(array)
    return array

def from_gpu(array):
    """Transfer array from GPU to CPU if needed."""
    if hasattr(array, 'get'):  # CuPy array
        return array.get()
    return array

# Example usage in Detector
def detect_particles(img, use_gpu=False):
    xp = get_array_module(use_gpu)
    img_gpu = to_gpu(img, use_gpu)
    
    # Processing happens on GPU
    result_gpu = some_operation(img_gpu)
    
    # Return CPU array for compatibility
    return from_gpu(result_gpu)
```

---

## File Format Support Priority

### Phase 1 (Essential):
- ✅ **TIFF** - Universal, well-documented
- ✅ **HDF5** - Modern standard, Python has excellent support
- ✅ **NanoScope (.spm)** - Very common in AFM community

### Phase 2 (Important):
- ✅ **JPK (.jpk, .h5-jpk)** - Bruker instruments
- ✅ **IBW** - Igor Pro format
- ✅ **ASD** - RIBM format

### Phase 3 (Nice to Have):
- ⏸️ **Gwyddion (.gwy)** - Open source AFM software format
- ⏸️ **NHF** - NanoHybrid format
- ⏸️ **ARIS** - Proprietary format

---

## Testing Strategy

### Unit Tests
```python
# tests/test_detector.py
def test_peak_detection():
    # Test Fast_peaks2D equivalent
    pass

def test_localization_precision():
    # Compare sub-pixel accuracy to MATLAB
    pass

def test_gpu_cpu_consistency():
    # Ensure GPU and CPU produce same results
    pass
```

### Integration Tests
- Process Example Data from original repo
- Compare outputs to MATLAB reference
- Validate file format round-trips

### Performance Benchmarks
```python
# benchmarks/benchmark_alignment.py
def benchmark_fft_alignment():
    # Compare MATLAB vs Python CPU vs Python GPU
    pass
```

---

## Project Structure

```
nanolocz-py/
├── README.md
├── LICENSE (GPL v3.0)
├── setup.py / pyproject.toml
├── requirements.txt
├── docs/
│   ├── installation.md
│   ├── user_guide/
│   └── api_reference/
├── nanolocz/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── io.py              # File I/O
│   │   ├── preprocessing.py   # Leveling, filtering
│   │   ├── detection.py       # Particle detection
│   │   ├── localization.py    # Sub-pixel localization
│   │   ├── tracking.py        # Particle tracking
│   │   └── alignment.py       # Image alignment
│   ├── gpu/
│   │   ├── __init__.py
│   │   ├── utils.py           # GPU abstraction
│   │   ├── kernels.py         # Custom CUDA kernels
│   │   └── accelerated_ops.py # GPU versions of core ops
│   ├── formats/
│   │   ├── __init__.py
│   │   ├── spm.py
│   │   ├── jpk.py
│   │   ├── ibw.py
│   │   ├── asd.py
│   │   └── hdf5.py
│   ├── lafm/
│   │   ├── renderer.py
│   │   ├── plotter.py
│   │   └── frc.py
│   ├── simafm/
│   │   └── simulation.py
│   └── gui/
│       ├── __init__.py
│       ├── main_window.py
│       ├── widgets/
│       └── viewers/
├── tests/
│   ├── test_io.py
│   ├── test_detection.py
│   └── ...
├── benchmarks/
│   ├── benchmark_core.py
│   └── benchmark_gpu.py
├── examples/
│   ├── basic_workflow.ipynb
│   └── gpu_acceleration.ipynb
└── tools/
    ├── convert_matlab_tests.py
    └── performance_profiler.py
```

---

## Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|-----------|
| GPU not available | Graceful fallback to CPU with Numba JIT |
| Proprietary format specs unavailable | Community collaboration, reverse engineering |
| Performance doesn't match MATLAB | Profile early, optimize hot paths, consider Cython |
| GUI complexity | Start with CLI/Jupyter, add GUI incrementally |

### Legal Risks
- ✅ Maintain GPL v3.0 license
- ✅ Credit original authors
- ✅ Document all ported algorithms
- ⚠️ Avoid copying MATLAB toolbox code (reimplement from scratch)

### Community Adoption
- Engage with AFM research community early
- Publish benchmark comparisons
- Provide easy migration path from MATLAB version
- Offer both CLI and GUI interfaces

---

## Success Metrics

### Functional Completeness
- [ ] All file formats supported
- [ ] All core algorithms implemented
- [ ] GUI feature parity (or better)
- [ ] Export options complete

### Performance Targets
- [ ] CPU version within 2x of MATLAB speed
- [ ] GPU version 5-10x faster than MATLAB
- [ ] Memory usage < MATLAB equivalent
- [ ] Startup time < 5 seconds

### Quality Standards
- [ ] >90% test coverage
- [ ] Comprehensive documentation
- [ ] Cross-platform testing (Linux, macOS, Windows)
- [ ] Peer-reviewed publication (optional)

---

## Immediate Next Steps

### Week 1 Actions:
1. **Create GitHub organization/repository**
   - Initialize with this planning document
   - Set up issue tracker with phase milestones

2. **Set up development environment**
   ```bash
   python -m venv nanolocz-env
   pip install numpy scipy scikit-image opencv-python h5py
   pip install cupy-cuda12x  # If NVIDIA GPU available
   pip install pytest matplotlib
   ```

3. **Clone and analyze Example Data**
   - Test file format parsers
   - Create ground truth datasets

4. **Implement first file reader** (TIFF or HDF5)
   - Prove concept
   - Establish coding patterns

5. **Port simplest algorithm** (e.g., `Fast_peaks2D.m`)
   - Create test against MATLAB output
   - Establish GPU abstraction pattern

---

## Contributing Guidelines

### For Developers:
1. Follow PEP 8 style guidelines
2. Write tests for all new features
3. Document GPU/CPU behavior differences
4. Benchmark performance changes
5. Maintain backward compatibility where possible

### For Users:
1. Report bugs with reproducible examples
2. Share AFM file format samples
3. Provide feedback on GUI usability
4. Contribute to documentation

---

## Contact & Resources

### Original Project:
- GitHub: https://github.com/george-r-heath/NanoLocz
- Publication: DOI: 10.1002/smtd.202301766
- User Guide: https://george-r-heath.github.io/NanoLocz/docs/

### Key References:
- CuPy Documentation: https://docs.cupy.dev/
- Scikit-image: https://scikit-image.org/
- PyQt6: https://www.riverbankcomputing.com/static/Docs/PyQt6/

### AFM Community Resources:
- Open AFM Data Repositories (see NanoLocz docs)
- Gwyddion project (open source AFM software)
- BioAFM community forums

---

## Appendix: Algorithm Details

### A.1 DFT Registration (from `align_trans.m`)
The `dftregistration` function implements frequency-domain image registration:
1. Compute FFT of both images
2. Calculate cross-power spectrum
3. Find peak in correlation surface
4. Refine to sub-pixel accuracy using upsampling
5. Return translation parameters

**Python Implementation**: Already exists in `scikit-image` as `phase_cross_correlation`

### A.2 2D Gaussian Fitting (from `localize.m`)
Fits: `I(x,y) = A * exp(-((x-x0)²/(2σx²) + (y-y0)²/(2σy²)))`

**Python Implementation**: Use `lmfit` or `scipy.optimize.curve_fit`

### A.3 Sphere Fitting (from `localize.m`)
Algebraic sphere fit using least squares:
- Solves for center (xc, yc, zc) and radius R
- Implemented as `sumith_fit` nested function

**Python Implementation**: Direct port of algebraic solution

### A.4 Particle Tracking (from `track_particles.m`)
Uses Hungarian algorithm with gap-closing:
1. Link particles between consecutive frames
2. Allow gaps up to N frames
3. Minimize total displacement

**Python Implementation**: `scipy.optimize.linear_sum_assignment` + custom gap logic

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Author: AI Code Assistant*  
*License: This planning document is released under the same GPL v3.0 license as NanoLocz*
