# NanoLocz Python Port - Action Plan Summary

## Overview

This document provides a concise action plan for porting NanoLocz from MATLAB to Python with GPU support for Linux systems.

## Phase 1: Immediate Actions (Week 1)

### 1. Repository Setup
```bash
# Create new GitHub repository
- Name: nanolocz-py or NanoLocz-python
- License: GPL v3.0 (same as original)
- Add README.md with project goals
- Set up issue tracker with milestones
```

### 2. Development Environment
```bash
cd /workspace/NanoLocz-Python-Port

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install numpy scipy scikit-image opencv-python h5py tifffile matplotlib pytest

# Optional: GPU support (NVIDIA CUDA required)
pip install cupy-cuda12x

# For development
pip install black flake8 mypy pre-commit
```

### 3. Project Structure
```
nanolocz-py/
├── nanolocz/           # Main package
│   ├── __init__.py
│   ├── core/           # Core algorithms
│   ├── gpu/            # GPU acceleration
│   ├── formats/        # File format readers
│   └── gui/            # User interface
├── tests/              # Test suite
├── examples/           # Example notebooks
├── docs/               # Documentation
├── benchmarks/         # Performance tests
└── tools/              # Utilities
```

### 4. First Implementation Priority

**Start with these files (in order):**

1. **`nanolocz/core/detection.py`** - Port `Fast_peaks2D.m`
   - Simplest algorithm
   - Establishes testing pattern
   - No external dependencies

2. **`nanolocz/formats/tiff_reader.py`** - TIFF file support
   - Well-documented format
   - Easy to test with example data

3. **`nanolocz/gpu/utils.py`** - GPU abstraction layer
   - Enables GPU acceleration from the start
   - Graceful CPU fallback

## Phase 2: Core Algorithms (Weeks 2-6)

### Week 2-3: Image Processing
- [ ] `level.py` - Image leveling (polynomial fitting)
- [ ] `filtering.py` - Gaussian filtering, temporal filters
- [ ] `alignment.py` - FFT-based image registration

### Week 4-5: Particle Detection
- [ ] `Detector.py` - Full particle detection pipeline
- [ ] `localize.py` - Sub-pixel localization (Gaussian/sphere fit)

### Week 6: Particle Tracking
- [ ] `tracking.py` - Hungarian algorithm with gap closing

## Phase 3: File Formats (Weeks 7-10)

Priority order based on AFM community usage:
1. [ ] TIFF (universal)
2. [ ] HDF5 (modern standard)
3. [ ] NanoScope .spm (Bruker)
4. [ ] JPK (.jpk, .h5-jpk)
5. [ ] IBW (Igor Pro)
6. [ ] ASD (RIBM)

## Phase 4: GPU Optimization (Weeks 11-14)

Focus on compute-intensive operations:
- [ ] Cross-correlation (Detector, alignment)
- [ ] FFT operations
- [ ] Image resizing/interpolation
- [ ] Gaussian filtering
- [ ] Morphological operations

Benchmark targets:
- 5-10x speedup over MATLAB for large images (>1024x1024)
- <2x slowdown for small images (overhead acceptable)

## Phase 5: GUI Development (Weeks 15-24)

Choose one approach:

**Option A: Desktop Application (PyQt6)**
- Full-featured desktop app
- Similar to MATLAB App Designer
- Better performance for large datasets

**Option B: Web Application (Streamlit/Dash)**
- Easier deployment
- Accessible from any browser
- Good for collaboration

**Hybrid Approach:**
- Start with Jupyter notebooks for algorithm development
- Build PyQt6 GUI for production use
- Offer web viewer for sharing results

## Key Technical Decisions

### GPU Backend
**Recommended: CuPy**
- Pros: NumPy-compatible, easy migration, good documentation
- Cons: Requires NVIDIA GPU, CUDA installation

**Alternative: PyTorch**
- Pros: More flexible, better for future ML features
- Cons: Steeper learning curve, overkill for basic ops

**Fallback: Numba**
- Pros: Works on CPU, can compile to CUDA
- Cons: More complex code, slower than CuPy for some ops

### File Format Strategy
1. Use existing libraries where possible (tifffile, h5py)
2. For proprietary formats:
   - Check if specs are publicly available
   - Look at Gwyddion source code for reference
   - Request sample files from AFM community
   - Consider reverse engineering (legal for interoperability)

### Testing Strategy
```python
# Test structure
tests/
├── test_detection.py      # Fast_peaks2D, Detector
├── test_localization.py   # Sub-pixel accuracy
├── test_tracking.py       # Particle tracking
├── test_io.py            # File format readers
└── test_gpu.py           # GPU vs CPU consistency
```

Test data: Use Example Data from original NanoLocz repo

## Performance Benchmarks

Create benchmark suite comparing:
- MATLAB NanoLocz (reference)
- Python CPU implementation
- Python GPU implementation

Key metrics:
- Execution time
- Memory usage
- Localization precision (nm)
- Detection accuracy (% true positives)

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Proprietary format specs unavailable | High | Medium | Focus on open formats first, collaborate with vendors |
| GPU performance disappointing | Low | Medium | Profile early, optimize hot paths, consider Cython |
| Community adoption slow | Medium | High | Publish benchmarks, engage AFM community, present at conferences |
| MATLAB feature parity difficult | Medium | Medium | Prioritize most-used features, add advanced features incrementally |

## Success Metrics

### MVP (Minimum Viable Product) - Month 3
- [ ] Read TIFF and HDF5 files
- [ ] Display images with histogram controls
- [ ] Detect particles (peak picking mode)
- [ ] Export results to CSV
- [ ] Basic CLI interface

### Version 1.0 - Month 6
- [ ] All core processing algorithms
- [ ] 5+ file formats supported
- [ ] GPU acceleration working
- [ ] Test coverage >80%
- [ ] Documentation complete

### Version 1.0 Full - Month 9
- [ ] GUI complete
- [ ] LAFM workflow
- [ ] Simulation AFM
- [ ] Batch processing
- [ ] Publication-ready

## Community Engagement

### Where to Find Users/Contributors
1. **AFM Research Groups**
   - University labs using NanoLocz
   - Authors citing NanoLocz paper (DOI: 10.1002/smtd.202301766)

2. **Online Communities**
   - BioAFM mailing list
   - Reddit r/biophysics
   - Twitter/X #AFM #StructuralBiology

3. **Conferences**
   - Biophysical Society Annual Meeting
   - Microscopy & Microanalysis
   - AFM-specific workshops

### How to Contribute
- Report bugs via GitHub Issues
- Submit pull requests for features
- Share AFM file format samples
- Write tutorials/examples
- Translate documentation

## Legal Considerations

✅ **Allowed:**
- Reimplementing algorithms from scratch
- Using same file format specifications (if public)
- Creating compatible software under GPL v3.0

⚠️ **Avoid:**
- Copying MATLAB toolbox code directly
- Using proprietary MathWorks code
- Violating file format patents (if any)

📄 **License Requirements:**
- Must maintain GPL v3.0
- Must credit original authors
- Must make source code available
- Derivative works must also be GPL v3.0

## Resources

### Documentation
- Original NanoLocz: https://github.com/george-r-heath/NanoLocz
- User Guide: https://george-r-heath.github.io/NanoLocz/docs/
- Publication: https://doi.org/10.1002/smtd.202301766

### Python Libraries
- CuPy: https://docs.cupy.dev/
- Scikit-image: https://scikit-image.org/
- OpenCV: https://docs.opencv.org/
- PyQt6: https://www.riverbankcomputing.com/static/Docs/PyQt6/

### AFM Community
- Gwyddion (open source AFM software): http://gwyddion.net/
- BioAFM community forums
- Open AFM Data Repositories (see NanoLocz docs)

## Next Meeting Agenda

For team kickoff meeting:
1. Review this action plan
2. Assign initial tasks
3. Set up development environment together
4. Choose first file format to implement
5. Schedule weekly check-ins
6. Discuss long-term vision

---

**Document Created:** January 2026  
**Status:** Ready for implementation  
**Priority:** HIGH - Start with Phase 1 immediately
