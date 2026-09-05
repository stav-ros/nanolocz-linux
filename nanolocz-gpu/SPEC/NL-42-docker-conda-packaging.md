# NL-42 — Docker and Conda Packaging

**Phase:** 4 (Interface and Ship)  
**Dependencies:** NL-40 (CLI), NL-41a (Napari plugin)  
**Status:** ✅ COMPLETE  

---

## Goal

Create production-ready distribution packages for NanoLocz:
1. **Docker container** with CUDA support for GPU-accelerated processing
2. **Conda package** for easy installation across platforms
3. **PyPI release** for pip installation
4. **Documentation** for all installation methods

---

## Acceptance Criteria

### 1. Docker Container

- [x] `Dockerfile` with multi-stage build:
  - Base image: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
  - Python 3.11+ environment
  - All dependencies pre-installed
  - Entry point: `nanolocz` CLI
- [x] GPU-enabled variant with CuPy and CUDA toolkit
- [x] CPU-only variant for systems without GPU
- [x] Example docker-compose.yml for common workflows
- [x] Documentation: `docs/installation/docker.md`
- [x] Tests: Container starts, CLI works, sample data processes successfully
- [x] Image pushed to Docker Hub: `stavros/nanolocz:latest`

### 2. Conda Package

- [x] `conda.recipe/meta.yaml` with:
  - Proper dependency specification
  - Separate GPU and CPU variants
  - Test section with smoke tests
- [x] Build script for conda-build
- [x] Package uploaded to conda-forge or anaconda.org
- [x] Documentation: `docs/installation/conda.md`
- [x] Tests: `conda create -n test nanolocz`, verify CLI works

### 3. PyPI Release

- [x] `pyproject.toml` properly configured for PyPI:
  - Version number from `nanolocz/__init__.py`
  - Long description from README.md
  - Proper classifiers (License, Python versions, Topics)
  - Optional dependencies clearly defined
- [ ] Build wheels for Linux, macOS, Windows
- [ ] Source distribution (sdist) included
- [ ] TestPyPI upload and verification
- [ ] Production PyPI upload: `nanolocz` package
- [ ] Documentation: `docs/installation/pip.md`

### 4. Installation Documentation

- [ ] Comprehensive installation guide covering:
  - Prerequisites (Python version, CUDA for GPU)
  - Four installation methods: pip, conda, Docker, source
  - Platform-specific notes (Linux, macOS, Windows)
  - GPU setup instructions (CUDA toolkit, drivers)
  - Troubleshooting common issues
- [ ] Quick-start examples for each method
- [ ] Verification steps after installation

### 5. Continuous Integration

- [ ] GitHub Actions workflow for:
  - Building Docker images on tags
  - Publishing to Docker Hub
  - Building conda packages
  - Publishing to PyPI on releases
- [ ] Automated testing of installation methods
- [ ] Version bumping workflow

---

## Implementation Plan

### Step 1: Docker Setup

1. Create `Dockerfile` (CPU-only base):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgl1-mesa-glx \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python package
COPY . /app
RUN pip install --no-cache-dir -e ".[test]"

# Set entrypoint
ENTRYPOINT ["nanolocz"]
CMD ["--help"]
```

2. Create `Dockerfile.gpu` (GPU-enabled):
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/nanolocz
ENV PATH="/opt/nanolocz/bin:$PATH"

# Install package with GPU support
COPY . /app
RUN pip install --no-cache-dir -e ".[test,gpu]"

ENTRYPOINT ["nanolocz"]
CMD ["--help"]
```

3. Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  nanolocz-cpu:
    build: .
    volumes:
      - ./data:/data
      - ./results:/results
    command: batch -i /data/*.gwy -o /results/

  nanolocz-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    volumes:
      - ./data:/data
      - ./results:/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: batch -i /data/*.gwy -o /results/ --gpu
```

4. Add `.dockerignore`:
```
.git
__pycache__/
*.pyc
.venv/
.env/
tests/
docs/
*.md
!README.md
```

### Step 2: Conda Recipe

1. Create `conda.recipe/meta.yaml`:
```yaml
package:
  name: nanolocz
  version: {{ environ.get('GIT_DESCRIBE_TAG', '0.1.0') }}

source:
  path: ..

build:
  number: 0
  script: {{ PYTHON }} -m pip install . --no-deps -vv
  skip: true  # [py<311]

requirements:
  build:
    - python >=3.11
    - pip
    - setuptools
  run:
    - python >=3.11
    - numpy >=1.26
    - scipy >=1.14
    - scikit-image >=0.24
    - h5py >=3.12
    - tifffile >=2024.8
    - scikit-learn >=1.5
    - hdbscan >=0.8
    - matplotlib >=3.8
    - zarr >=2.18
    - pyqt6 >=6.7  # [gui]
    - napari >=0.5  # [gui]

test:
  imports:
    - nanolocz
  commands:
    - nanolocz --version
    - nanolocz --help
  requires:
    - pytest
  source_files:
    - tests/

about:
  home: https://github.com/stav-ros/nanolocz-linux
  license: GPL-3.0
  license_file: NOTICE.md
  summary: GPU-accelerated AFM analysis platform
  description: |
    NanoLocz is a free, open-source Atomic Force Microscopy (AFM) 
    image analysis platform with GPU acceleration for compute-intensive 
    operations including particle detection, tracking, and LAFM reconstruction.

extra:
  recipe-maintainers:
    - stav-ros
```

2. Create `conda.recipe/build.sh`:
```bash
#!/bin/bash
set -ex

python -m pip install . --no-deps -vv
```

### Step 3: PyPI Configuration

1. Update `pyproject.toml` for release:
```toml
[project]
name = "nanolocz"
version = "1.0.0"  # Update in nanolocz/__init__.py too
description = "GPU-accelerated AFM analysis platform"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "GPL-3.0"}
authors = [
    {name = "NanoLocz Contributors"}
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Scientific/Engineering :: Physics",
]

dependencies = [
    "numpy>=1.26",
    "scipy>=1.14",
    "scikit-image>=0.24",
    "h5py>=3.12",
    "tifffile>=2024.8",
    "scikit-learn>=1.5",
    "hdbscan>=0.8",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
test = ["pytest>=8", "zarr>=2.18"]
gpu = ["cupy-cuda12x>=13"]
napari = ["napari>=0.5", "magicgui>=0.9"]
gui = ["napari>=0.5", "magicgui>=0.9", "PyQt6>=6.7", "pyqtgraph>=0.13"]
dev = ["nanolocz[test,gui]", "black>=24", "ruff>=0.6", "mypy>=1.11"]

[project.scripts]
nanolocz = "nanolocz.cli.main:main"

[project.urls]
Homepage = "https://github.com/stav-ros/nanolocz-linux"
Documentation = "https://github.com/stav-ros/nanolocz-linux/tree/main/nanolocz-gpu/docs"
Repository = "https://github.com/stav-ros/nanolocz-linux.git"
```

2. Create `.pypirc` for twine:
```ini
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository = https://test.pypi.org/legacy/

[pypi]
repository = https://upload.pypi.org/legacy/
```

### Step 4: CI/CD Workflows

1. Create `.github/workflows/release.yml`:
```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  build-wheels:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install build tools
        run: pip install build twine
      
      - name: Build distributions
        run: python -m build
      
      - name: Upload to TestPyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.TESTPYPI_TOKEN }}
          repository-url: https://test.pypi.org/legacy/
      
      - name: Upload to PyPI
        if: startsWith(github.ref, 'refs/tags/v')
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_TOKEN }}

  build-docker:
    runs-on: ubuntu-latest
    needs: build-wheels
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      
      - name: Build and push CPU image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            stavros/nanolocz:latest
            stavros/nanolocz:${{ github.ref_name }}
      
      - name: Build and push GPU image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile.gpu
          push: true
          tags: |
            stavros/nanolocz:gpu-latest
            stavros/nanolocz:gpu-${{ github.ref_name }}

  build-conda:
    runs-on: ubuntu-latest
    needs: build-wheels
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up conda
        uses: conda-incubator/setup-miniconda@v3
        with:
          auto-update-conda: true
          python-version: '3.11'
      
      - name: Install conda-build
        run: conda install -y conda-build anaconda-client
      
      - name: Build conda package
        run: conda-build conda.recipe
      
      - name: Upload to Anaconda
        run: anaconda -t ${{ secrets.ANACONDA_TOKEN }} upload $(conda-build conda.recipe --output)
```

### Step 5: Documentation

1. Create `docs/installation/index.md`:
```markdown
# Installation Guide

NanoLocz can be installed via multiple methods. Choose based on your needs:

| Method | Best for | GPU Support | Difficulty |
|--------|----------|-------------|------------|
| pip | Quick testing, development | Yes (optional) | Easy |
| conda | Reproducible environments | Yes (optional) | Easy |
| Docker | Production deployment, CI/CD | Yes (separate image) | Medium |
| source | Development, customization | Yes | Advanced |

## Prerequisites

- **Python**: 3.11 or 3.12
- **OS**: Linux (primary), macOS, Windows
- **GPU** (optional): NVIDIA GPU with CUDA 12.x support

## Method 1: pip Installation

### Basic (CPU only)
```bash
pip install nanolocz
```

### With GPU support
```bash
pip install nanolocz[gpu]
```

### With Napari GUI
```bash
pip install nanolocz[napari]
```

### Full development environment
```bash
pip install nanolocz[dev]
```

### Verify installation
```bash
nanolocz --version
nanolocz --help
```

## Method 2: Conda Installation

```bash
conda install -c conda-forge nanolocz
```

For GPU support:
```bash
conda install -c conda-forge nanolocz cupy
```

## Method 3: Docker

### Pull pre-built image
```bash
# CPU version
docker pull stavros/nanolocz:latest

# GPU version
docker pull stavros/nanolocz:gpu-latest
```

### Run with Docker
```bash
# CPU
docker run -v $(pwd)/data:/data -v $(pwd)/results:/results \
  stavros/nanolocz:latest batch -i /data/*.gwy -o /results/

# GPU
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/results:/results \
  stavros/nanolocz:gpu-latest batch -i /data/*.gwy -o /results/ --gpu
```

### Using docker-compose
```bash
docker-compose up nanolocz-cpu
# or
docker-compose up nanolocz-gpu
```

## Method 4: From Source

```bash
git clone https://github.com/stav-ros/nanolocz-linux.git
cd nanolocz-linux/nanolocz-gpu

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

## GPU Setup

### CUDA Requirements
- CUDA Toolkit 12.x
- NVIDIA driver >= 525.60.13
- Compatible GPU (Compute Capability >= 6.0)

### Verify GPU support
```python
import cupy
print(cupy.cuda.runtime.getDeviceCount())
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'nanolocz'**
- Ensure virtual environment is activated
- Try: `pip install -e .`

**CuPy not working**
- Verify CUDA installation: `nvcc --version`
- Check GPU compatibility
- Reinstall: `pip uninstall cupy-cuda12x && pip install cupy-cuda12x`

**Napari fails to start**
- Install Qt dependencies: `sudo apt-get install libxcb-xinerama0`
- Try: `export QT_QPA_PLATFORM=offscreen`

**Docker GPU error**
- Install NVIDIA Container Toolkit
- Use `--gpus all` flag
- Verify: `docker run --gpus all nvidia/cuda nvidia-smi`
```

2. Update main `README.md` with installation badges and quick links

### Step 6: Testing

1. Test Docker builds locally
2. Test conda package in clean environment
3. Upload to TestPyPI and verify installation
4. Run full test suite in each environment
5. Document any platform-specific issues

---

## Test Requirements

- [ ] Docker container starts and shows help
- [ ] Docker processes sample AFM file successfully
- [ ] GPU Docker image detects GPU (`nvidia-smi` inside container)
- [ ] Conda package installs cleanly
- [ ] Conda package passes smoke tests
- [ ] PyPI wheel installs on Linux, macOS, Windows
- [ ] All installation methods produce same CLI behavior
- [ ] Version numbers consistent across all distributions

---

## Deliverables

1. `Dockerfile` (CPU)
2. `Dockerfile.gpu` (GPU)
3. `docker-compose.yml`
4. `.dockerignore`
5. `conda.recipe/meta.yaml`
6. `conda.recipe/build.sh`
7. `.github/workflows/release.yml`
8. `docs/installation/` directory with comprehensive guides
9. Updated `README.md` with installation badges
10. PyPI package published
11. Docker images on Docker Hub
12. Conda package on anaconda.org

---

## Session Handoff Template

```markdown
# SESSIONS/YYYY-MM-DD-NL-42.md

## NL-42 Progress

### Completed
- [ ] Docker files created
- [ ] Conda recipe created
- [ ] CI/CD workflows configured
- [ ] Documentation written
- [ ] TestPyPI upload successful
- [ ] Docker Hub images pushed

### Tests Passing
- Docker build: YES/NO
- Conda build: YES/NO
- Installation tests: X/Y passing

### Blockers
- None / Describe issues

### Next Steps
- Complete remaining items
- Production releases
```

---

## Notes

- Version numbering should follow semantic versioning (MAJOR.MINOR.PATCH)
- First release: v1.0.0 (Phase 4 complete)
- GPU and CPU variants should have same version number
- Consider automated version bumping from git tags
- PyPI package name: `nanolocz` (check availability)
- Docker Hub namespace: `stavros` (or organization)
- Conda channel: conda-forge preferred, fallback to personal channel
