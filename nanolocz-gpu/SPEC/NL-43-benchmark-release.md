# NL-43 — Benchmark report and v1.0-gpu release

**Phase:** P4 — Interface and Ship  
**Depends:** NL-40 (CLI), NL-41 (Napari plugin), NL-42 (Docker/conda packaging)  
**Priority:** Final card before v1.0 release  
**Estimated effort:** 2-3 days  

---

## Objective

Produce a comprehensive benchmark report demonstrating NanoLocz v1.0 capabilities across CPU and GPU configurations, validate all workflows with real AFM data, package release artifacts, and publish v1.0-gpu to PyPI, Docker Hub, and conda-forge.

This card represents the **final milestone** before the official v1.0 release announcement.

---

## Scope

### 1. Benchmark Suite

Create `benchmarks/` directory with reproducible performance tests:

**Performance benchmarks:**
- File I/O throughput (`.gwy`, `.h5-jpk`, `.spm`, `.jpk`, `.asd`, `.zarr`)
- Preprocessing speed (leveling, filtering, scar removal) on CPU vs GPU
- Detection throughput (particles/frame, frames/sec)
- Tracking performance (trajectories length, gap closing)
- LAFM splatting performance (localizations count, resolution)
- FRC computation time
- 3D reconstruction time (back-projection, SIRT iterations)
- Memory usage profiling

**Dataset sizes:**
- Small: 64×64, 10 frames
- Medium: 256×256, 100 frames
- Large: 512×512, 500 frames
- Extra large: 1024×1024, 1000 frames (optional)

**Hardware configurations:**
- CPU-only (Intel/AMD)
- GPU entry-level (GTX 16xx, RTX 3050)
- GPU mid-range (RTX 3060, 3070)
- GPU high-end (RTX 3090, 4090, A100)

### 2. Validation Workflows

Run complete end-to-end workflows on provided test datasets:

**Workflow 1: Basic AFM analysis**
```bash
nanolocz preprocess --input sample.gwy --output preprocessed.zarr
nanolocz detect --input preprocessed.zarr --output detected.zarr
nanolocz track --input detected.zarr --output tracked.zarr
nanolocz lafm --input tracked.zarr --output lafm_result.zarr --frc
```

**Workflow 2: Advanced LAFM+**
```bash
nanolocz batch --config lafm_plus_config.json --input-dir raw_data/ --output-dir results/
# Includes: drift correction, substack extraction, PCA/HDBSCAN, alignment, averaging
```

**Workflow 3: Simulation parity**
```bash
# Load PDB, simulate AFM, compare with experimental data
python examples/simafm_workflow.py --pdb 2abc.pdb --output simulated.zarr
```

**Workflow 4: Napari GUI**
```bash
napari -w nanolocz
# Manual validation: open file, preprocess, detect, track, LAFM, export
```

### 3. Release Artifacts

**PyPI package:**
- Source distribution (`.tar.gz`)
- Wheel for Python 3.11+ (universal)
- TestPyPI validation before production PyPI

**Docker images:**
- `stavros/nanolocz:cpu-v1.0.0` (CPU-only, ~500MB)
- `stavros/nanolocz:gpu-v1.0.0` (CUDA 12.x, CuPy, ~2GB)
- `stavros/nanolocz:latest` (points to v1.0.0)

**Conda packages:**
- `conda-forge/nanolocz` (CPU-only)
- `conda-forge/nanolocz-gpu` (with CuPy CUDA support)

**Documentation:**
- API reference (auto-generated from docstrings)
- User guide (installation, CLI, GUI, workflows)
- Tutorial notebooks (Jupyter examples)
- Migration guide (MATLAB → Python)

### 4. Quality Gates

**Must pass before release:**
- ✅ All 473+ tests passing (GPU tests may skip if CuPy unavailable)
- ✅ Zero critical or high-severity bugs open
- ✅ Documentation complete and accurate
- ✅ Benchmarks run successfully on at least 2 hardware configurations
- ✅ Docker images build and run without errors
- ✅ Conda packages install and function correctly
- ✅ CLI smoke tests pass in clean environments
- ✅ Napari plugin loads and displays widget without errors
- ✅ License headers present in all source files
- ✅ SECURITY.md and CODE_OF_CONDUCT.md present

---

## Deliverables

### D1. Benchmark Report (`docs/benchmarks/v1.0-report.md`)

Markdown document containing:
- Executive summary
- Hardware configurations tested
- Performance tables (throughput, latency, memory)
- CPU vs GPU comparison graphs
- Scalability analysis (dataset size vs time)
- Recommendations for users

### D2. Benchmark Scripts (`benchmarks/`)

Executable Python scripts:
- `benchmarks/benchmark_io.py` — File I/O throughput
- `benchmarks/benchmark_preprocess.py` — Leveling/filtering speed
- `benchmarks/benchmark_detection.py` — Particle detection throughput
- `benchmarks/benchmark_tracking.py` — Tracking performance
- `benchmarks/benchmark_lafm.py` — LAFM splatting and FRC
- `benchmarks/benchmark_reconstruction.py` — 3D reconstruction time
- `benchmarks/run_all.sh` — Master script to run all benchmarks
- `benchmarks/plot_results.py` — Generate comparison graphs

### D3. Example Gallery (`examples/gallery/`)

Working examples with expected outputs:
- `01_basic_workflow.py` — Simple preprocess → detect → track → LAFM
- `02_batch_processing.py` — Process multiple files with config
- `03_lafm_plus.py` — Full LAFM+ pipeline with classification
- `04_simulation.py` — PDB to AFM simulation
- `05_deconvolution.py` — Tip estimation and deconvolution
- `06_dynamics.py` — HMM fitting and dwell time analysis
- `07_3d_reconstruction.py` — Volume reconstruction from particle stacks
- `08_napari_plugin.py` — Scripted Napari workflow (headless)

Each example includes:
- Well-commented code
- Expected runtime
- Sample input/output paths
- Visualization code (matplotlib/napari screenshots)

### D4. Release Checklist (`RELEASE_CHECKLIST.md`)

Comprehensive checklist covering:
- Code quality checks
- Test coverage verification
- Documentation review
- Security audit
- License compliance
- Distribution artifacts
- Announcement preparation

### D5. Version Tags and Releases

- Git tag: `v1.0.0`
- GitHub Release with changelog
- PyPI upload: `nanolocz==1.0.0`
- Docker Hub tags: `v1.0.0`, `latest`
- Conda-forge PR submitted

### D6. Session Handoff

`SESSIONS/YYYY-MM-DD-NL-43.md` documenting:
- Benchmark execution details
- Hardware used
- Issues encountered and resolved
- Final test counts
- Release artifact locations

---

## Acceptance Criteria

**NL-43 is DONE when:**

1. ✅ Benchmark suite created with 6+ benchmark scripts
2. ✅ Benchmarks executed on at least 2 different hardware configurations
3. ✅ Benchmark report published in `docs/benchmarks/v1.0-report.md`
4. ✅ Example gallery with 8+ working examples
5. ✅ All quality gates passed (see "Quality Gates" section above)
6. ✅ RELEASE_CHECKLIST.md completed and signed off
7. ✅ Git tag `v1.0.0` created
8. ✅ GitHub Release published with changelog
9. ✅ PyPI package uploaded and verified
10. ✅ Docker images built and pushed to Docker Hub
11. ✅ Conda-forge PR submitted (or package available)
12. ✅ Session handoff created in `SESSIONS/`
13. ✅ STATUS.md updated to mark Phase 4 COMPLETE
14. ✅ README.md updated with v1.0 badge and installation instructions

---

## Implementation Plan

### Step 1: Create Benchmark Infrastructure (0.5 day)

- Create `benchmarks/` directory structure
- Implement base benchmark runner utility
- Define dataset generators for different sizes
- Set up timing and memory profiling utilities

### Step 2: Write Benchmark Scripts (1 day)

- Implement 6 benchmark scripts (IO, preprocess, detection, tracking, LAFM, reconstruction)
- Add CSV/JSON output for results
- Create plotting script for visualization
- Test benchmarks on available hardware

### Step 3: Run Benchmarks and Generate Report (0.5 day)

- Execute benchmarks on CPU configuration
- Execute benchmarks on GPU configuration (if available)
- Compile results into markdown report
- Generate comparison graphs

### Step 4: Create Example Gallery (0.5 day)

- Write 8 example scripts with full documentation
- Test each example end-to-end
- Add expected outputs (screenshots, result files)
- Create index page in documentation

### Step 5: Quality Assurance (0.25 day)

- Run full test suite one final time
- Verify all quality gates
- Complete RELEASE_CHECKLIST.md
- Review documentation for accuracy

### Step 6: Release Packaging (0.25 day)

- Create git tag `v1.0.0`
- Build source distribution and wheel
- Upload to PyPI
- Build and push Docker images
- Submit conda-forge PR

### Step 7: Documentation Updates (0.25 day)

- Update STATUS.md to mark Phase 4 COMPLETE
- Update README.md with v1.0 information
- Create GitHub Release with changelog
- Update project website/dashboard

---

## Test Requirements

**Minimum 10 tests covering:**

1. Benchmark runner utility functions
2. Dataset generator correctness
3. Timing measurement accuracy
4. Memory profiling functionality
5. Results serialization (CSV/JSON)
6. Plot generation (non-interactive backend)
7. Example script execution (smoke tests)
8. Docker image basic functionality
9. CLI version command returns correct version
10. Package metadata (version, license, authors)

**Test file:** `tests/test_release_nl43.py`

---

## Configuration Files

### `pyproject.toml` updates

Ensure version is set to `1.0.0`:

```toml
[project]
name = "nanolocz"
version = "1.0.0"  # ← Update from 0.1.0.dev0
description = "High-performance AFM analysis with GPU acceleration"
# ... rest of metadata
```

### `.github/workflows/release.yml`

GitHub Actions workflow for automated release:

```yaml
name: Release v1.0.0

on:
  push:
    tags:
      - 'v1.0.0'

jobs:
  pypi-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install build tools
        run: pip install build twine
      - name: Build distribution
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_TOKEN }}
      - name: Build and push CPU image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: stavros/nanolocz:v1.0.0,stawros/nanolocz:latest
      - name: Build and push GPU image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.gpu
          push: true
          tags: stavros/nanolocz:gpu-v1.0.0
```

---

## Changelog Template

```markdown
# Changelog - v1.0.0

## 🎉 Major Features

### Core Analysis Pipeline
- Complete AFM file format support (.gwy, .h5-jpk, .spm, .jpk, .ibw, .asd, .tiff, .zarr)
- Advanced preprocessing: line/plane leveling, filters, scar removal
- High-performance particle detection with prominence and min-distance filtering
- Deterministic single-particle tracking with gap closing
- LAFM reconstruction with Fourier Ring Correlation resolution estimation
- 3D volume reconstruction from particle stacks (back-projection, SIRT)

### GPU Acceleration
- CUDA kernels for leveling, detection, LAFM splatting, FRC, and simulation
- Automatic CPU fallback when GPU unavailable
- Mixed precision support (float32/float64)
- 10-100x speedup on supported hardware

### Command-Line Interface
- Five subcommands: preprocess, detect, track, lafm, batch
- JSON configuration files for reproducible workflows
- Parallel batch processing with job queue
- Comprehensive help and error messages

### Napari Plugin
- Interactive AFM image/movie visualization
- Dock widget with preprocessing, detection, tracking, and LAFM controls
- Real-time layer overlays (points, tracks, masks, volumes)
- Export to Zarr, TIFF, PNG, CSV formats

### Simulation Tools
- PDB to AFM simulation workflow
- Conical tip convolution
- Noise and artifact modeling
- Tip estimation and deconvolution

## 📦 Installation

```bash
# CPU only
pip install nanolocz

# With GUI
pip install 'nanolocz[napari]'

# With GPU support (CUDA 12.x)
pip install 'nanolocz[gpu]'

# Docker
docker pull stavros/nanolocz:latest
docker pull stavros/nanolocz:gpu-v1.0.0
```

## 🔧 Technical Details

- Python >= 3.11
- Linux (primary), macOS, Windows
- GPL-3.0 license
- 473+ tests, full CI/CD coverage

## 🙏 Acknowledgments

Ported from MATLAB implementation by [original authors].
Built with NumPy, SciPy, scikit-image, CuPy, Napari, and many other open-source packages.
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU benchmarks fail on some hardware | Medium | Provide CPU-only benchmarks as fallback; document minimum CUDA compute capability |
| Conda-forge review delays release | Low | Release PyPI and Docker first; conda can follow post-release |
| Documentation incomplete | Medium | Use docstring auto-generation; focus on user-facing guides first |
| Last-minute bugs discovered | High | Freeze features 48h before release; only accept critical bug fixes |
| Docker image too large | Low | Use multi-stage builds; separate CPU and GPU images |

---

## Success Metrics

**Quantitative:**
- ✅ 473+ tests passing
- ✅ 6+ benchmark scripts functional
- ✅ 8+ example workflows documented
- ✅ 3 distribution channels (PyPI, Docker, conda)
- ✅ < 5 minute installation time for all methods
- ✅ Benchmarks show 10-100x GPU speedup where applicable

**Qualitative:**
- ✅ User can complete full AFM analysis workflow in < 5 commands
- ✅ Documentation enables new users to get started in < 30 minutes
- ✅ GPU acceleration accessible via simple `--gpu` flag
- ✅ Napari plugin provides intuitive GUI for interactive analysis
- ✅ Reproducible results across CPU/GPU configurations within tolerance

---

## Post-Release Tasks (Not Part of NL-43)

After v1.0.0 release:
- Announce on relevant mailing lists, social media, conferences
- Gather user feedback for v1.1.0 planning
- Monitor issue tracker for bug reports
- Consider NL-41b advanced features for next minor release
- Plan BioAFMviewer integration validation (NL-54)

---

## References

- SPEC/tasks.md — Task catalog
- STATUS.md — Current project status
- SPEC/NL-40-cli-batch-runner.md — CLI specification
- SPEC/NL-41-napari-plugin.md — Napari plugin specification
- SPEC/NL-42-docker-conda-packaging.md — Packaging specification
- SESSIONS/ — Session handoffs for all completed cards
