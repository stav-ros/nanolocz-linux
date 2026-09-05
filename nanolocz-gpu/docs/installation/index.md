# Installation Guide

NanoLocz can be installed via multiple methods. Choose based on your needs:

| Method | Best for | GPU Support | Difficulty |
|--------|----------|-------------|------------|
| **pip** | Quick testing, development | Yes (optional) | Easy |
| **conda** | Reproducible environments | Yes (optional) | Easy |
| **Docker** | Production deployment, CI/CD | Yes (separate image) | Medium |
| **source** | Development, customization | Yes | Advanced |

## Prerequisites

- **Python**: 3.11 or 3.12
- **OS**: Linux (primary), macOS, Windows
- **GPU** (optional): NVIDIA GPU with CUDA 12.x support

---

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

### From source (editable install)

```bash
git clone https://github.com/stav-ros/nanolocz-linux.git
cd nanolocz-linux/nanolocz-gpu

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

---

## Method 2: Conda Installation

```bash
conda install -c conda-forge nanolocz
```

For GPU support:

```bash
conda install -c conda-forge nanolocz cupy
```

---

## Method 3: Docker

### Pull pre-built images

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
# CPU
docker-compose up nanolocz-cpu

# GPU
docker-compose up nanolocz-gpu
```

---

## Method 4: From Source

```bash
git clone https://github.com/stav-ros/nanolocz-linux.git
cd nanolocz-linux/nanolocz-gpu

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

Verify:

```bash
nanolocz --version
python -m pytest -q
```

---

## GPU Setup

### CUDA Requirements

- CUDA Toolkit 12.x
- NVIDIA driver >= 525.60.13
- Compatible GPU (Compute Capability >= 6.0)

### Install CUDA Toolkit

On Ubuntu:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/local/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1
```

### Verify GPU support

```python
import cupy
print(f"GPU count: {cupy.cuda.runtime.getDeviceCount()}")
print(f"GPU name: {cupy.cuda.runtime.getDeviceProperties(0)['name']}")
```

Or from command line:

```bash
nvidia-smi
```

---

## Optional Dependencies

### Format Support

| Format | Package | Notes |
|--------|---------|-------|
| `.tiff`, `.tif` | `tifffile` (included) | Standard TIFF files |
| `.zarr` | `zarr` (test extra) | Multi-dimensional arrays |
| `.h5`, `.hdf5` | `h5py` (included) | HDF5 format |
| `.gwy` | `gwymation` | Gwyddion format |
| `.jpk`, `.h5-jpk` | `h5py` (included) | JPK Instruments |
| `.spm` | Built-in | NanoScope format |
| `.ibw` | Optional | Igor Binary Wave |
| `.asd` | Built-in | JPK trace data |

### GUI Options

- **Napari plugin**: `pip install nanolocz[napari]`
- **Full GUI**: `pip install nanolocz[gui]` (includes PyQt6, pyqtgraph)

---

## Troubleshooting

### Common Issues

#### ImportError: No module named 'nanolocz'

- Ensure virtual environment is activated
- Try: `pip install -e .`
- Check Python version: `python --version` (must be >= 3.11)

#### CuPy not working

```bash
# Verify CUDA installation
nvcc --version

# Check GPU compatibility
nvidia-smi

# Reinstall CuPy
pip uninstall cupy-cuda12x
pip install cupy-cuda12x
```

#### Napari fails to start

```bash
# Install Qt dependencies (Linux)
sudo apt-get install libxcb-xinerama0 libgl1-mesa-glx

# Try software rendering
export QT_QPA_PLATFORM=offscreen

# Reinstall Napari
pip install --force-reinstall napari
```

#### Docker GPU error

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

#### Permission denied when writing output

```bash
# Ensure output directory exists and is writable
mkdir -p ./results
chmod 755 ./results

# Or run Docker with user mapping
docker run --user $(id -u):$(id -g) ...
```

### Getting Help

- Documentation: [GitHub Repository](https://github.com/stav-ros/nanolocz-linux)
- Issues: [GitHub Issues](https://github.com/stav-ros/nanolocz-linux/issues)
- Session notes: See `SESSIONS/` directory in repository

---

## Verification After Installation

Run these commands to verify your installation:

```bash
# Check version
nanolocz --version

# Show help
nanolocz --help

# Run tests (if installed with [test] extra)
python -m pytest -q

# Verify GPU (if installed with [gpu] extra)
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

# Test Napari plugin (if installed with [napari] extra)
python -c "from nanolocz.plugins.napari_plugin import NanoLoczWidget; print('Plugin OK')"
```

Expected output:

```
nanolocz version 1.0.0
usage: nanolocz [-h] [--version] {preprocess,detect,track,lafm,batch} ...
...
473 passed, 112 skipped
GPU count: 1
Plugin OK
```

---

## Next Steps

After successful installation:

1. **Quick Start**: See `README.md` for basic usage examples
2. **CLI Reference**: Run `nanolocz <command> --help` for detailed options
3. **Tutorials**: Check `examples/` directory for sample workflows
4. **GUI Usage**: Launch Napari and enable NanoLocz plugin from menu
5. **Advanced Topics**: Review `SPEC/` directory for detailed specifications
