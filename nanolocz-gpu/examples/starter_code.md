# NanoLocz Python - Starter Implementation

This directory contains initial code examples for porting NanoLocz from MATLAB to Python.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install numpy scipy scikit-image opencv-python h5py matplotlib

# Optional: GPU support (requires NVIDIA CUDA)
pip install cupy-cuda12x

# Install in development mode
pip install -e .
```

## Example: Fast Peaks 2D Detection

MATLAB original: `NanoLocz-lib/Fast_peaks2D.m`

```python
import numpy as np
from scipy.ndimage import maximum_filter

def fast_peaks2d(img, thresh, kernel_size, min_prom=None):
    """
    Fast 2D peak detection for particle localization.
    
    Parameters
    ----------
    img : ndarray
        2D grayscale image
    thresh : float
        Intensity threshold for peak detection
    kernel_size : int
        Size of local neighborhood for maximum filtering
    min_prom : float, optional
        Minimum prominence for peak filtering
        
    Returns
    -------
    locs : ndarray
        Nx4 array of [x, y, height, prominence]
    """
    kernel_size = kernel_size + 2
    
    # Maximum filter to find local maxima
    max_filtered = maximum_filter(img, size=kernel_size, mode='constant')
    local_maxima_mask = (max_filtered == img) & (img > thresh)
    
    # Exclude edge pixels
    h, w = img.shape
    edge_margin = 2
    local_maxima_mask[:edge_margin, :] = False
    local_maxima_mask[-edge_margin:, :] = False
    local_maxima_mask[:, :edge_margin] = False
    local_maxima_mask[:, -edge_margin:] = False
    
    # Get coordinates and heights
    y_coords, x_coords = np.where(local_maxima_mask)
    peak_heights = img[local_maxima_mask]
    
    if len(x_coords) == 0:
        return np.array([]).reshape(0, 4)
    
    locs = np.column_stack([x_coords, y_coords, peak_heights])
    
    # Add prominence calculation if requested
    if min_prom is not None and min_prom > 0:
        prominences = calculate_prominence(img, locs[:, :2], peak_heights)
        keep = prominences > min_prom
        locs = np.column_stack([locs[keep, :3], prominences[keep]])
    else:
        prominences = np.zeros(len(x_coords))
        locs = np.column_stack([locs, prominences])
    
    return locs


def calculate_prominence(img, peaks, heights):
    """Calculate peak prominence (similar to MATLAB impofile approach)."""
    n_peaks = len(peaks)
    prominences = np.zeros(n_peaks)
    
    for j in range(n_peaks):
        # Calculate distances to all other peaks
        dists = np.sqrt(np.sum((peaks - peaks[j])**2, axis=1))
        
        # Sort by distance
        order = np.argsort(dists)
        
        # Find closest higher peak
        higher = heights[order] > heights[j]
        if np.any(higher):
            first_higher_idx = np.where(higher)[0][0]
            neighbor_idx = order[first_higher_idx]
            
            # Sample intensity profile between peaks
            from skimage.measure import profile_line
            profile = profile_line(
                img, 
                peaks[j], 
                peaks[neighbor_idx]
            )
            prominences[j] = heights[j] - np.min(profile)
        else:
            prominences[j] = heights[j]
    
    return prominences
```

## Example: GPU-Accelerated Operations

```python
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class GPUArrayModule:
    """Abstraction layer for CPU/GPU array operations."""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
    
    def to_device(self, array):
        """Transfer array to GPU if enabled."""
        if self.use_gpu:
            return cp.asarray(array)
        return np.asarray(array)
    
    def from_device(self, array):
        """Transfer array from GPU to CPU."""
        if hasattr(array, 'get'):
            return array.get()
        return array
    
    def gaussian_filter(self, img, sigma):
        """GPU-accelerated Gaussian filtering."""
        if self.use_gpu:
            from cupyx.scipy.ndimage import gaussian_filter
            return gaussian_filter(img, sigma)
        else:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(img, sigma)
    
    def fft2(self, img):
        """GPU-accelerated 2D FFT."""
        if self.use_gpu:
            return cp.fft.fft2(img)
        return np.fft.fft2(img)
    
    def resize(self, img, scale, method='bilinear'):
        """GPU-accelerated image resizing."""
        if self.use_gpu:
            from cupyx.scipy.ndimage import zoom
            return zoom(img, scale, order={'bilinear': 1, 'bicubic': 3}.get(method, 1))
        else:
            from scipy.ndimage import zoom
            return zoom(img, scale, order={'bilinear': 1, 'bicubic': 3}.get(method, 1))


# Usage example
def detect_with_gpu(img, use_gpu=True):
    gpu_module = GPUArrayModule(use_gpu=use_gpu)
    
    # Transfer to GPU
    img_gpu = gpu_module.to_device(img)
    
    # Process on GPU
    filtered = gpu_module.gaussian_filter(img_gpu, sigma=1.5)
    
    # Return to CPU
    result = gpu_module.from_device(filtered)
    return result
```

## Example: File I/O (HDF5)

```python
import h5py
import numpy as np

def read_h5_afm(filepath, channel='height', frames='all'):
    """
    Read AFM data from HDF5 file.
    
    Parameters
    ----------
    filepath : str
        Path to HDF5 file
    channel : str
        Channel name to read
    frames : int or 'all'
        Number of frames to load
        
    Returns
    -------
    data : ndarray
        3D image stack
    metadata : dict
        Image metadata
    """
    with h5py.File(filepath, 'r') as f:
        # Navigate to data
        if channel in f:
            dataset = f[channel]
        else:
            raise ValueError(f"Channel '{channel}' not found in file")
        
        # Load data
        if frames == 'all':
            data = dataset[:]
        else:
            data = dataset[:frames]
        
        # Extract metadata
        metadata = {}
        for key, value in f.attrs.items():
            metadata[key] = value
        
        # Try to get pixel size info
        if 'pixel_size' in f.attrs:
            metadata['pixel_size_nm'] = f.attrs['pixel_size']
        
        if 'scan_size' in f.attrs:
            metadata['scan_size_nm'] = f.attrs['scan_size']
    
    return data, metadata
```

## Testing

```bash
# Run tests
pytest tests/

# Run benchmarks
python benchmarks/benchmark_detection.py
```

## Next Steps

1. **Implement file format readers** - Start with TIFF and HDF5
2. **Port core algorithms** - Begin with `Fast_peaks2D.m` and `Detector.m`
3. **Create test suite** - Use Example Data from original NanoLocz repo
4. **Set up GPU benchmarks** - Compare CPU vs GPU performance
5. **Design GUI prototype** - PyQt6 or web-based interface

## Resources

- Original NanoLocz: https://github.com/george-r-heath/NanoLocz
- CuPy Documentation: https://docs.cupy.dev/
- Scikit-image: https://scikit-image.org/
- OpenCV Python: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

## License

This project is released under GPL v3.0, same as the original NanoLocz.
