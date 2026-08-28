"""
GPU acceleration utilities for NanoLocz.

Provides abstraction layer for CPU/GPU array operations using CuPy.
Gracefully falls back to NumPy when GPU is unavailable.
"""

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def get_array_module(use_gpu=False):
    """
    Get numpy or cupy module based on availability and preference.
    
    Parameters
    ----------
    use_gpu : bool
        Whether to use GPU acceleration if available
        
    Returns
    -------
    module : numpy or cupy
        Array module to use for computations
    """
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


def to_gpu(array, use_gpu=False):
    """
    Transfer array to GPU if requested and available.
    
    Parameters
    ----------
    array : ndarray
        Input array (CPU or GPU)
    use_gpu : bool
        Whether to transfer to GPU
        
    Returns
    -------
    array : ndarray
        Array on target device
    """
    if use_gpu and CUPY_AVAILABLE:
        return cp.asarray(array)
    return np.asarray(array)


def from_gpu(array):
    """
    Transfer array from GPU to CPU if needed.
    
    Parameters
    ----------
    array : ndarray
        Input array (CPU or GPU)
        
    Returns
    -------
    array : ndarray
        Array on CPU
    """
    if hasattr(array, 'get'):  # CuPy array
        return array.get()
    return array


class GPUArrayModule:
    """
    Abstraction layer for CPU/GPU array operations.
    
    Provides unified interface for common array operations that can be
    accelerated on GPU. Automatically uses CuPy when available and enabled.
    
    Examples
    --------
    >>> gpu_mod = GPUArrayModule(use_gpu=True)
    >>> img_gpu = gpu_mod.to_device(image)
    >>> filtered = gpu_mod.gaussian_filter(img_gpu, sigma=1.5)
    >>> result = gpu_mod.from_device(filtered)
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize GPU module.
        
        Parameters
        ----------
        use_gpu : bool
            Enable GPU acceleration if available
        """
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
    
    def ifft2(self, img):
        """GPU-accelerated 2D inverse FFT."""
        if self.use_gpu:
            return cp.fft.ifft2(img)
        return np.fft.ifft2(img)
    
    def resize(self, img, scale, method='bilinear'):
        """GPU-accelerated image resizing."""
        order = {'bilinear': 1, 'bicubic': 3, 'lanczos': 4}.get(method, 1)
        if self.use_gpu:
            from cupyx.scipy.ndimage import zoom
            return zoom(img, scale, order=order)
        else:
            from scipy.ndimage import zoom
            return zoom(img, scale, order=order)
    
    def maximum_filter(self, img, size):
        """GPU-accelerated maximum filter."""
        if self.use_gpu:
            from cupyx.scipy.ndimage import maximum_filter
            return maximum_filter(img, size=size)
        else:
            from scipy.ndimage import maximum_filter
            return maximum_filter(img, size=size)
