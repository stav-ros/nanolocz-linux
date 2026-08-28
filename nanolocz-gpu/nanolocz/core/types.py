"""
Typed core contracts for NanoLocz.

NL-03: Typed Contracts Implementation

This module defines the core data structures and type contracts used throughout
the NanoLocz package. These types ensure consistency between MATLAB and Python
implementations and provide clear interfaces for GPU-accelerated operations.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple, Optional, Union, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Basic Type Aliases
# =============================================================================

#: 2D image array (height, width)
Image2D = np.ndarray

#: 3D image stack (frames, height, width)
ImageStack = np.ndarray

#: Array of particle coordinates (N, 2) where columns are [x, y]
Coordinates2D = np.ndarray

#: Array of particle coordinates with frame (N, 3) where columns are [frame, x, y]
Coordinates3D = np.ndarray

#: Localization results table
LocalizationTable = np.ndarray

#: Boolean mask for image regions
Mask2D = np.ndarray


# =============================================================================
# Detection Types
# =============================================================================

class DetectionMethod(Enum):
    """Particle detection algorithm selection."""
    FAST_PEAKS = "fast_peaks"
    LAPLACIAN = "laplacian"
    WAVELET = "wavelet"
    CUSTOM = "custom"


@dataclass(frozen=True)
class DetectionParams:
    """Parameters for particle detection algorithms.
    
    Attributes
    ----------
    threshold : float
        Intensity threshold for peak detection
    min_distance : int
        Minimum distance between detected peaks (pixels)
    box_size : int
        Size of the analysis box around each peak
    method : DetectionMethod
        Detection algorithm to use
    """
    threshold: float
    min_distance: int = 3
    box_size: int = 7
    method: DetectionMethod = DetectionMethod.FAST_PEAKS


@dataclass
class DetectionResult:
    """Result from particle detection algorithm.
    
    Attributes
    ----------
    coordinates : Coordinates2D
        Detected peak coordinates [x, y]
    intensities : np.ndarray
        Peak intensities at detected locations
    scores : np.ndarray
        Detection confidence scores
    metadata : dict
        Additional detection metadata
    """
    coordinates: Coordinates2D
    intensities: np.ndarray
    scores: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("coordinates must be shape (N, 2)")
        if len(self.intensities) != len(self.coordinates):
            raise ValueError("intensities length must match coordinates")
        if len(self.scores) != len(self.coordinates):
            raise ValueError("scores length must match coordinates")


# =============================================================================
# Localization Types
# =============================================================================

class LocalizationModel(Enum):
    """Fitting model for sub-pixel localization."""
    GAUSSIAN_2D = "gaussian_2d"
    SPHERE = "sphere"
    ELLIPSE = "ellipse"
    INTERPOLATED = "interpolated"


@dataclass(frozen=True)
class LocalizationParams:
    """Parameters for sub-pixel localization.
    
    Attributes
    ----------
    model : LocalizationModel
        Fitting model to use
    box_size : int
        Size of the fitting box around each peak
    max_iterations : int
        Maximum iterations for optimization
    tolerance : float
        Convergence tolerance
    """
    model: LocalizationModel = LocalizationModel.GAUSSIAN_2D
    box_size: int = 9
    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass
class LocalizedParticle:
    """Single localized particle with sub-pixel precision.
    
    Attributes
    ----------
    x : float
        Sub-pixel x coordinate
    y : float
        Sub-pixel y coordinate
    intensity : float
        Fitted intensity
    sigma_x : float
        Fitted sigma in x direction (if applicable)
    sigma_y : float
        Fitted sigma in y direction (if applicable)
    background : float
        Estimated local background
    chi_squared : float
        Goodness of fit metric
    frame : int
        Frame number (for stacks)
    """
    x: float
    y: float
    intensity: float
    sigma_x: Optional[float] = None
    sigma_y: Optional[float] = None
    background: float = 0.0
    chi_squared: float = float('nan')
    frame: int = 0
    
    @property
    def position(self) -> np.ndarray:
        """Return position as numpy array."""
        return np.array([self.x, self.y])
    
    @property
    def sigma(self) -> Optional[np.ndarray]:
        """Return sigma as numpy array if available."""
        if self.sigma_x is not None and self.sigma_y is not None:
            return np.array([self.sigma_x, self.sigma_y])
        return None


# =============================================================================
# Tracking Types
# =============================================================================

@dataclass(frozen=True)
class TrackParams:
    """Parameters for particle tracking.
    
    Attributes
    ----------
    max_displacement : float
        Maximum displacement between frames (pixels)
    gap_closing : int
        Maximum number of frames to close gaps
    memory : int
        Memory for track linking
    """
    max_displacement: float = 5.0
    gap_closing: int = 2
    memory: int = 1


@dataclass
class ParticleTrack:
    """Complete trajectory of a tracked particle.
    
    Attributes
    ----------
    track_id : int
        Unique track identifier
    particles : List[LocalizedParticle]
        List of localized particles in temporal order
    frames : np.ndarray
        Frame numbers for each localization
    displacements : np.ndarray
        Frame-to-frame displacements
    """
    track_id: int
    particles: List[LocalizedParticle]
    frames: np.ndarray = field(init=False)
    displacements: np.ndarray = field(init=False)
    
    def __post_init__(self):
        object.__setattr__(self, 'frames', 
                          np.array([p.frame for p in self.particles]))
        
        if len(self.particles) > 1:
            positions = np.array([p.position for p in self.particles])
            diffs = np.diff(positions, axis=0)
            object.__setattr__(self, 'displacements', 
                              np.sqrt(np.sum(diffs**2, axis=1)))
        else:
            object.__setattr__(self, 'displacements', np.array([]))
    
    @property
    def duration(self) -> int:
        """Track duration in frames."""
        if len(self.frames) < 2:
            return 1
        return int(self.frames[-1] - self.frames[0] + 1)
    
    @property
    def mean_displacement(self) -> float:
        """Mean frame-to-frame displacement."""
        if len(self.displacements) == 0:
            return 0.0
        return float(np.mean(self.displacements))


# =============================================================================
# File I/O Types
# =============================================================================

class FileFormat(Enum):
    """Supported file formats."""
    TIFF = "tiff"
    HDF5 = "hdf5"
    NPZ = "npz"
    MAT = "mat"
    UNKNOWN = "unknown"


@dataclass
class ImageMetadata:
    """Metadata for image files.
    
    Attributes
    ----------
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    frames : int
        Number of frames (1 for single images)
    pixel_size : Optional[float]
        Pixel size in physical units (if known)
    pixel_size_unit : str
        Unit for pixel_size (e.g., 'nm', 'um')
    bit_depth : int
        Bit depth per pixel
    format : FileFormat
        File format
    """
    width: int
    height: int
    frames: int = 1
    pixel_size: Optional[float] = None
    pixel_size_unit: str = "pixel"
    bit_depth: int = 16
    format: FileFormat = FileFormat.UNKNOWN


# =============================================================================
# GPU Execution Types
# =============================================================================

class DeviceType(Enum):
    """Computing device selection."""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


@dataclass(frozen=True)
class GPUConfig:
    """GPU execution configuration.
    
    Attributes
    ----------
    device_type : DeviceType
        Preferred computing device
    device_id : Optional[int]
        Specific GPU device ID (None for auto-selection)
    memory_limit : Optional[int]
        Maximum GPU memory to use (bytes)
    batch_size : int
        Batch size for GPU processing
    """
    device_type: DeviceType = DeviceType.AUTO
    device_id: Optional[int] = None
    memory_limit: Optional[int] = None
    batch_size: int = 64
    
    def resolve_device(self) -> DeviceType:
        """Resolve actual device to use (with fallback logic)."""
        if self.device_type == DeviceType.AUTO:
            # Try to detect GPU availability
            try:
                import cupy
                cupy.cuda.runtime.getDeviceCount()
                return DeviceType.GPU
            except (ImportError, Exception):
                return DeviceType.CPU
        return self.device_type


# =============================================================================
# Pipeline Result Types
# =============================================================================

@dataclass
class AnalysisPipeline:
    """Complete analysis pipeline result.
    
    Attributes
    ----------
    detections : List[DetectionResult]
        Detection results per frame
    localizations : List[List[LocalizedParticle]]
        Localized particles per frame
    tracks : List[ParticleTrack]
        Complete particle tracks
    metadata : ImageMetadata
        Image metadata
    config : Dict[str, Any]
        Configuration used for analysis
    """
    detections: List[DetectionResult]
    localizations: List[List[LocalizedParticle]]
    tracks: List[ParticleTrack]
    metadata: ImageMetadata
    config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_particles(self) -> int:
        """Total number of localized particles."""
        return sum(len(frame_locs) for frame_locs in self.localizations)
    
    @property
    def total_tracks(self) -> int:
        """Total number of tracked trajectories."""
        return len(self.tracks)
    
    def to_dataframe(self) -> Any:
        """Convert localizations to pandas DataFrame."""
        try:
            import pandas as pd
            
            rows = []
            for frame_idx, frame_locs in enumerate(self.localizations):
                for particle in frame_locs:
                    rows.append({
                        'frame': particle.frame,
                        'x': particle.x,
                        'y': particle.y,
                        'intensity': particle.intensity,
                        'sigma_x': particle.sigma_x,
                        'sigma_y': particle.sigma_y,
                        'background': particle.background,
                        'chi_squared': particle.chi_squared,
                    })
            
            return pd.DataFrame(rows)
        except ImportError:
            raise ImportError("pandas required for DataFrame conversion")
