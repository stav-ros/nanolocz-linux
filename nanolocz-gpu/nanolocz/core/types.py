"""Small, explicit data contracts for the port.

This module intentionally contains no numerical logic. The contracts are the seam
between file openers, analysis functions, and future NumPy/CuPy backends.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
import numpy as np
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class DeviceType(Enum):
    """Device type for GPU/CPU execution."""
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"
    CUDA = "cuda"


class FileFormat(Enum):
    """Supported file formats."""
    UNKNOWN = "unknown"
    TIFF = "tiff"
    HDF5 = "hdf5"
    NPZ = "npz"
    MAT = "mat"
    GWY = "gwy"
    SPM = "spm"
    JPK = "jpk"
    IBW = "ibw"
    ASD = "asd"


class DetectionMethod(Enum):
    """Detection algorithm methods."""
    FAST_PEAKS = "fast_peaks"
    THRESHOLD = "threshold"
    LAPLACIAN = "laplacian"
    WAVELET = "wavelet"
    CUSTOM = "custom"


class LocalizationModel(Enum):
    """Localization fitting models."""
    GAUSSIAN_2D = "gaussian_2d"
    SPHERE = "sphere"
    ELLIPSE = "ellipse"
    INTERPOLATED = "interpolated"
    MLE = "mle"
    INTERPOLATED_CENTROID = "interpolated_centroid"
    CUSTOM = "custom"


# ============================================================================
# Type aliases
# ============================================================================

Image2D = np.ndarray  # 2D numpy array


# ============================================================================
# Core data contracts (NL-01 foundation)
# ============================================================================

@dataclass(frozen=True)
class Meta:
    """Acquisition metadata normalized across input formats."""

    pixel_size: tuple[float, float]
    height_unit: str = "nm"
    channel: str = "height"
    scan_direction: str = "forward"
    scan_size: tuple[float, float] | None = None
    scan_rate: float | None = None
    lines: int | None = None
    samples_per_line: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    """One normalized height frame."""

    data: Any
    meta: Meta
    frame_index: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        """Validate frame data."""
        if self.meta is None:
            object.__setattr__(self, 'meta', None)
            raise ValueError("Frame must have metadata")
        
        # Convert list to numpy array
        if isinstance(self.data, list):
            object.__setattr__(self, 'data', np.asarray(self.data, dtype=np.float64))
        
        # Validate 2D data
        if hasattr(self.data, 'ndim') and self.data.ndim != 2:
            raise ValueError(f"Frame data must be 2D, got {self.data.ndim}D")
    
    @property
    def shape(self) -> tuple[int, int]:
        """Return frame shape."""
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Return frame dtype."""
        return self.data.dtype


@dataclass(frozen=True)
class Localizations:
    """Particle/localization coordinates in image coordinates."""

    xy: list[tuple[float, float]]
    frame_index: list[int]
    intensities: list[float] | None = None
    sigmas: list[tuple[float, float]] | None = None
    score: list[float] | None = None

    def __post_init__(self):
        """Validate localization data."""
        if len(self.xy) != len(self.frame_index):
            raise ValueError("xy and frame_index must have same length")
    
    def __len__(self) -> int:
        """Return number of localizations."""
        return len(self.xy)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array with columns [frame, x, y]."""
        arr = np.zeros((len(self), 3), dtype=np.float64)
        arr[:, 0] = self.frame_index
        arr[:, 1:] = self.xy
        return arr


@dataclass(frozen=True)
class ParticleStack:
    """Extracted particle substacks with shape (particles, time, height, width)."""

    data: Any
    centers_xy: list[tuple[float, float]]
    frame_index: list[int]
    box_size: int | None = None

    def __post_init__(self):
        """Validate particle stack data."""
        if len(self.centers_xy) != len(self.frame_index):
            raise ValueError("centers_xy and frame_index must have same length")
        
        # Validate data dimensions (3D or 4D)
        if hasattr(self.data, 'ndim'):
            if self.data.ndim not in (3, 4):
                raise ValueError(f"ParticleStack data must be 3D or 4D, got {self.data.ndim}D")
    
    @property
    def n_particles(self) -> int:
        """Return number of unique particles (not total detections)."""
        if len(self.centers_xy) == 0:
            return 0
        # Count unique particle centers
        unique_centers = set(tuple(c) for c in self.centers_xy)
        return len(unique_centers)


# ============================================================================
# Detection contracts
# ============================================================================

@dataclass
class DetectionParams:
    """Parameters for particle detection."""
    
    threshold: float = 0.5
    min_distance: float = 3.0
    method: DetectionMethod = DetectionMethod.THRESHOLD
    exclude_edges: bool = True
    edge_margin: int = 5
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Typed result from particle detection.

    Coordinates use the image convention ``(x, y)``. ``mask`` is the boolean
    candidate mask produced by the detector, while ``statistics`` contains
    per-detection arrays such as area, volume, and eccentricity.
    """
    
    coordinates: np.ndarray  # Shape (N, 2), ordered as (x, y)
    intensities: np.ndarray  # Shape (N,)
    scores: np.ndarray  # Shape (N,)
    mask: np.ndarray | None = None  # Shape (height, width)
    statistics: dict[str, np.ndarray] = field(default_factory=dict)
    angle: float | None = None

    def __contains__(self, key: str) -> bool:
        """Support the legacy dictionary membership checks."""
        return key in {"locs", "scores", "angle"}

    def __getitem__(self, key: str) -> Any:
        """Provide a small compatibility view for the former result dict."""
        if key == "locs":
            return np.column_stack((self.coordinates, self.intensities, self.scores))
        if key == "scores":
            return self.scores
        if key == "angle":
            return self.angle
        raise KeyError(key)
    
    def __post_init__(self):
        """Validate detection result."""
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError(f"coordinates must be shape (N, 2), got {self.coordinates.shape}")
        
        n_detections = len(self.coordinates)
        if len(self.intensities) != n_detections:
            raise ValueError(f"intensities length must match coordinates, got {len(self.intensities)} vs {n_detections}")
        if len(self.scores) != n_detections:
            raise ValueError(f"scores length must match coordinates, got {len(self.scores)} vs {n_detections}")
        if self.mask is not None and self.mask.dtype != bool:
            raise ValueError("mask must have boolean dtype")
        for name, values in self.statistics.items():
            if len(values) != n_detections:
                raise ValueError(f"statistics[{name!r}] length must match coordinates")
    
    @property
    def n_detections(self) -> int:
        """Return number of detections."""
        return len(self.coordinates)


# ============================================================================
# Localization contracts
# ============================================================================

@dataclass
class LocalizationParams:
    """Parameters for particle localization."""
    
    model: LocalizationModel = LocalizationModel.GAUSSIAN_2D
    fit_radius: float = 5.0
    max_iterations: int = 100
    tolerance: float = 1e-6
    estimate_background: bool = True
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizedParticle:
    """A single localized particle with fitted parameters."""
    
    x: float
    y: float
    intensity: float
    sigma_x: float | None = None
    sigma_y: float | None = None
    background: float = 0.0
    chi_squared: float = np.nan
    frame: int = 0
    
    @property
    def position(self) -> np.ndarray:
        """Return position as numpy array."""
        return np.array([self.x, self.y])
    
    @property
    def sigma(self) -> np.ndarray | None:
        """Return sigma as numpy array if available."""
        if self.sigma_x is not None and self.sigma_y is not None:
            return np.array([self.sigma_x, self.sigma_y])
        return None


# ============================================================================
# Tracking contracts
# ============================================================================

@dataclass
class TrackParams:
    """Parameters for particle tracking."""
    
    max_displacement: float = 10.0
    gap_closing_max_frames: int = 2
    gap_closing_max_distance: float = 15.0
    memory: int = 2
    penalize_gap_closing: bool = True
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParticleTrack:
    """A tracked particle trajectory."""
    
    track_id: int
    particles: list[LocalizedParticle]
    
    @property
    def duration(self) -> int:
        """Return track duration in frames."""
        return len(self.particles)
    
    @property
    def frames(self) -> list[int]:
        """Return frame indices."""
        return [p.frame for p in self.particles]
    
    @property
    def displacements(self) -> np.ndarray:
        """Calculate frame-to-frame displacements."""
        if len(self.particles) < 2:
            return np.array([])
        
        positions = np.array([[p.x, p.y] for p in self.particles])
        diffs = np.diff(positions, axis=0)
        return np.sqrt(np.sum(diffs**2, axis=1))
    
    @property
    def mean_displacement(self) -> float:
        """Calculate mean displacement."""
        disps = self.displacements
        if len(disps) == 0:
            return 0.0
        return float(np.mean(disps))


# ============================================================================
# GPU configuration and context
# ============================================================================

@dataclass
class GPUConfig:
    """GPU execution configuration."""
    
    device_type: DeviceType = DeviceType.AUTO
    device_id: int | None = None
    memory_limit: int | None = None
    batch_size: int = 64
    
    def resolve_device(self) -> DeviceType:
        """Resolve actual device type based on availability."""
        if self.device_type != DeviceType.AUTO:
            return self.device_type
        
        # Try to detect CUDA via CuPy
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() > 0:
                return DeviceType.CUDA
        except ImportError:
            pass
        except Exception:
            pass
        
        return DeviceType.CPU


@dataclass
class ProcessingContext:
    """Execution context for GPU/CPU operations."""
    
    config: GPUConfig
    xp: Any = None
    device_id: int | None = None
    
    def __post_init__(self):
        """Initialize processing context."""
        resolved_device = self.config.resolve_device()
        
        if resolved_device == DeviceType.CUDA:
            try:
                import cupy as cp
                object.__setattr__(self, 'xp', cp)
                if self.config.device_id is not None:
                    object.__setattr__(self, 'device_id', self.config.device_id)
                    cp.cuda.Device(self.device_id).use()
            except ImportError:
                import numpy as np
                object.__setattr__(self, 'xp', np)
                object.__setattr__(self, 'device_id', None)
        else:
            import numpy as np
            object.__setattr__(self, 'xp', np)
            object.__setattr__(self, 'device_id', None)
    
    def allocate(self, shape: tuple, dtype=np.float64) -> np.ndarray:
        """Allocate array on current device."""
        return self.xp.zeros(shape, dtype=dtype)
    
    def to_device(self, arr: np.ndarray) -> Any:
        """Transfer array to device."""
        return self.xp.asarray(arr)
    
    def to_cpu(self, arr: Any) -> np.ndarray:
        """Transfer array to CPU."""
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)


# ============================================================================
# Metadata contracts
# ============================================================================

@dataclass
class ImageMetadata:
    """Image metadata for analysis pipeline."""
    
    width: int
    height: int
    pixel_size: tuple[float, float] = (1.0, 1.0)
    bit_depth: int = 16
    file_format: FileFormat = FileFormat.UNKNOWN
    acquisition_time: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Pipeline contracts
# ============================================================================

@dataclass
class AnalysisPipeline:
    """Complete analysis pipeline results."""
    
    detections: list[DetectionResult]
    localizations: list[list[LocalizedParticle]]
    tracks: list[ParticleTrack]
    metadata: ImageMetadata
    
    @property
    def total_particles(self) -> int:
        """Return total number of localized particles."""
        return sum(len(frame_locs) for frame_locs in self.localizations)
    
    @property
    def total_tracks(self) -> int:
        """Return total number of tracks."""
        return len(self.tracks)
    
    def to_dataframe(self):
        """Convert pipeline results to pandas DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame containing all localization data.
            
        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe() - install with: pip install pandas")
        
        data = []
        for frame_locs in self.localizations:
            for particle in frame_locs:
                data.append({
                    'frame': particle.frame,
                    'x': particle.x,
                    'y': particle.y,
                    'intensity': particle.intensity,
                    'sigma_x': particle.sigma_x,
                    'sigma_y': particle.sigma_y,
                    'background': particle.background,
                    'chi_squared': particle.chi_squared
                })
        return pd.DataFrame(data)


# ============================================================================
# Protocols for algorithm interfaces
# ============================================================================

@runtime_checkable
class Detector(Protocol):
    """Protocol for detection algorithms."""
    
    params: DetectionParams
    
    def detect(self, image: Image2D) -> DetectionResult:
        """Detect particles in an image."""
        ...


@runtime_checkable
class Localizer(Protocol):
    """Protocol for localization algorithms."""
    
    params: LocalizationParams
    
    def localize(self, image_patch: Image2D, 
                 initial_guess: tuple[float, float]) -> LocalizedParticle:
        """Localize a particle in an image patch."""
        ...


@runtime_checkable
class Tracker(Protocol):
    """Protocol for tracking algorithms."""
    
    params: TrackParams
    
    def track(self, localizations: list[list[LocalizedParticle]], 
              n_frames: int) -> list[ParticleTrack]:
        """Track particles across frames."""
        ...


@runtime_checkable
class FileReader(Protocol):
    """Protocol for file readers."""
    
    supported_formats: list[FileFormat]
    
    def read(self, path: str) -> tuple[np.ndarray, ImageMetadata]:
        """Read file and return data with metadata."""
        ...


@runtime_checkable
class FileWriter(Protocol):
    """Protocol for file writers."""
    
    supported_formats: list[FileFormat]
    
    def write(self, path: str, data: np.ndarray, metadata: ImageMetadata) -> None:
        """Write data to file."""
        ...
