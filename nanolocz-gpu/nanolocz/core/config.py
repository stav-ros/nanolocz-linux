"""Canonical configuration for NanoLocz pipeline.

This module defines the single source of truth for pipeline configuration,
used by both CLI and Napari plugin to ensure consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Canonical configuration for the full NanoLocz processing pipeline.
    
    This configuration is used by:
    - CLI commands (nanolocz.cli)
    - Napari plugin (nanolocz.plugins)
    - Batch processing
    
    All fields have sensible defaults for typical AFM data.
    """

    # =========================================================================
    # Preprocessing
    # =========================================================================
    leveling: str = "plane"
    """Leveling method: 'line', 'plane', 'weighted'."""
    
    filter_type: str = "gaussian"
    """Filter type: 'gaussian', 'median', 'uniform'."""
    
    filter_sigma: float = 1.0
    """Filter strength (sigma for Gaussian, size for median/uniform)."""
    
    remove_scars: bool = False
    """Whether to apply scar removal preprocessing."""

    # =========================================================================
    # Detection
    # =========================================================================
    threshold: float = 3.0
    """Detection threshold in standard deviations above background."""
    
    min_distance: int = 5
    """Minimum distance between detected particles (pixels)."""
    
    prominence: float = 0.0
    """Minimum prominence for peak detection (0 = disabled)."""

    # =========================================================================
    # Tracking
    # =========================================================================
    max_displacement: float = 10.0
    """Maximum displacement between frames (pixels)."""
    
    gap_closing: int = 2
    """Number of frames allowed for gap closing (0 = disabled)."""
    
    memory: int = 3
    """Memory for track extension (frames)."""

    # =========================================================================
    # LAFM Reconstruction
    # =========================================================================
    pixel_size: float | None = None
    """Pixel size in nanometers (None = unknown)."""
    
    sigma: float = 0.5
    """Gaussian splatting sigma (pixels)."""
    
    frc: bool = False
    """Whether to compute FRC resolution estimate."""

    # =========================================================================
    # Backend / Hardware
    # =========================================================================
    gpu: bool = False
    """Whether to use GPU acceleration (requires CuPy)."""
    
    precision: str = "float64"
    """Numerical precision: 'float32', 'float64', or 'mixed'."""

    # =========================================================================
    # I/O (optional, not used in CLI)
    # =========================================================================
    input_path: str = ""
    """Input file path (used by GUI)."""
    
    output_path: str = ""
    """Output directory/file path (used by GUI)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create from dictionary, ignoring unknown fields."""
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: Path | str) -> PipelineConfig:
        """Load configuration from JSON file."""
        import json
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, path: Path | str) -> None:
        """Save configuration to JSON file."""
        import json
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.
        
        Returns:
            List of error messages (empty if valid).
        """
        errors = []
        
        # Validate leveling
        valid_leveling = {"line", "plane", "weighted"}
        if self.leveling not in valid_leveling:
            errors.append(f"leveling must be one of {valid_leveling}, got '{self.leveling}'")
        
        # Validate filter
        valid_filters = {"gaussian", "median", "uniform"}
        if self.filter_type not in valid_filters:
            errors.append(f"filter_type must be one of {valid_filters}, got '{self.filter_type}'")
        
        # Validate precision
        valid_precision = {"float32", "float64", "mixed"}
        if self.precision not in valid_precision:
            errors.append(f"precision must be one of {valid_precision}, got '{self.precision}'")
        
        # Validate numeric ranges
        if self.filter_sigma <= 0:
            errors.append(f"filter_sigma must be positive, got {self.filter_sigma}")
        
        if self.threshold <= 0:
            errors.append(f"threshold must be positive, got {self.threshold}")
        
        if self.min_distance < 1:
            errors.append(f"min_distance must be >= 1, got {self.min_distance}")
        
        if self.max_displacement <= 0:
            errors.append(f"max_displacement must be positive, got {self.max_displacement}")
        
        if self.gap_closing < 0:
            errors.append(f"gap_closing must be >= 0, got {self.gap_closing}")
        
        if self.memory < 0:
            errors.append(f"memory must be >= 0, got {self.memory}")
        
        if self.sigma <= 0:
            errors.append(f"sigma must be positive, got {self.sigma}")
        
        if self.pixel_size is not None and self.pixel_size <= 0:
            errors.append(f"pixel_size must be positive, got {self.pixel_size}")
        
        return errors

    def __post_init__(self) -> None:
        """Post-initialization validation (optional)."""
        # Can add automatic coercion or validation here if needed
        pass


# Default configuration instance for reference
DEFAULT_CONFIG = PipelineConfig()
