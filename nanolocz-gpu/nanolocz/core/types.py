"""Small, explicit data contracts for the port.

This module intentionally contains no numerical logic. The contracts are the seam
between file openers, analysis functions, and future NumPy/CuPy backends.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Meta:
    """Acquisition metadata normalized across input formats."""

    pixel_size: tuple[float, float]
    height_unit: str
    channel: str
    scan_direction: str = "forward"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    """One normalized height frame."""

    data: Any
    meta: Meta


@dataclass(frozen=True)
class Localizations:
    """Particle/localization coordinates in image coordinates."""

    xy: Any
    frame_index: Any
    score: Any | None = None


@dataclass(frozen=True)
class ParticleStack:
    """Extracted particle substacks with shape (particles, time, height, width)."""

    data: Any
    centers_xy: Any
    frame_index: Any
