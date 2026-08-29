"""Small AFM topography simulator and rough structure fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .pdb import MolecularStructure


@dataclass(frozen=True)
class TipParameters:
    """Simple conical AFM tip model."""

    radius_nm: float = 2.0
    cone_angle_deg: float = 20.0

    def __post_init__(self) -> None:
        if self.radius_nm <= 0:
            raise ValueError("tip radius must be positive")
        if not 0 < self.cone_angle_deg < 90:
            raise ValueError("tip cone angle must be between 0 and 90 degrees")


@dataclass(frozen=True)
class FitResult:
    """Best result from the intentionally coarse fitting search."""

    score: float
    tip: TipParameters
    shift_nm: tuple[float, float]
    simulated: np.ndarray


def _coordinate_grid(shape: tuple[int, int], pixel_size_nm: float) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = shape
    y, x = np.indices(shape, dtype=np.float64)
    return (x - (columns - 1) / 2.0) * pixel_size_nm, (y - (rows - 1) / 2.0) * pixel_size_nm


def simulate_afm(
    structure: MolecularStructure,
    shape: tuple[int, int] = (128, 128),
    pixel_size_nm: float = 0.1,
    tip: TipParameters | None = None,
    shift_nm: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Render a molecule as a coarse AFM height image.

    The molecule is centered in the scan. Each atom contributes a rounded
    conical footprint; this is deliberately a fast visual approximation, not
    a calibrated force-interaction model.
    """
    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError("shape must contain two positive dimensions")
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be positive")
    tip = tip or TipParameters()
    grid_x, grid_y = _coordinate_grid(shape, pixel_size_nm)
    coordinates = structure.coordinates_nm - structure.coordinates_nm.mean(axis=0)
    coordinates = coordinates + np.asarray([shift_nm[0], shift_nm[1], 0.0])
    z_base = float(coordinates[:, 2].min())
    image = np.zeros(shape, dtype=np.float64)
    slope = np.tan(np.deg2rad(tip.cone_angle_deg))

    for x_atom, y_atom, z_atom in coordinates:
        height = float(z_atom - z_base + 0.2)
        radial_limit = tip.radius_nm + height / slope
        radius = np.hypot(grid_x - x_atom, grid_y - y_atom)
        contribution = height - np.maximum(radius - tip.radius_nm, 0.0) * slope
        contribution = np.where(radius <= radial_limit, contribution, 0.0)
        image = np.maximum(image, contribution)
    return image


def estimate_tip_from_afm(image: np.ndarray, pixel_size_nm: float = 0.1) -> TipParameters:
    """Estimate a usable tip radius from the width of the AFM foreground.

    This is a practical starting estimate for rough fitting. It intentionally
    avoids claiming to recover a unique physical tip shape from one image.
    """
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("AFM image must be 2D")
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be positive")
    peak = float(np.max(data))
    if peak <= 0:
        return TipParameters(radius_nm=pixel_size_nm)
    foreground = data >= peak * 0.5
    width_pixels = float(np.sqrt(np.count_nonzero(foreground)))
    radius_nm = max(pixel_size_nm, 0.5 * width_pixels * pixel_size_nm)
    return TipParameters(radius_nm=radius_nm)


def _normalized_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    scale = max(float(np.ptp(reference)), np.finfo(np.float64).eps)
    return float(np.sqrt(np.mean((reference - candidate) ** 2)) / scale)


def fit_structure_to_afm(
    structure: MolecularStructure,
    experimental: np.ndarray,
    pixel_size_nm: float = 0.1,
    tip_candidates: Iterable[TipParameters] | None = None,
    shifts_nm: Iterable[tuple[float, float]] | None = None,
) -> FitResult:
    """Perform a coarse tip/translation search and return the best match."""
    reference = np.asarray(experimental, dtype=np.float64)
    if reference.ndim != 2:
        raise ValueError("experimental AFM image must be 2D")
    tips = tuple(tip_candidates or (estimate_tip_from_afm(reference, pixel_size_nm), TipParameters()))
    shifts = tuple(shifts_nm or ((0.0, 0.0),))
    best: tuple[float, TipParameters, tuple[float, float], np.ndarray] | None = None
    for tip in tips:
        for shift in shifts:
            simulated = simulate_afm(structure, reference.shape, pixel_size_nm, tip, shift)
            score = 1.0 - _normalized_error(reference, simulated)
            if best is None or score > best[0]:
                best = (score, tip, shift, simulated)
    assert best is not None
    return FitResult(score=best[0], tip=best[1], shift_nm=best[2], simulated=best[3])
