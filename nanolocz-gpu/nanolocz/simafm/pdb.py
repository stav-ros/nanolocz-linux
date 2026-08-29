"""Minimal PDB structure import for AFM simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MolecularStructure:
    """Atom coordinates normalized to nanometres."""

    coordinates_nm: np.ndarray
    elements: tuple[str, ...]
    atom_names: tuple[str, ...]
    residue_names: tuple[str, ...]
    chain_ids: tuple[str, ...]
    residue_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates_nm, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates_nm must have shape (N, 3)")
        if coordinates.shape[0] == 0:
            raise ValueError("structure must contain at least one atom")
        object.__setattr__(self, "coordinates_nm", coordinates)
        n_atoms = coordinates.shape[0]
        for name in ("elements", "atom_names", "residue_names", "chain_ids", "residue_ids"):
            if len(getattr(self, name)) != n_atoms:
                raise ValueError(f"{name} must contain one value per atom")

    @property
    def n_atoms(self) -> int:
        return self.coordinates_nm.shape[0]


def _element_from_atom_name(atom_name: str) -> str:
    letters = "".join(character for character in atom_name if character.isalpha())
    return (letters[:2] if letters[:2].upper() in {"CL", "BR"} else letters[:1]).upper() or "C"


def load_pdb(filepath: str | Path) -> MolecularStructure:
    """Load ATOM/HETATM coordinates from a PDB file.

    PDB coordinates are stored in Angstroms and are converted to nanometres.
    Alternate locations other than blank or ``A`` are ignored.
    """
    path = Path(filepath)
    coordinates: list[tuple[float, float, float]] = []
    elements: list[str] = []
    atom_names: list[str] = []
    residue_names: list[str] = []
    chain_ids: list[str] = []
    residue_ids: list[int] = []

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if len(line) < 54:
            raise ValueError(f"PDB coordinate record on line {line_number} is too short")
        altloc = line[16].strip()
        if altloc not in ("", "A"):
            continue
        try:
            xyz_angstrom = tuple(float(line[start:stop]) for start, stop in ((30, 38), (38, 46), (46, 54)))
            residue_id = int(line[22:26].strip())
        except ValueError as exc:
            raise ValueError(f"invalid PDB coordinates on line {line_number}") from exc
        atom_name = line[12:16].strip()
        coordinates.append(tuple(value / 10.0 for value in xyz_angstrom))
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        elements.append(element or _element_from_atom_name(atom_name))
        atom_names.append(atom_name)
        residue_names.append(line[17:20].strip())
        chain_ids.append(line[21].strip())
        residue_ids.append(residue_id)

    if not coordinates:
        raise ValueError(f"PDB file contains no usable atom coordinates: {path}")
    return MolecularStructure(
        coordinates_nm=np.asarray(coordinates, dtype=np.float64),
        elements=tuple(elements),
        atom_names=tuple(atom_names),
        residue_names=tuple(residue_names),
        chain_ids=tuple(chain_ids),
        residue_ids=tuple(residue_ids),
    )
