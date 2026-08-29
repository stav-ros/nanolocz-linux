# Minimal BioAFM workflow

## Scope

Added only the small BioAFMviewer-inspired workflow requested: import a PDB, render a coarse AFM-like topography with a standard conical tip, estimate a usable tip radius from an AFM image, and perform a rough tip/translation fit.

## Implementation

- `nanolocz/simafm/pdb.py`
  - `MolecularStructure` typed atom model
  - `load_pdb()` for ATOM/HETATM records
  - Angstrom-to-nanometre coordinate conversion
- `nanolocz/simafm/simulator.py`
  - `TipParameters`
  - `simulate_afm()` conical footprint approximation
  - `estimate_tip_from_afm()` foreground-width estimate
  - `fit_structure_to_afm()` coarse tip and translation search
- `nanolocz/simafm/__init__.py`
  - public exports for the workflow

This is intentionally a visual/rough fitting model, not a calibrated force
interaction model or full BioAFMviewer replacement.

## Validation

- Focused tests: `tests/test_simafm_simple.py` — 4 passed
- Tests cover PDB parsing, AFM rendering, estimated-tip workflow, fitting, and explicit invalid-input failures.
