"""Focused tests for the minimal BioAFM-style PDB/simulation workflow."""

from pathlib import Path

import numpy as np

from nanolocz.simafm import (
    TipParameters,
    estimate_tip_from_afm,
    fit_structure_to_afm,
    load_pdb,
    simulate_afm,
)


PDB_TEXT = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C
ATOM      2  CB  ALA A   1       5.000   0.000   2.000  1.00 10.00           C
HETATM    3  O   HOH B   2       0.000   5.000   1.000  1.00 10.00           O
END
"""


def test_load_pdb_extracts_atoms_and_converts_angstroms(tmp_path: Path):
    path = tmp_path / "model.pdb"
    path.write_text(PDB_TEXT)

    molecule = load_pdb(path)

    assert molecule.coordinates_nm.shape == (3, 3)
    np.testing.assert_allclose(molecule.coordinates_nm[1], [0.5, 0.0, 0.2])
    assert molecule.elements == ("C", "C", "O")


def test_simulate_standard_tip_returns_nonempty_topography(tmp_path: Path):
    path = tmp_path / "model.pdb"
    path.write_text(PDB_TEXT)
    molecule = load_pdb(path)

    image = simulate_afm(
        molecule,
        shape=(32, 32),
        pixel_size_nm=0.1,
        tip=TipParameters(radius_nm=1.0, cone_angle_deg=20.0),
    )

    assert image.shape == (32, 32)
    assert image.dtype == np.float64
    assert np.isfinite(image).all()
    assert float(image.max()) > 0.0


def test_estimated_tip_and_rough_fit_work_on_synthetic_image(tmp_path: Path):
    path = tmp_path / "model.pdb"
    path.write_text(PDB_TEXT)
    molecule = load_pdb(path)
    true_tip = TipParameters(radius_nm=1.2, cone_angle_deg=20.0)
    experimental = simulate_afm(molecule, shape=(32, 32), pixel_size_nm=0.1, tip=true_tip)

    estimated = estimate_tip_from_afm(experimental, pixel_size_nm=0.1)
    result = fit_structure_to_afm(
        molecule,
        experimental,
        pixel_size_nm=0.1,
        tip_candidates=[true_tip, TipParameters(radius_nm=2.0, cone_angle_deg=20.0)],
    )

    assert estimated.radius_nm >= 0.1
    assert result.score >= 0.99
    assert result.tip == true_tip
    assert result.simulated.shape == experimental.shape


def test_invalid_pdb_and_tip_inputs_are_explicit(tmp_path: Path):
    path = tmp_path / "bad.pdb"
    path.write_text("not a structure\n")

    try:
        load_pdb(path)
    except ValueError as exc:
        assert "coordinates" in str(exc)
    else:
        raise AssertionError("invalid PDB should fail explicitly")

    try:
        TipParameters(radius_nm=0.0)
    except ValueError as exc:
        assert "radius" in str(exc)
    else:
        raise AssertionError("invalid tip radius should fail explicitly")
