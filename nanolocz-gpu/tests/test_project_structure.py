from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_project_metadata_is_single_valid_configuration():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)

    assert metadata["project"]["name"] == "nanolocz"
    assert "test" in metadata["project"]["optional-dependencies"]
    assert metadata["tool"]["setuptools"]["packages"]["find"]["where"] == ["."]


def test_canonical_package_contains_foundation_apis():
    from nanolocz.core import Frame, Localizations, Meta, ParticleStack
    from nanolocz.parity import CPU_TOLERANCE, load_npy_fixture

    assert Frame and Localizations and Meta and ParticleStack
    assert CPU_TOLERANCE.name == "cpu-float64"
    assert callable(load_npy_fixture)


def test_obsolete_src_package_is_not_present():
    assert not (ROOT / "src" / "nanolocz").exists()
