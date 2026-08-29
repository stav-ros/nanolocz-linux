"""Acceptance tests for NL-12 legacy AFM openers."""

import h5py
import numpy as np
import pytest

from nanolocz.io import open_nanolocz
from nanolocz.formats import read_ibw, read_jpk, read_spm


def test_spm_ascii_fixture_is_normalized(tmp_path):
    path = tmp_path / "scan.spm"
    path.write_text("# pixel_size_x=2.0\n# pixel_size_y=3.0\n1 2\n3 4\n")
    data, metadata = read_spm(path)
    np.testing.assert_array_equal(data, [[1, 2], [3, 4]])
    assert metadata["format"] == "SPM"
    assert metadata["pixel_size"] == (2.0, 3.0)


def test_jpk_hdf5_fixture_reads_through_unified_opener(tmp_path):
    path = tmp_path / "scan.jpk"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/data/topography", data=np.arange(6).reshape(2, 3))
    with open_nanolocz(path) as opened:
        np.testing.assert_array_equal(opened.load_movie(), np.arange(6).reshape(2, 3))
        assert opened.load_metadata()["format"] == "JPK"


def test_ibw_requires_optional_igor_reader(tmp_path):
    path = tmp_path / "scan.ibw"
    path.write_bytes(b"not-an-igor-wave")
    with pytest.raises((ImportError, ValueError)):
        read_ibw(path)


def test_legacy_openers_are_read_only(tmp_path):
    path = tmp_path / "scan.spm"
    path.write_text("1\n")
    with pytest.raises(ValueError, match="read-only"):
        open_nanolocz(path, mode="w")
