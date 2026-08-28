from hashlib import sha256

import numpy as np
import pytest

from nanolocz.parity import (
    CPU_TOLERANCE,
    FixtureError,
    assert_close,
    load_npy_fixture,
)


def _write_fixture(tmp_path, array: np.ndarray, sidecar: str | None = None):
    path = tmp_path / "case.npy"
    np.save(path, array, allow_pickle=False)
    digest = sha256(path.read_bytes()).hexdigest()
    (tmp_path / "case.npy.sha256").write_text(
        sidecar if sidecar is not None else f"{digest}  case.npy\n", encoding="utf-8"
    )
    return path


def test_fixture_loader_verifies_checksum_and_returns_read_only_array(tmp_path):
    path = _write_fixture(tmp_path, np.array([[1.0, np.nan], [3.0, 4.0]]))

    loaded = load_npy_fixture(path)

    np.testing.assert_equal(loaded, np.array([[1.0, np.nan], [3.0, 4.0]]))
    assert loaded.flags.writeable is False


def test_fixture_loader_rejects_tampered_bytes(tmp_path):
    path = _write_fixture(tmp_path, np.array([1, 2, 3], dtype=np.int64))
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(FixtureError, match="checksum mismatch"):
        load_npy_fixture(path)


def test_fixture_loader_rejects_bad_sidecar_filename(tmp_path):
    path = _write_fixture(tmp_path, np.array([1.0]), "a" * 64 + "  other.npy\n")

    with pytest.raises(FixtureError, match="filename"):
        load_npy_fixture(path)


def test_fixture_loader_rejects_missing_sidecar(tmp_path):
    path = tmp_path / "missing.npy"
    np.save(path, np.array([1.0]), allow_pickle=False)

    with pytest.raises(FixtureError, match="sidecar"):
        load_npy_fixture(path)


def test_cpu_tolerance_policy_is_centralized():
    assert CPU_TOLERANCE.rtol == 1e-5
    assert CPU_TOLERANCE.atol == 1e-8
    assert_close(np.array([1.0]), np.array([1.0 + 5e-6]), tolerance=CPU_TOLERANCE)


def test_assert_close_requires_matching_shape():
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_close(np.zeros((2,)), np.zeros((1,)))
