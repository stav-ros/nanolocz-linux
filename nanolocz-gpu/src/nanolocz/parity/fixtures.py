"""Safe, checksum-verified loading of committed NumPy golden fixtures."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np


class FixtureError(ValueError):
    """Raised when a golden fixture is missing, malformed, or tampered with."""


_SHA256 = re.compile(r"^(?P<digest>[0-9a-fA-F]{64})(?:\s+\*?(?P<name>[^\s]+))?\s*$")


def _read_expected_digest(sidecar: Path, fixture: Path) -> str:
    text = sidecar.read_text(encoding="utf-8").strip()
    match = _SHA256.fullmatch(text)
    if match is None:
        raise FixtureError(f"malformed SHA-256 sidecar: {sidecar}")
    name = match.group("name")
    if name is not None and Path(name).name != fixture.name:
        raise FixtureError(f"sidecar filename does not match fixture: {sidecar}")
    return match.group("digest").lower()


def load_npy_fixture(path: str | Path) -> np.ndarray:
    """Load a non-pickle NumPy fixture after verifying its SHA-256 sidecar.

    ``path`` must point to a committed ``.npy`` file with a neighboring
    ``.npy.sha256`` sidecar. The returned array is read-only to prevent a test from
    accidentally mutating the shared oracle.
    """

    fixture = Path(path)
    if fixture.suffix != ".npy":
        raise FixtureError(f"fixture must be an .npy file: {fixture}")
    if not fixture.is_file():
        raise FixtureError(f"fixture not found: {fixture}")
    sidecar = fixture.with_name(fixture.name + ".sha256")
    if not sidecar.is_file():
        raise FixtureError(f"checksum sidecar not found: {sidecar}")

    expected = _read_expected_digest(sidecar, fixture)
    actual = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if actual != expected:
        raise FixtureError(
            f"checksum mismatch for {fixture}: expected {expected}, got {actual}"
        )

    try:
        array = np.load(fixture, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FixtureError(f"could not load NumPy fixture: {fixture}") from exc
    if not isinstance(array, np.ndarray):
        raise FixtureError(f"fixture did not contain an ndarray: {fixture}")
    array.setflags(write=False)
    return array
