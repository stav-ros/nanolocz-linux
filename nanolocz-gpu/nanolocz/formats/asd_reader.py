"""Reader for portable ASD (RIBM) exports, including trace-only files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _normalise_metadata(raw: dict[str, Any], path: Path, data: np.ndarray) -> dict[str, Any]:
    metadata = dict(raw)
    if "pixel_size" in metadata:
        metadata["pixel_size"] = tuple(float(value) for value in metadata["pixel_size"])
    metadata.update({
        "format": "ASD",
        "source": str(path),
        "shape": data.shape,
        "dtype": str(data.dtype),
        "trace_only": bool(metadata.get("trace_only", data.ndim == 1)),
    })
    return metadata


def read_asd(filepath: str | Path, frame: int | str = "all") -> tuple[np.ndarray, dict[str, Any]]:
    """Read an ASD export stored as a NumPy-compatible container.

    The reader accepts the portable ``npz`` representation used for fixtures
    and interchange: a ``data`` array and optional JSON ``metadata`` member.
    One-dimensional arrays are deliberately preserved as trace-only inputs.
    Vendor-native ASD variants that are not NumPy containers fail explicitly
    with a format error rather than being guessed at.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"ASD file not found: {path}")
    if path.suffix.lower() != ".asd":
        raise ValueError(f"Expected .asd file, got: {path.suffix}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive:
                raise ValueError("ASD container must contain a 'data' array")
            data = np.asarray(archive["data"], dtype=np.float64)
            raw = {}
            if "metadata" in archive:
                value = archive["metadata"].item()
                raw = json.loads(value) if isinstance(value, str) else dict(value)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to decode ASD file {path}: {exc}") from exc
    if data.ndim not in (1, 2, 3):
        raise ValueError(f"ASD data must be a trace, image, or movie; got {data.ndim}D")
    if data.ndim == 3 and frame != "all":
        index = int(frame)
        data = data[index:index + 1]
    return data, _normalise_metadata(raw, path, data)
