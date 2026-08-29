"""Reader for Igor Pro binary wave (IBW) files."""

from pathlib import Path
import numpy as np


def read_ibw(filepath, frame="all"):
    """Read an IBW file through the optional ``igor2`` dependency.

    ``igor2`` is intentionally optional because it is not required by the
    NumPy reference installation. The returned array is always float64.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"IBW file not found: {path}")
    if path.suffix.lower() != ".ibw":
        raise ValueError(f"Expected .ibw file, got: {path.suffix}")
    try:
        from igor2 import binarywave
    except ImportError as exc:
        raise ImportError("igor2 is required for .ibw support; install with: pip install igor2") from exc
    try:
        wave = binarywave.load(str(path))["wave"]
        data = np.asarray(wave["wData"], dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Unable to decode IBW file {path}: {exc}") from exc
    if data.ndim == 3 and frame != "all":
        data = data[int(frame):int(frame) + 1]
    if data.ndim < 2:
        raise ValueError(f"IBW wave must contain 2-D image data, got {data.ndim}D")
    return data, {"shape": data.shape, "dtype": str(data.dtype), "filepath": str(path), "format": "IBW"}
