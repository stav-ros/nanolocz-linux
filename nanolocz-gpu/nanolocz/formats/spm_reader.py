"""Reader for simple NanoScope SPM exports.

The parser supports the ASCII matrix exports commonly used for tests and
portable interchange. Binary vendor-specific channels fail with a useful
message instead of silently returning corrupt data.
"""

from pathlib import Path
import re
import numpy as np


def read_spm(filepath, channel="height", frame="all"):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"SPM file not found: {path}")
    if path.suffix.lower() != ".spm":
        raise ValueError(f"Expected .spm file, got: {path.suffix}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata = {"filepath": str(path), "format": "SPM", "channel": channel}
    rows = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            match = re.match(r"#\s*([^=]+)\s*=\s*(.+)", stripped)
            if match:
                key, value = match.groups()
                try:
                    value = float(value)
                except ValueError:
                    pass
                metadata[key.strip()] = value
            continue
        if stripped:
            try:
                rows.append([float(value) for value in stripped.replace(",", " ").split()])
            except ValueError:
                if rows:
                    break
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"No rectangular ASCII image data found in SPM file: {path}")
    data = np.asarray(rows, dtype=np.float64)
    metadata["shape"] = data.shape
    metadata["dtype"] = str(data.dtype)
    if "pixel_size_x" in metadata and "pixel_size_y" in metadata:
        metadata["pixel_size"] = (metadata["pixel_size_x"], metadata["pixel_size_y"])
    return data, metadata
