"""Reader for JPK HDF5 image files with the ``.jpk`` extension."""

from nanolocz.formats.h5jpk_reader import read_h5jpk, write_h5jpk


def read_jpk(filepath, channel="topography", frame="all"):
    """Read a JPK HDF5 image using the shared JPK channel extractor."""
    data, metadata = read_h5jpk(filepath, channel=channel, frame=frame)
    metadata["format"] = "JPK"
    return data, metadata


def write_jpk(data, filepath, **kwargs):
    """Write the portable JPK-compatible HDF5 representation."""
    return write_h5jpk(data, filepath, **kwargs)
