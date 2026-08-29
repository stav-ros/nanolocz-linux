"""Acceptance tests for NL-13 ASD image and trace-only inputs."""

import json
import numpy as np

from nanolocz.formats import read_asd
from nanolocz.io import open_nanolocz


def write_asd_fixture(path, data, metadata):
    with path.open("wb") as handle:
        np.savez(handle, data=data, metadata=json.dumps(metadata))


def test_asd_image_fixture_round_trip(tmp_path):
    path = tmp_path / "image.asd"
    write_asd_fixture(path, np.arange(12).reshape(3, 4), {"pixel_size": [2.0, 3.0], "channel": "height"})

    data, metadata = read_asd(path)

    np.testing.assert_array_equal(data, np.arange(12).reshape(3, 4))
    assert metadata["format"] == "ASD"
    assert metadata["pixel_size"] == (2.0, 3.0)
    assert metadata["trace_only"] is False


def test_asd_trace_only_fixture_is_preserved(tmp_path):
    path = tmp_path / "trace.asd"
    trace = np.linspace(0, 1, 8)
    write_asd_fixture(path, trace, {"axis_unit": "nm", "trace_only": True})

    with open_nanolocz(path) as opened:
        np.testing.assert_allclose(opened.load_movie(), trace)
        metadata = opened.load_metadata()
    assert metadata["trace_only"] is True
    assert metadata["shape"] == (8,)


def test_asd_metadata_defaults_are_stable(tmp_path):
    path = tmp_path / "minimal.asd"
    write_asd_fixture(path, np.zeros((2, 2)), {})

    _, metadata = read_asd(path)

    assert metadata["format"] == "ASD"
    assert metadata["trace_only"] is False
    assert metadata["source"] == str(path)
