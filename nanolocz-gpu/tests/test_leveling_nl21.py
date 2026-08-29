"""Acceptance tests for NL-21 batched levelling."""

import numpy as np
import pytest

from nanolocz.core.leveling import batch_line_leveling
from nanolocz.gpu.backend import Backend, BackendConfig, BackendContext, PrecisionMode


def test_batch_line_leveling_matches_reference_per_frame():
    movie = np.array([
        [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]],
        [[10.0, 12.0, 14.0], [20.0, 22.0, 24.0]],
    ])
    leveled, offsets = batch_line_leveling(movie)

    assert leveled.dtype == np.float64
    assert leveled.shape == movie.shape
    assert offsets.shape == (2, 2)
    reference_medians = np.median(leveled[:, :1, :], axis=2)
    np.testing.assert_allclose(
        np.median(leveled, axis=2),
        np.broadcast_to(reference_medians, (movie.shape[0], movie.shape[1])),
    )


def test_batch_line_leveling_honors_shared_mask():
    movie = np.array([[[1.0, 100.0, 3.0], [10.0, 20.0, 30.0]]])
    mask = np.array([[True, False, True], [True, True, True]])

    leveled, offsets = batch_line_leveling(movie, mask=mask)

    np.testing.assert_allclose(offsets, [[0.0, 18.0]])
    np.testing.assert_allclose(leveled[0], [[1.0, 100.0, 3.0], [-8.0, 2.0, 12.0]])


def test_batch_line_leveling_uses_backend_precision_policy():
    context = BackendContext(BackendConfig(backend=Backend.CPU, precision=PrecisionMode.REFERENCE))
    leveled, _ = batch_line_leveling(np.ones((2, 3, 4), dtype=np.float32), context=context)

    assert leveled.dtype == np.float64
    # Reference-line preservation keeps a constant movie at its original level.
    assert np.allclose(leveled, 1.0)


def test_batch_line_leveling_rejects_invalid_shapes_and_masks():
    with pytest.raises(ValueError, match="3D"):
        batch_line_leveling(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="mask"):
        batch_line_leveling(np.zeros((2, 4, 4)), mask=np.ones((4, 3), dtype=bool))
