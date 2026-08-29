"""Acceptance tests for NL-16 detection, statistics, and masks."""

import numpy as np
import pytest

from nanolocz.core.detection import detect_particles, fast_peaks2d
from nanolocz.core.types import DetectionResult


def image_with_peaks():
    image = np.zeros((32, 32), dtype=np.float64)
    image[10, 8] = 10.0
    image[20, 23] = 7.0
    image[25, 14] = 5.0
    return image


def test_fast_peaks2d_returns_xy_height_and_prominence_deterministically():
    image = image_with_peaks()
    expected = np.array([[8, 10], [23, 20], [14, 25]])

    first = fast_peaks2d(image, thresh=1.0, kernel_size=3)
    second = fast_peaks2d(image, thresh=1.0, kernel_size=3)

    np.testing.assert_array_equal(first[:, :2], expected)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 4)
    np.testing.assert_allclose(first[:, 2], [10, 7, 5])
    assert np.all(first[:, 3] >= 0)


def test_fast_peaks2d_prominence_filters_weak_peak():
    image = image_with_peaks()
    peaks = fast_peaks2d(image, thresh=1.0, kernel_size=3, min_prom=8.0)

    np.testing.assert_array_equal(peaks[:, :2], [[8, 10]])
    assert peaks[0, 3] >= 8.0


def test_detection_returns_typed_result_with_statistics_and_mask():
    result = detect_particles(
        image_with_peaks(),
        method="direct",
        thresh=1.0,
        kernel_size=3,
    )

    assert isinstance(result, DetectionResult)
    assert result.coordinates.shape == (3, 2)
    np.testing.assert_array_equal(result.coordinates, [[8, 10], [23, 20], [14, 25]])
    np.testing.assert_allclose(result.intensities, [10, 7, 5])
    assert result.mask.shape == (32, 32)
    assert result.mask.dtype == bool
    assert result.statistics["volume"].shape == (3,)
    assert result.statistics["eccentricity"].shape == (3,)
    assert result.statistics["area"].shape == (3,)


def test_detection_respects_input_mask():
    image = image_with_peaks()
    mask = np.zeros_like(image, dtype=bool)
    mask[5:16, 3:14] = True

    result = detect_particles(image, thresh=1.0, mask=mask)

    np.testing.assert_array_equal(result.coordinates, [[8, 10]])
    assert not result.mask[20, 23]
    assert np.all(result.mask <= mask)


def test_detection_applies_min_distance_and_orders_ties():
    image = np.zeros((24, 24), dtype=np.float64)
    image[10, 10] = 9
    image[10, 12] = 8
    image[18, 18] = 7

    result = detect_particles(image, thresh=1, kernel_size=1, min_distance=3)

    np.testing.assert_array_equal(result.coordinates, [[10, 10], [18, 18]])


def test_empty_and_invalid_inputs_are_explicit():
    empty = detect_particles(np.zeros((20, 20)), thresh=1)
    assert isinstance(empty, DetectionResult)
    assert empty.coordinates.shape == (0, 2)
    assert empty.intensities.shape == (0,)
    assert empty.mask.shape == (20, 20)
    assert all(values.size == 0 for values in empty.statistics.values())

    with pytest.raises(ValueError, match="2D"):
        fast_peaks2d(np.zeros((2, 3, 4)), thresh=1, kernel_size=3)
    with pytest.raises(ValueError, match="mask"):
        detect_particles(np.zeros((10, 10)), thresh=1, mask=np.ones((5, 5), bool))


def test_gpu_request_keeps_cpu_reference_result_when_unavailable():
    image = image_with_peaks()
    cpu = fast_peaks2d(image, thresh=1, kernel_size=3, use_gpu=False)
    fallback = fast_peaks2d(image, thresh=1, kernel_size=3, use_gpu=True)
    np.testing.assert_allclose(cpu, fallback)


def test_crosscorr_returns_typed_result():
    image = np.zeros((40, 40), dtype=np.float64)
    reference = np.zeros((5, 5), dtype=np.float64)
    reference[2, 2] = 1
    image[20:25, 15:20] = reference

    result = detect_particles(image, method="crosscorr", ref_img=reference, thresh=1)

    assert isinstance(result, DetectionResult)
    assert result.angle == 0
    assert result.scores.shape == (result.n_detections,)


def test_invalid_method_and_reference_are_rejected():
    image = np.zeros((10, 10))
    with pytest.raises(ValueError, match="Unknown method"):
        detect_particles(image, method="invalid_method")
    with pytest.raises(ValueError, match="ref_img"):
        detect_particles(image, method="crosscorr")
