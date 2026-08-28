"""Centralized numerical comparison policy for CPU and future GPU paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Tolerance:
    """Relative and absolute tolerances for one numerical backend."""

    rtol: float
    atol: float
    name: str


CPU_TOLERANCE = Tolerance(rtol=1e-5, atol=1e-8, name="cpu-float64")
GPU_TOLERANCE = Tolerance(rtol=1e-3, atol=1e-5, name="gpu-float32")


def assert_close(
    actual: Any,
    expected: Any,
    *,
    tolerance: Tolerance = CPU_TOLERANCE,
    label: str = "array",
) -> None:
    """Assert parity with shape checks, NaN equality, and policy-owned tolerance."""

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{label} shape mismatch: {actual_array.shape} != {expected_array.shape}"
        )
    if actual_array.dtype.kind in "biu" and expected_array.dtype.kind in "biu":
        np.testing.assert_array_equal(actual_array, expected_array, err_msg=label)
        return
    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
        equal_nan=True,
        err_msg=f"{label} ({tolerance.name})",
    )
