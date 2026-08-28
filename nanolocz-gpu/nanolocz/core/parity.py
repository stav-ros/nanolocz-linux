"""
Parity testing utilities for validating Python implementations against MATLAB originals.

This module provides tools to ensure numerical equivalence between the Python port
and the original MATLAB NanoLocz implementation.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ParityResult:
    """Result of a parity test between Python and MATLAB implementations."""
    
    test_name: str
    passed: bool
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    shape_match: bool
    dtype_match: bool
    details: Dict[str, Any]
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} {self.test_name}\n"
            f"  Max Absolute Error: {self.max_absolute_error:.2e}\n"
            f"  Max Relative Error: {self.max_relative_error:.2e}\n"
            f"  Mean Absolute Error: {self.mean_absolute_error:.2e}\n"
            f"  Shape Match: {self.shape_match}\n"
            f"  Dtype Match: {self.dtype_match}"
        )


def compare_arrays(
    python_result: np.ndarray,
    matlab_result: np.ndarray,
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-6,
    test_name: str = "Unnamed Test"
) -> ParityResult:
    """
    Compare Python and MATLAB array results for parity.
    
    Parameters
    ----------
    python_result : np.ndarray
        Output from Python implementation
    matlab_result : np.ndarray
        Expected output from MATLAB implementation
    abs_tol : float
        Absolute tolerance for comparison
    rel_tol : float
        Relative tolerance for comparison
    test_name : str
        Name of the test for reporting
        
    Returns
    -------
    ParityResult
        Structured result with pass/fail status and error metrics
    """
    # Check shapes
    shape_match = python_result.shape == matlab_result.shape
    
    # Check dtypes
    dtype_match = python_result.dtype == matlab_result.dtype
    
    # If shapes don't match, we can't compute element-wise errors
    if not shape_match:
        return ParityResult(
            test_name=test_name,
            passed=False,
            max_absolute_error=float('inf'),
            max_relative_error=float('inf'),
            mean_absolute_error=float('inf'),
            shape_match=False,
            dtype_match=dtype_match,
            details={
                'python_shape': python_result.shape,
                'matlab_shape': matlab_result.shape
            }
        )
    
    # Compute absolute error
    abs_error = np.abs(python_result.astype(np.float64) - matlab_result.astype(np.float64))
    max_abs_error = float(np.max(abs_error))
    mean_abs_error = float(np.mean(abs_error))
    
    # Compute relative error (avoid division by zero)
    matlab_nonzero = matlab_result != 0
    if np.any(matlab_nonzero):
        rel_error = abs_error[matlab_nonzero] / np.abs(matlab_result[matlab_nonzero])
        max_rel_error = float(np.max(rel_error))
    else:
        max_rel_error = 0.0 if max_abs_error == 0 else float('inf')
    
    # Determine pass/fail
    passed = (
        shape_match and
        max_abs_error <= abs_tol and
        (max_rel_error <= rel_tol or max_rel_error == 0.0)
    )
    
    return ParityResult(
        test_name=test_name,
        passed=passed,
        max_absolute_error=max_abs_error,
        max_relative_error=max_rel_error,
        mean_absolute_error=mean_abs_error,
        shape_match=shape_match,
        dtype_match=dtype_match,
        details={
            'absolute_tolerance': abs_tol,
            'relative_tolerance': rel_tol,
            'python_dtype': str(python_result.dtype),
            'matlab_dtype': str(matlab_result.dtype),
            'array_size': python_result.size
        }
    )


def run_parity_test(
    python_func,
    matlab_expected: Dict[str, np.ndarray],
    test_inputs: Dict[str, np.ndarray],
    test_name: str,
    input_mapping: Optional[Dict[str, str]] = None,
    output_key: str = 'result',
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-6
) -> ParityResult:
    """
    Run a complete parity test comparing Python function to MATLAB expected output.
    
    Parameters
    ----------
    python_func : callable
        Python function to test
    matlab_expected : dict
        Dictionary containing expected outputs from MATLAB
    test_inputs : dict
        Dictionary of input arrays for the test
    test_name : str
        Name of the test
    input_mapping : dict, optional
        Mapping from Python parameter names to MATLAB input keys
    output_key : str
        Key in matlab_expected containing the expected output
    abs_tol : float
        Absolute tolerance
    rel_tol : float
        Relative tolerance
        
    Returns
    -------
    ParityResult
        Test result
    """
    if input_mapping is None:
        input_mapping = {}
    
    # Call Python function with mapped inputs
    python_kwargs = {}
    for py_param, mat_key in input_mapping.items():
        if mat_key in test_inputs:
            python_kwargs[py_param] = test_inputs[mat_key]
    
    # If no mapping, use inputs directly as positional args
    if not input_mapping:
        python_result = python_func(*test_inputs.values())
    else:
        python_result = python_func(**python_kwargs)
    
    # Get MATLAB expected result
    if output_key not in matlab_expected:
        return ParityResult(
            test_name=test_name,
            passed=False,
            max_absolute_error=float('inf'),
            max_relative_error=float('inf'),
            mean_absolute_error=float('inf'),
            shape_match=False,
            dtype_match=False,
            details={'error': f'Missing expected output key: {output_key}'}
        )
    
    matlab_result = matlab_expected[output_key]
    
    # Compare results
    return compare_arrays(
        python_result,
        matlab_result,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        test_name=test_name
    )


def generate_parity_report(results: list[ParityResult]) -> str:
    """
    Generate a summary report of multiple parity tests.
    
    Parameters
    ----------
    results : list of ParityResult
        Results from multiple parity tests
        
    Returns
    -------
    str
        Formatted report string
    """
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    report = [
        "=" * 60,
        "PARITY TEST REPORT",
        "=" * 60,
        f"Total Tests: {total}",
        f"Passed: {passed} ({100*passed/total:.1f}%)",
        f"Failed: {failed} ({100*failed/total:.1f}%)",
        "=" * 60,
        ""
    ]
    
    # Report failures first
    failures = [r for r in results if not r.passed]
    if failures:
        report.append("FAILURES:")
        report.append("-" * 60)
        for result in failures:
            report.append(str(result))
            report.append("")
    
    # Report successes
    successes = [r for r in results if r.passed]
    if successes:
        report.append("SUCCESSES:")
        report.append("-" * 60)
        for result in successes:
            report.append(f"✅ {result.test_name}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)
