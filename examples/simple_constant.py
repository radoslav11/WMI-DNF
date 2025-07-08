#!/usr/bin/env python3
"""
Simple example with a single real variable x in [0, 10] and constant weight function.

This example demonstrates the basic usage of the WMI solver with:
- Single real variable x with universe [0, 10] and constraint x <= 2
- DNF formula: x <= 2
- Constant weight function: 1 (i.e., 1*x^0)

Expected result should be close to 2 (the volume of the constraint region x <= 2).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_wmi_solver import SimpleWMISolver
from utils.reals_universe import RealsUniverse
from utils.weight_function import WeightFunction
import numpy as np
import time


def run_example(eps=0.25, delta=0.15, verbose=False):
    """
    Run the simple constant weight example.

    Args:
        eps: Approximation parameter (default: 0.25)
        delta: Confidence parameter (default: 0.15)
        verbose: Whether to print progress information (default: False)

    Returns:
        dict: Results containing:
            - result: The computed WMI result
            - expected: The expected result (2.0)
            - execution_time: Time taken to run the example
            - error: Absolute error from expected result
    """
    np.random.seed(42)

    # Problem setup
    cntReals = 1  # Single real variable
    cntBools = 0  # No boolean variables

    # Create universe: single real variable in [0, 10] (slack bounds)
    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=10)

    # DNF formula: x <= 2 (this constraint creates a proper clause)
    # Real variables are indexed starting from cntBools (0 in this case)
    expr = [[[(0, 1), ("<=", 2)]]]  # x <= 2

    # Weight function: constant 1 (represented as 1*x^0)
    # Monomials format: [coefficient, [powers for each variable]]
    monomials = [[1, [0]]]  # 1 * x_0^0 = 1 (constant)

    # No boolean weights needed since cntBools = 0
    poly_wf = WeightFunction(monomials, np.array([]))

    if verbose:
        print("=== Simple Constant Weight Example ===")
        print(f"Real variables: {cntReals} (x ∈ [0, 10])")
        print(f"Boolean variables: {cntBools}")
        print(f"DNF formula: x ≤ 2")
        print(f"Weight function: 1 (constant)")
        print(f"Expected result: ≈ 2.0 (volume of x ≤ 2 region)")
        print(f"Parameters: eps={eps}, delta={delta}")
        print()

    timestamp_start = time.time()

    # Run WMI solver
    task = SimpleWMISolver(expr, cntBools, uni, poly_wf)
    result = task.simpleCoverage(eps, delta)

    timestamp_end = time.time()
    execution_time = timestamp_end - timestamp_start

    expected = 2.0
    error = abs(result - expected)

    if verbose:
        print(f"Result: {result:.6f}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Error from expected ({expected}): {error:.6f}")

    return {
        "result": result,
        "expected": expected,
        "execution_time": execution_time,
        "error": error,
    }


def main():
    """Command-line interface for the example."""
    run_example(verbose=True)


if __name__ == "__main__":
    main()
