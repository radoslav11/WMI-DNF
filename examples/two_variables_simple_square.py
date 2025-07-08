#!/usr/bin/env python3
"""
Simple square integration example.

This example demonstrates:
- Two real variables x, y in [0, 30]
- DNF formula defining a square region: x ≥ 10 ∧ x ≤ 30 ∧ y ≥ 10 ∧ y ≤ 30
- Weight function: constant 1
- Expected result: (30-10) * (30-10) = 400
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_wmi_solver import SimpleWMISolver
from utils.reals_universe import RealsUniverse
from utils.weight_function import WeightFunction
import numpy as np
import time


def run_example(eps=0.2, delta=0.1, verbose=False):
    """
    Run the simple square integration example.

    Args:
        eps: Approximation parameter (default: 0.2)
        delta: Confidence parameter (default: 0.1)
        verbose: Whether to print progress information (default: False)

    Returns:
        dict: Results containing:
            - result: The computed WMI result (or None if LattE unavailable)
            - expected: The expected result (400.0)
            - execution_time: Time taken to run the example
            - error: Absolute error from expected result (or None if failed)
            - success: Boolean indicating if integration succeeded
    """
    np.random.seed(42)

    # Problem setup
    cntReals = 2  # Two real variables
    cntBools = 0  # No boolean variables

    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=30)

    # DNF formula with explicit constraints defining the square [10,30] x [10,30]
    expr = [
        [
            [(0, 1), (1, 0), (">=", 10)],  # x >= 10
            [(0, 1), (1, 0), ("<=", 30)],  # x <= 30
            [(0, 0), (1, 1), (">=", 10)],  # y >= 10
            [(0, 0), (1, 1), ("<=", 30)],  # y <= 30
        ]
    ]

    # Weight function: constant 1 (integrate over area)
    monomials = [[1, [0, 0]]]  # 1 * x^0 * y^0 = 1

    poly_wf = WeightFunction(monomials, np.array([]))

    if verbose:
        print("=== Simple Square Integration Example ===")
        print(f"Real variables: {cntReals} (x, y ∈ [0, 30])")
        print(f"Boolean variables: {cntBools}")
        print(f"DNF formula: x ≥ 10 ∧ x ≤ 30 ∧ y ≥ 10 ∧ y ≤ 30")
        print(f"Weight function: 1 (constant)")
        print(f"Expected result: (30-10) * (30-10) = 400")
        print(f"Parameters: eps={eps}, delta={delta}")
        print()

    # Check if temp directory exists, create if not
    if not os.path.exists("temp"):
        os.makedirs("temp")
        if verbose:
            print("Created temp/ directory for LattE intermediate files.")

    timestamp_start = time.time()

    try:
        # Run WMI solver
        task = SimpleWMISolver(expr, cntBools, uni, poly_wf)
        result = task.simpleCoverage(eps, delta)

        timestamp_end = time.time()
        execution_time = timestamp_end - timestamp_start

        expected = 400.0
        error = abs(result - expected)

        if verbose:
            print(f"Result: {result:.6f}")
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"Error from expected ({expected}): {error:.6f}")
            print()
            print("SUCCESS: LattE integration completed!")

        return {
            "result": result,
            "expected": expected,
            "execution_time": execution_time,
            "error": error,
            "success": True,
        }

    except FileNotFoundError as e:
        timestamp_end = time.time()
        execution_time = timestamp_end - timestamp_start

        if verbose:
            print(f"ERROR: {e}")
            print()
            print("LattE integration tool not found. Please install LattE:")
            print(
                "1. Download LattE from https://www.math.ucdavis.edu/~latte/"
            )
            print("2. Install in latte-distro/ folder in repository root")
            print("3. Ensure latte-distro/dest/bin/integrate is executable")
            print()
            print(
                "This example demonstrates the full capability of the WMI"
                " solver"
            )
            print(
                "with complex polytopic constraints and polynomial weight"
                " functions."
            )

        return {
            "result": None,
            "expected": 400.0,
            "execution_time": execution_time,
            "error": None,
            "success": False,
        }


def main():
    """Command-line interface for the example."""
    run_example(verbose=True)


if __name__ == "__main__":
    main()
