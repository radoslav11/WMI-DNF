#!/usr/bin/env python3
"""
Simple example with just boolean variables.

This example demonstrates:
- Two boolean variables a, b
- DNF formula: a OR b
- Weight function: constant 1
- Boolean weights: P(a=true) = 0.6, P(b=true) = 0.8
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
    Run the simple boolean variables example.

    Args:
        eps: Approximation parameter (default: 0.25)
        delta: Confidence parameter (default: 0.15)
        verbose: Whether to print progress information (default: False)

    Returns:
        dict: Results containing:
            - result: The computed WMI result
            - expected: The expected result (0.92)
            - execution_time: Time taken to run the example
            - error: Absolute error from expected result
    """
    np.random.seed(42)

    # Problem setup
    cntReals = 1  # Need one dummy real variable for the solver to work
    cntBools = 2  # Two boolean variables

    # Create minimal universe with dummy real variable
    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=1)

    # DNF formula: a OR b
    # Boolean variables: 0 = a, 1 = b
    expr = [
        [0],  # a = true
        [1],  # b = true
    ]

    # Weight function: constant 1
    # With dummy real variable, use constant monomial
    monomials = [[1, [0]]]  # 1 * x^0 = 1 (constant)

    # Boolean weights: P(a=true) = 0.6, P(b=true) = 0.8
    bool_weights = np.array([0.6, 0.8])

    poly_wf = WeightFunction(monomials, bool_weights)

    if verbose:
        print("=== Simple Boolean Variables Example ===")
        print(f"Real variables: {cntReals} (x ∈ [0, 1], dummy variable)")
        print(f"Boolean variables: {cntBools} (a, b)")
        print(f"Boolean weights: P(a=true) = 0.6, P(b=true) = 0.8")
        print(f"DNF formula: a ∨ b")
        print(f"Weight function: 1 (constant)")
        print(
            f"Expected result: P(a ∨ b) = P(a) + P(b) - P(a ∧ b) = 0.6 + 0.8 -"
            f" 0.6*0.8 = 0.92"
        )
        print(f"Parameters: eps={eps}, delta={delta}")
        print()

    timestamp_start = time.time()

    # Run WMI solver
    task = SimpleWMISolver(expr, cntBools, uni, poly_wf)
    result = task.simpleCoverage(eps, delta)

    timestamp_end = time.time()
    execution_time = timestamp_end - timestamp_start

    expected = 0.92
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
