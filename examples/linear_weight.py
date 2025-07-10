#!/usr/bin/env python3
"""
Example with a single real variable and linear weight function.

This example demonstrates:
- Single real variable x with universe [0, 10] and constraint x <= 2
- Linear weight function: x (i.e., 1*x^1)
- Expected result should be close to ∫[0,2] x dx = 2

The weight function is x, so we're computing the weighted integral of x over the constraint region x <= 2.
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
    Run the linear weight function example.

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

    # Weight function: x (represented as 1*x^1)
    # Monomials format: [coefficient, [powers for each variable]]
    monomials = [[1, [1]]]  # 1 * x_0^1 = x

    poly_wf = WeightFunction(monomials, np.array([]))

    if verbose:
        print("=== Linear Weight Function Example ===")
        print(f"Real variables: {cntReals} (x ∈ [0, 10])")
        print(f"Boolean variables: {cntBools}")
        print(f"DNF formula: x ≤ 2")
        print(f"Weight function: x")
        print(f"Expected result: ∫[0,2] x dx = 2.0")
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
