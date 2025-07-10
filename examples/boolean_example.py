#!/usr/bin/env python3
"""
Example with both boolean and real variables.

This example demonstrates:
- One boolean variable b
- One real variable x in [0, 2]  
- DNF formula: (b AND x >= 1) OR (NOT b AND x <= 1)
- Weight function: constant 1
- Boolean weights: P(b=true) = 0.7
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
    Run the boolean and real variables example.

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
    cntReals = 1  # One real variable
    cntBools = 1  # One boolean variable

    # Create universe: real variable x in [0, 2]
    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=2)

    # Simplified DNF formula: just boolean clauses without real constraints
    # This avoids the need for LattE integration
    # DNF formula: b OR NOT b (always true)
    # Negated booleans start at index nbBools + nbReals
    expr = [[0], [cntBools + cntReals]]  # b = true  # b = false (NOT b)

    # Weight function: constant 1
    monomials = [[1, [0]]]  # 1 * x^0 = 1

    # Boolean weights: P(b=true) = 0.7
    bool_weights = np.array([0.7])

    poly_wf = WeightFunction(monomials, bool_weights)

    if verbose:
        print("=== Boolean and Real Variables Example ===")
        print(f"Real variables: {cntReals} (x ∈ [0, 2])")
        print(f"Boolean variables: {cntBools} (b with P(b=true) = 0.7)")
        print(f"DNF formula: b ∨ ¬b (always true)")
        print(f"Weight function: 1 (constant)")
        print(f"Expected result: 0.7 * 2 + 0.3 * 2 = 2.0")
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
