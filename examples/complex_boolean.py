#!/usr/bin/env python3
"""
Complex boolean-only example with non-trivial result.

This example demonstrates:
- Three boolean variables a, b, c
- DNF formula: (a ∧ b) ∨ (a ∧ c) ∨ (b ∧ c)
- Weight function: constant 1
- Boolean weights: P(a=true) = P(b=true) = P(c=true) = 0.5
- Expected result: 4/8 = 0.5

This computes P((a ∧ b) ∨ (a ∧ c) ∨ (b ∧ c)) where each variable is true with probability 0.5.
By enumeration of all 8 combinations:
- (0,0,0): FALSE  - (0,0,1): FALSE  - (0,1,0): FALSE  - (0,1,1): TRUE
- (1,0,0): FALSE  - (1,0,1): TRUE   - (1,1,0): TRUE   - (1,1,1): TRUE
So 4 out of 8 combinations satisfy the formula: 4 × (1/8) = 1/2 = 0.5
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_wmi_solver import SimpleWMISolver
from utils.reals_universe import RealsUniverse
from utils.weight_function import WeightFunction
import numpy as np
import time


def run_example(eps=0.05, delta=0.05, verbose=False):
    """
    Run the complex boolean variables example.

    Args:
        eps: Approximation parameter (default: 0.05)
        delta: Confidence parameter (default: 0.05)
        verbose: Whether to print progress information (default: False)

    Returns:
        dict: Results containing:
            - result: The computed WMI result
            - expected: The expected result (0.5)
            - execution_time: Time taken to run the example
            - error: Absolute error from expected result
    """
    np.random.seed(42)

    # Problem setup
    cntReals = 1  # Need at least one real variable for the solver to work
    cntBools = 3  # Three boolean variables

    # Create minimal universe: real variable in [0, 1]
    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=1)

    # DNF formula: (a ∧ b) ∨ (a ∧ c) ∨ (b ∧ c)
    # Boolean variables: 0 = a, 1 = b, 2 = c
    expr = [
        [0, 1],  # a ∧ b
        [0, 2],  # a ∧ c
        [1, 2],  # b ∧ c
    ]

    # Weight function: constant 1
    monomials = [[1, [0]]]  # 1 * x^0 = 1 (constant)

    # Boolean weights: P(a=true) = P(b=true) = P(c=true) = 0.5
    bool_weights = np.array([0.5, 0.5, 0.5])

    poly_wf = WeightFunction(monomials, bool_weights)

    if verbose:
        print("=== Complex Boolean Example ===")
        print(f"Real variables: {cntReals} (x ∈ [0, 1], dummy variable)")
        print(f"Boolean variables: {cntBools} (a, b, c)")
        print(f"Boolean weights: P(a=true) = P(b=true) = P(c=true) = 0.5")
        print(f"DNF formula: (a ∧ b) ∨ (a ∧ c) ∨ (b ∧ c)")
        print(f"Weight function: 1 (constant)")
        print(f"Expected result: 4/8 = 0.5")
        print(
            f"Enumeration: Out of 8 boolean combinations, 4 satisfy the"
            f" formula:"
        )
        print(
            f"           (0,1,1), (1,0,1), (1,1,0), (1,1,1) each with"
            f" probability 1/8"
        )
        print(f"           Total: 4 × (1/8) = 1/2 = 0.5")
        print(f"Parameters: eps={eps}, delta={delta}")
        print("Note: Lower eps value for higher accuracy (longer computation)")
        print()

    timestamp_start = time.time()

    # Run WMI solver
    task = SimpleWMISolver(expr, cntBools, uni, poly_wf)
    result = task.simpleCoverage(eps, delta)

    timestamp_end = time.time()
    execution_time = timestamp_end - timestamp_start

    expected = 4.0 / 8.0  # = 0.5
    error = abs(result - expected)

    if verbose:
        print(f"Result: {result:.6f}")
        print(f"Expected: {expected:.6f}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Error from expected: {error:.6f}")

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
