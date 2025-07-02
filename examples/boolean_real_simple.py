#!/usr/bin/env python3
"""
Simple example with one boolean and one real variable.

This example demonstrates:
- One boolean variable b
- One real variable x in [0, 1]
- DNF formula: always true (no constraints)
- Weight function: constant 1
- Boolean weights: P(b=true) = 0.7

This integrates over all possible assignments, weighted by boolean probabilities.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SimpleWMISolver import SimpleWMISolver
from utils.realsUniverse import RealsUniverse
from utils.weightFunction import WeightFunction
import numpy as np
import time


def main():
    np.random.seed(42)

    # Problem setup
    cntReals = 1  # One real variable
    cntBools = 1  # One boolean variable

    # Create universe: real variable x in [0, 1]
    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=1)

    # DNF formula: always true (no additional constraints)
    expr = [[]]  # Single empty clause = always true

    # Weight function: constant 1
    monomials = [[1, [0]]]  # 1 * x^0 = 1 (constant)

    # Boolean weights: P(b=true) = 0.7
    bool_weights = np.array([0.7])

    poly_wf = WeightFunction(monomials, bool_weights)

    # Algorithm parameters
    eps = 0.25
    delta = 0.15

    print("=== Boolean + Real Variables Example ===")
    print(f"Real variables: {cntReals} (x ∈ [0, 1])")
    print(f"Boolean variables: {cntBools} (b with P(b=true) = 0.7)")
    print(f"DNF formula: always true")
    print(f"Weight function: 1 (constant)")
    print(f"Expected result: 1.0 (volume of [0,1] with constant weight)")
    print(f"Parameters: eps={eps}, delta={delta}")
    print()

    timestamp_start = time.time()

    # Run WMI solver
    task = SimpleWMISolver(expr, cntBools, uni, poly_wf)
    result = task.simpleCoverage(eps, delta)

    timestamp_end = time.time()
    execution_time = timestamp_end - timestamp_start

    print(f"Result: {result:.6f}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Error from expected (1.0): {abs(result - 1.0):.6f}")


if __name__ == "__main__":
    main()
