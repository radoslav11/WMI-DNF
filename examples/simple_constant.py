#!/usr/bin/env python3
"""
Simple example with a single real variable x in [0, 2] and constant weight function.

This example demonstrates the basic usage of the WMI solver with:
- Single real variable x constrained to 0 <= x <= 2
- DNF formula: just x (always true since x is constrained)
- Constant weight function: 1 (i.e., 1*x^0)

Expected result should be close to 2 (the volume of the interval [0,2]).
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
    cntReals = 1  # Single real variable
    cntBools = 0  # No boolean variables

    # Create universe: single real variable in [0, 2]
    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=2)

    # DNF formula: Since we only have one real variable and it's always within bounds,
    # we create a trivial DNF that's always satisfiable
    # Empty clause list means always true
    expr = [[]]  # Single empty clause = always true

    # Weight function: constant 1 (represented as 1*x^0)
    # Monomials format: [coefficient, [powers for each variable]]
    monomials = [[1, [0]]]  # 1 * x_0^0 = 1 (constant)

    # No boolean weights needed since cntBools = 0
    poly_wf = WeightFunction(monomials, np.array([]))

    # Algorithm parameters
    eps = 0.25  # Approximation parameter
    delta = 0.15  # Confidence parameter

    print("=== Simple Constant Weight Example ===")
    print(f"Real variables: {cntReals} (x ∈ [0, 2])")
    print(f"Boolean variables: {cntBools}")
    print(f"DNF formula: always true")
    print(f"Weight function: 1 (constant)")
    print(f"Expected result: ≈ 2.0")
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
    print(f"Error from expected (2.0): {abs(result - 2.0):.6f}")


if __name__ == "__main__":
    main()
