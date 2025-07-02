#!/usr/bin/env python3
"""
Example with a single real variable and linear weight function.

This example demonstrates:
- Single real variable x in [0, 2]
- Linear weight function: x (i.e., 1*x^1)
- Expected result should be close to ∫[0,2] x dx = 2

The weight function is x, so we're computing the weighted integral of x over [0,2].
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

    # DNF formula: always true (empty clause)
    expr = [[]]

    # Weight function: x (represented as 1*x^1)
    # Monomials format: [coefficient, [powers for each variable]]
    monomials = [[1, [1]]]  # 1 * x_0^1 = x

    poly_wf = WeightFunction(monomials, np.array([]))

    # Algorithm parameters
    eps = 0.25
    delta = 0.15

    print("=== Linear Weight Function Example ===")
    print(f"Real variables: {cntReals} (x ∈ [0, 2])")
    print(f"Boolean variables: {cntBools}")
    print(f"DNF formula: always true")
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

    print(f"Result: {result:.6f}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Error from expected (2.0): {abs(result - 2.0):.6f}")


if __name__ == "__main__":
    main()
