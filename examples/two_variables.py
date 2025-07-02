#!/usr/bin/env python3
"""
Example with two real variables and polynomial weight function.

This example demonstrates:
- Two real variables x, y in [0, 1]
- DNF formula with constraints: (x + y <= 1.5)
- Polynomial weight function: 1 + x + y
- Expected result: integration over the constrained region
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
    cntReals = 2  # Two real variables
    cntBools = 0  # No boolean variables

    # Create universe: two real variables in [0, 1]
    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=1)

    # DNF formula: always true (no constraints beyond universe bounds)
    # Since we want to integrate over the full unit square [0,1] x [0,1]
    expr = [[]]  # Single empty clause = always true

    # Weight function: 1 + x + y
    # Represented as sum of monomials: 1*x^0*y^0 + 1*x^1*y^0 + 1*x^0*y^1
    monomials = [
        [1, [0, 0]],  # 1 * x^0 * y^0 = 1
        [1, [1, 0]],  # 1 * x^1 * y^0 = x
        [1, [0, 1]],  # 1 * x^0 * y^1 = y
    ]

    poly_wf = WeightFunction(monomials, np.array([]))

    # Algorithm parameters
    eps = 0.25
    delta = 0.15

    print("=== Two Variables with Polynomial Weight ===")
    print(f"Real variables: {cntReals} (x, y ∈ [0, 1])")
    print(f"Boolean variables: {cntBools}")
    print(f"DNF formula: always true")
    print(f"Weight function: 1 + x + y")
    print(f"Expected result: ∫∫[0,1]² (1 + x + y) dx dy = 2.0")
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
