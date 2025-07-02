#!/usr/bin/env python3
"""
Advanced example requiring LattE integration.

This example demonstrates:
- Two real variables x, y in [0, 3]
- Complex DNF formula with multiple polytopic constraints:
  (x + 2y ≤ 4 ∧ x - y ≥ 0) ∨ (2x + y ≤ 5 ∧ x + y ≥ 2)
- Polynomial weight function: x² + y² + xy + 1
- This requires LattE for computing volumes of complex polytopes

WARNING: This example requires LattE to be installed in latte-distro/ folder.
If LattE is not available, this will fail with FileNotFoundError.
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

    # Create universe: two real variables in [0, 3]
    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=3)

    # Complex DNF formula with multiple constraints per clause
    # Clause 1: x + 2y ≤ 4 AND x - y ≥ 0
    # Clause 2: 2x + y ≤ 5 AND x + y ≥ 2
    # Real variables are indexed starting from cntBools (0 in this case)
    expr = [
        [
            [(0, 1), (1, 2), ("<=", 4)],  # x + 2y ≤ 4
            [(0, 1), (1, -1), (">=", 0)],  # x - y ≥ 0
        ],
        [
            [(0, 2), (1, 1), ("<=", 5)],  # 2x + y ≤ 5
            [(0, 1), (1, 1), (">=", 2)],  # x + y ≥ 2
        ],
    ]

    # Complex polynomial weight function: x² + y² + xy + 1
    # Represented as sum of monomials: 1*x^2*y^0 + 1*x^0*y^2 + 1*x^1*y^1 + 1*x^0*y^0
    monomials = [
        [1, [2, 0]],  # x²
        [1, [0, 2]],  # y²
        [1, [1, 1]],  # xy
        [1, [0, 0]],  # 1 (constant)
    ]

    poly_wf = WeightFunction(monomials, np.array([]))

    # Algorithm parameters
    eps = 0.2
    delta = 0.1

    print("=== Advanced LattE Integration Example ===")
    print(f"Real variables: {cntReals} (x, y ∈ [0, 3])")
    print(f"Boolean variables: {cntBools}")
    print(f"DNF formula: (x + 2y ≤ 4 ∧ x - y ≥ 0) ∨ (2x + y ≤ 5 ∧ x + y ≥ 2)")
    print(f"Weight function: x² + y² + xy + 1")
    print(f"Expected result: Complex integration over constrained polytopes")
    print(f"Parameters: eps={eps}, delta={delta}")
    print()
    print("WARNING: This example requires LattE integration tool!")
    print("Install LattE in latte-distro/ folder or this will fail.")
    print()

    # Check if temp directory exists, create if not
    if not os.path.exists("temp"):
        os.makedirs("temp")
        print("Created temp/ directory for LattE intermediate files.")

    timestamp_start = time.time()

    try:
        # Run WMI solver
        task = SimpleWMISolver(expr, cntBools, uni, poly_wf)
        result = task.simpleCoverage(eps, delta)

        timestamp_end = time.time()
        execution_time = timestamp_end - timestamp_start

        print(f"Result: {result:.6f}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print()
        print("SUCCESS: LattE integration completed!")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print()
        print("LattE integration tool not found. Please install LattE:")
        print("1. Download LattE from https://www.math.ucdavis.edu/~latte/")
        print("2. Install in latte-distro/ folder in repository root")
        print("3. Ensure latte-distro/dest/bin/integrate is executable")
        print()
        print(
            "This example demonstrates the full capability of the WMI solver"
        )
        print(
            "with complex polytopic constraints and polynomial weight"
            " functions."
        )


if __name__ == "__main__":
    main()
