#!/usr/bin/env python3
"""
Example with four real variables showing active vs free variables.

This example demonstrates:
- Four real variables x, y, z, w in [0, 10]  
- DNF formula with constraints only on x and y: (x ≥ 2 ∧ x ≤ 8 ∧ y ≥ 3 ∧ y ≤ 7)
- Variables z and w are "free" (unconstrained, only universe bounds apply)
- Constant weight function: 1
- Expected result: (8-2) * (7-3) * 10 * 10 = 6 * 4 * 10 * 10 = 2400

Active variables: x, y (constrained to rectangle [2,8] × [3,7])
Free variables: z, w (only universe bounds [0,10])

The Chebyshev center should find:
- For active variables (x,y): center of [2,8] × [3,7] = (5, 5)  
- For free variables (z,w): center of [0,10] × [0,10] = (5, 5)
- Expected interior point: [5, 5, 5, 5]
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_wmi_solver import SimpleWMISolver
from utils.reals_universe import RealsUniverse
from utils.weight_function import WeightFunction
from utils.polytope_utils import (
    find_interior_point_active_vars,
    separate_active_variables,
)
import numpy as np
import time


def run_example(eps=0.2, delta=0.1, verbose=False):
    """
    Run the four variables mixed active/free example.

    Args:
        eps: Approximation parameter (default: 0.2)
        delta: Confidence parameter (default: 0.1)
        verbose: Whether to print progress information (default: False)

    Returns:
        dict: Results containing:
            - result: The computed WMI result (or None if LattE unavailable)
            - expected: The expected result (2400.0)
            - execution_time: Time taken to run the example
            - error: Absolute error from expected result (or None if failed)
            - success: Boolean indicating if integration succeeded
            - active_vars: List of active variable indices
            - free_vars: List of free variable indices
            - interior_point: Computed interior point
    """
    np.random.seed(42)

    # Problem setup
    cntReals = 4  # Four real variables: x, y, z, w
    cntBools = 0  # No boolean variables

    # Create universe: four real variables in [0, 10]
    uni = RealsUniverse(cntReals, lowerBound=0, upperBound=10)

    # DNF formula with constraints only on x (var 0) and y (var 1)
    # z (var 2) and w (var 3) are free variables
    expr = [
        [
            [(0, 1), (1, 0), (2, 0), (3, 0), (">=", 2)],  # x ≥ 2
            [(0, 1), (1, 0), (2, 0), (3, 0), ("<=", 8)],  # x ≤ 8
            [(0, 0), (1, 1), (2, 0), (3, 0), (">=", 3)],  # y ≥ 3
            [(0, 0), (1, 1), (2, 0), (3, 0), ("<=", 7)],  # y ≤ 7
            # Note: z and w have no constraints (free variables)
        ]
    ]

    # Weight function: constant 1 (integrate over volume)
    monomials = [[1, [0, 0, 0, 0]]]  # 1 * x^0 * y^0 * z^0 * w^0 = 1

    poly_wf = WeightFunction(monomials, np.array([]))

    if verbose:
        print("=== Four Variables Mixed Active/Free Example ===")
        print(f"Real variables: {cntReals} (x, y, z, w ∈ [0, 10])")
        print(f"Boolean variables: {cntBools}")
        print(f"DNF formula: x ≥ 2 ∧ x ≤ 8 ∧ y ≥ 3 ∧ y ≤ 7")
        print(f"  (z and w are free variables - unconstrained)")
        print(f"Weight function: 1 (constant)")
        print(f"Expected result: (8-2) * (7-3) * 10 * 10 = 2400")
        print(f"Parameters: eps={eps}, delta={delta}")
        print()

    # Check if temp directory exists, create if not
    if not os.path.exists("temp"):
        os.makedirs("temp")
        if verbose:
            print("Created temp/ directory for LattE intermediate files.")

    # Debug the active/free variable separation
    task = SimpleWMISolver(expr, cntBools, uni, poly_wf)
    lraAtoms = list(filter(lambda x: type(x) == list, task.clauseList[0]))

    appearing, active_vars, free_vars = separate_active_variables(
        lraAtoms, cntReals, cntBools
    )

    interior_point = find_interior_point_active_vars(
        lraAtoms, cntReals, cntBools, uni
    )

    if verbose:
        print("=== Variable Analysis ===")
        print(f"LRA constraints: {len(lraAtoms)}")
        print(
            f"Active variables (constrained): {active_vars}"
        )  # Should be [0, 1] for x, y
        print(
            f"Free variables (unconstrained): {free_vars}"
        )  # Should be [2, 3] for z, w
        print(f"Computed interior point: {interior_point}")
        print(f"Expected interior point: [5, 5, 5, 5]")
        print()

    timestamp_start = time.time()

    try:
        # Run WMI solver
        result = task.simpleCoverage(eps, delta)

        timestamp_end = time.time()
        execution_time = timestamp_end - timestamp_start

        expected = 2400.0
        error = abs(result - expected)

        if verbose:
            print(f"Result: {result:.6f}")
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"Error from expected ({expected}): {error:.6f}")
            print()
            print("SUCCESS: Mixed active/free variable integration completed!")

        return {
            "result": result,
            "expected": expected,
            "execution_time": execution_time,
            "error": error,
            "success": True,
            "active_vars": active_vars.tolist(),
            "free_vars": free_vars.tolist(),
            "interior_point": interior_point.tolist(),
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
            print("This example demonstrates active vs free variable handling")
            print(
                "with the Chebyshev center approach for optimal interior"
                " points."
            )

        return {
            "result": None,
            "expected": 2400.0,
            "execution_time": execution_time,
            "error": None,
            "success": False,
            "active_vars": active_vars.tolist(),
            "free_vars": free_vars.tolist(),
            "interior_point": (
                interior_point.tolist() if interior_point is not None else None
            ),
        }


def main():
    """Command-line interface for the example."""
    run_example(verbose=True)


if __name__ == "__main__":
    main()
