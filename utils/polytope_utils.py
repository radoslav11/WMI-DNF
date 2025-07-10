import numpy as np
from scipy.optimize import linprog


def separate_active_variables(lraAtoms, nbReals, nbBools):
    """
    Separate variables that appear in constraints from those that don't.
    Based on logic from runLatte.py lines 40-48.

    Args:
        lraAtoms: List of LRA atoms (constraints)
        nbReals: Number of real variables
        nbBools: Number of boolean variables

    Returns:
        appearing: Boolean array indicating which variables (including constant) appear in constraints
        active_vars: Indices of variables that appear in constraints (excluding constant term)
        free_vars: Indices of variables that don't appear in constraints
    """
    # Track which variables appear in constraints (including constant term at index 0)
    appearing = np.array([False] * (nbReals + 1))
    appearing[0] = True  # Constant term always appears

    # Convert constraints to string format like in runLatte.py
    lines = []
    for _, atom in enumerate(lraAtoms):
        operator = atom[-1][0]
        constant = atom[-1][1]
        vec = [0] * (nbReals + 1)

        if operator in [">=", ">"]:
            vec[0] = -constant
            for i, v in atom[:-1]:
                vec[i - nbBools + 1] = v
        elif operator in ["<=", "<"]:
            vec[0] = constant
            for i, v in atom[:-1]:
                vec[i - nbBools + 1] = -v
        elif operator == "=":
            vec[0] = -constant
            for i, v in atom[:-1]:
                vec[i - nbBools + 1] = v

        lines.append(" ".join([str(x) for x in vec]))

    # Check which variables appear in constraints
    for line in lines:
        appearing |= np.array([x != "0" for x in line.split()])

    # Extract variable indices (excluding constant term)
    active_vars = np.where(appearing[1:])[
        0
    ]  # Variables that appear in constraints
    free_vars = np.where(~appearing[1:])[
        0
    ]  # Variables that don't appear in constraints

    return appearing, active_vars, free_vars


def get_variables_for_latte_integration(
    lraAtoms, weightFunction, nbReals, nbBools
):
    """
    Get variables that should be used for LattE integration.

    CORRECTED LOGIC: All variables appearing in constraints should use LattE integration,
    regardless of their weight coefficients. The weight coefficients only affect the
    monomial passed to LattE, not the decision of whether to use LattE for constraint handling.

    Args:
        lraAtoms: List of LRA atoms (constraints)
        weightFunction: Weight function with monomials (used for monomial construction only)
        nbReals: Number of real variables
        nbBools: Number of boolean variables

    Returns:
        appearing_for_latte: Boolean array indicating which variables to use for LattE
        appearing: Boolean array indicating which variables appear in constraints
        active_vars: Indices of variables that appear in constraints
        free_vars: Indices of variables that don't appear in constraints
    """
    appearing, active_vars, free_vars = separate_active_variables(
        lraAtoms, nbReals, nbBools
    )

    # CORRECTED: Use LattE for ALL variables appearing in constraints
    # This ensures proper integration over constrained regions regardless of weight coefficients
    appearing_for_latte = appearing.copy()

    return appearing_for_latte, appearing, active_vars, free_vars


def find_interior_point_active_vars(lraAtoms, nbReals, nbBools, universeReals):
    """
    Find an interior point for a polytope using the Chebyshev center approach.
    Only solves LP for variables that appear in constraints (active variables).
    Other variables (free variables) are set to the center of their universe bounds.

    The Chebyshev center is the center of the largest sphere that fits inside the polytope,
    making it the point that is maximally far from all constraint boundaries. This ensures
    we get a point that is strictly inside the polytope, not on its boundary.

    Args:
        lraAtoms: List of LRA atoms (constraints)
        nbReals: Number of real variables
        nbBools: Number of boolean variables
        universeReals: Universe bounds for real variables

    Returns:
        point: Interior point as numpy array, or None if infeasible
    """
    _, active_vars, _ = separate_active_variables(lraAtoms, nbReals, nbBools)

    if len(active_vars) == 0:
        # No variables in constraints, return center point for all
        center_point = (
            universeReals.lowerBound + universeReals.upperBound
        ) / 2.0
        return np.full(nbReals, center_point)

    # Build constraint matrix A and vector b for active variables only
    A_active = []
    b_active = []

    for _, atom in enumerate(lraAtoms):
        operator = atom[-1][0]
        constant = atom[-1][1]

        # Build constraint row for active variables only
        row = np.zeros(len(active_vars))

        for i, v in atom[:-1]:
            var_idx = i - nbBools  # Convert to 0-based real variable index
            if var_idx in active_vars:
                active_pos = np.where(active_vars == var_idx)[0][0]
                if operator in ["<=", "<"]:
                    row[active_pos] = v  # Ax <= b form
                elif operator in [">=", ">"]:
                    row[active_pos] = -v  # Convert to <= form: -Ax <= -b

        if operator in ["<=", "<"]:
            A_active.append(row)
            b_active.append(constant)
        elif operator in [">=", ">"]:
            A_active.append(row)
            b_active.append(-constant)
        elif operator == "=":
            # For equality, add both directions
            A_active.append(row)
            b_active.append(constant)
            A_active.append(-row)
            b_active.append(-constant)

    # Add universe bounds for active variables
    for i, var_idx in enumerate(active_vars):
        # Variable bounds: lower_bound <= x <= upper_bound
        # Convert to: x <= upper_bound and -x <= -lower_bound
        upper_row = np.zeros(len(active_vars))
        upper_row[i] = 1
        A_active.append(upper_row)
        b_active.append(universeReals.upperBound)

        lower_row = np.zeros(len(active_vars))
        lower_row[i] = -1
        A_active.append(lower_row)
        b_active.append(-universeReals.lowerBound)

    if len(A_active) == 0:
        # No constraints, return upper bounds
        return np.full(nbReals, universeReals.upperBound)

    A_active = np.array(A_active)
    b_active = np.array(b_active)

    # Use Chebyshev center: find center of largest sphere that fits inside polytope
    # Formulation: max r subject to ||A_i||*r + A_i*x <= b_i for all i
    # This becomes: max r subject to A_i*x + ||A_i||*r <= b_i

    # Extend problem: variables are [x_1, x_2, ..., x_n, r]
    n_vars = len(active_vars)
    A_extended = np.zeros((len(A_active), n_vars + 1))
    A_extended[:, :-1] = A_active  # Original A matrix
    A_extended[:, -1] = np.linalg.norm(
        A_active, axis=1
    )  # ||A_i|| for each row

    b_extended = b_active.copy()

    # Add universe bounds for active variables
    for i, var_idx in enumerate(active_vars):
        # Variable bounds: lower_bound <= x <= upper_bound
        # Convert to: x <= upper_bound and -x <= -lower_bound
        upper_row = np.zeros(n_vars + 1)
        upper_row[i] = 1
        A_extended = np.vstack([A_extended, upper_row])
        b_extended = np.append(b_extended, universeReals.upperBound)

        lower_row = np.zeros(n_vars + 1)
        lower_row[i] = -1
        A_extended = np.vstack([A_extended, lower_row])
        b_extended = np.append(b_extended, -universeReals.lowerBound)

    # Objective: maximize r (minimize -r)
    c_extended = np.zeros(n_vars + 1)
    c_extended[-1] = -1  # Maximize r

    # Solve Chebyshev center problem
    result = linprog(
        c_extended, A_ub=A_extended, b_ub=b_extended, method="highs"
    )

    if not result.success:
        return None  # Infeasible

    # Extract the center point (exclude the radius r from result)
    chebyshev_center = result.x[:-1]  # Remove the r variable

    # Construct full point: active variables from Chebyshev center, free variables at center of bounds
    center_point = (universeReals.lowerBound + universeReals.upperBound) / 2.0
    full_point = np.full(nbReals, center_point)
    full_point[active_vars] = chebyshev_center

    return full_point
