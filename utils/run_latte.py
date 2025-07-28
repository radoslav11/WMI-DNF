import subprocess
import numpy as np
import string, os


def _write_latte_input_file(
    latte_file_path, lraAtoms, nbBools, nbReals, universeReals
):
    lines = []
    max_precision = 0

    for _, atom in enumerate(lraAtoms):
        # Determine max precision needed for scaling
        for i, v in atom[:-1]:
            if isinstance(v, float):
                max_precision = max(max_precision, len(str(v).split(".")[-1]))
        constant = atom[-1][1]
        if isinstance(constant, float):
            max_precision = max(
                max_precision, len(str(constant).split(".")[-1])
            )

    if isinstance(universeReals.lowerBound, float):
        max_precision = max(
            max_precision, len(str(universeReals.lowerBound).split(".")[-1])
        )
    if isinstance(universeReals.upperBound, float):
        max_precision = max(
            max_precision, len(str(universeReals.upperBound).split(".")[-1])
        )

    if max_precision > 9:
        print(
            "Warning: Required precision for LattE coefficients"
            f" ({max_precision}) is greater than 9. This may lead to loss of"
            " precision."
        )

    scaling_factor = 10**max_precision

    for _, atom in enumerate(lraAtoms):
        # Convert constraints to >= form for LattE: ax + by + c >= 0
        operator = atom[-1][0]
        constant = atom[-1][1] * scaling_factor
        vec = [0] * (nbReals + 1)

        if operator in [">=", ">"]:
            # ax + by >= c becomes ax + by - c >= 0
            vec[0] = -constant
            for i, v in atom[:-1]:
                vec[i - nbBools + 1] = v * scaling_factor
        elif operator in ["<=", "<"]:
            # ax + by <= c becomes -ax - by + c >= 0
            vec[0] = constant
            for i, v in atom[:-1]:
                vec[i - nbBools + 1] = -v * scaling_factor
        elif operator == "=":
            # For equality, we'll handle one direction here
            vec[0] = -constant
            for i, v in atom[:-1]:
                vec[i - nbBools + 1] = v * scaling_factor

        lines.append(" ".join([str(int(x)) for x in vec]))


    original_lines = lines[:]

    for i in range(nbReals):
        # Add upper bound constraint: x_i <= upperBound  =>  -x_i + upperBound >= 0
        upper_bound_vec = [0] * (nbReals + 1)
        upper_bound_vec[0] = universeReals.upperBound * scaling_factor
        upper_bound_vec[i + 1] = -1 * scaling_factor
        lines.append(" ".join([str(int(x)) for x in upper_bound_vec]))

        # Add lower bound constraint: x_i >= lowerBound  =>  x_i - lowerBound >= 0
        lower_bound_vec = [0] * (nbReals + 1)
        lower_bound_vec[0] = -universeReals.lowerBound * scaling_factor
        lower_bound_vec[i + 1] = 1 * scaling_factor
        lines.append(" ".join([str(int(x)) for x in lower_bound_vec]))

    # Now that all lines (original constraints + bounds) are added, determine appearing variables
    appearing = np.array([False] * (nbReals + 1))
    appearing[0] = True
    for line in lines:
        appearing |= np.array([x != "0" for x in line.split()])

    appearing_for_latte = appearing.copy()

    # Filter lines based on appearing_for_latte after it's fully determined
    lines = [
        " ".join(list(np.array(line.split())[appearing_for_latte]))
        for line in lines
    ]

    with open(latte_file_path, "w") as f:
        f.write(
            str(len(lines)) + " " + str(int(appearing_for_latte.sum())) + "\n"
        )
        f.write("\n".join(lines))

    return appearing_for_latte, original_lines, scaling_factor, lines


def integrate(lraAtoms_filter, weightFunction, nbBools, universeReals):
    nbReals = universeReals.nbReals
    lraAtoms = list(lraAtoms_filter)  # Convert filter object to list

    random_hash = "".join(
        np.random.choice([c for c in string.ascii_uppercase + string.digits])
        for _ in range(30)
    )

    polytope_path = "temp/polytope" + random_hash + ".hrep.latte"
    monomial_path = "temp/monomial" + random_hash + ".txt"

    # Initialize integration result
    latte_ret = 1.0  # Default to 1.0 (multiplicative identity)
    manual_integration = np.longdouble(weightFunction[0][0])  # Initial coefficient

    # Separate free variables from constrained ones
    # A variable is "free" if it only appears in simple bound constraints (≤ or ≥ with single variable)
    constrained_vars = set()
    simple_bounds = {}  # var_idx -> [lower_bound, upper_bound]
    
    # Initialize bounds for all variables
    for i in range(nbReals):
        simple_bounds[i] = [universeReals.lowerBound, universeReals.upperBound]
    
    # Analyze constraints to identify free vs constrained variables
    has_complex_constraints = False
    
    for atom in lraAtoms:
        # Extract operator and constant (always the last element)
        op, constant = atom[-1]
        
        # Check if this is effectively a single-variable constraint
        # Count variables with non-zero coefficients
        active_vars_in_constraint = []
        for var_idx, coeff in atom[:-1]:
            if coeff != 0:
                active_vars_in_constraint.append((var_idx, coeff))
        
        if len(active_vars_in_constraint) == 1:
            # Single variable constraint
            var_idx, coeff = active_vars_in_constraint[0]
            var_idx = var_idx - nbBools  # Adjust for boolean variables
            
            if var_idx >= 0 and var_idx < nbReals:
                constrained_vars.add(var_idx)
                # Update bounds for this variable
                if op in ["<", "<="]:
                    if coeff > 0:
                        simple_bounds[var_idx][1] = min(simple_bounds[var_idx][1], constant / coeff)
                    else:
                        simple_bounds[var_idx][0] = max(simple_bounds[var_idx][0], constant / coeff)
                elif op in [">", ">="]:
                    if coeff > 0:
                        simple_bounds[var_idx][0] = max(simple_bounds[var_idx][0], constant / coeff)
                    else:
                        simple_bounds[var_idx][1] = min(simple_bounds[var_idx][1], constant / coeff)
        else:
            # Complex constraint involving multiple variables
            has_complex_constraints = True
            for var_idx, coeff in atom[:-1]:
                var_idx = var_idx - nbBools
                if var_idx >= 0 and var_idx < nbReals:
                    constrained_vars.add(var_idx)
    
    # Separate free and constrained variables
    free_vars = [i for i in range(nbReals) if i not in constrained_vars]
    active_vars = [i for i in range(nbReals) if i in constrained_vars]

    # If we have complex constraints, use LattE for integration
    if has_complex_constraints and len(active_vars) > 0:
        # Use the existing _write_latte_input_file function
        appearing_for_latte, original_lines, scaling_factor, lines = _write_latte_input_file(
            polytope_path, lraAtoms, nbBools, nbReals, universeReals
        )

        # Write monomial file
        with open(monomial_path, "w") as f:
            f.write(
                str(
                    [
                        [
                            1,
                            list(
                                np.array(monomial[1])[appearing_for_latte[1:]]
                            ),
                        ]
                        for monomial in weightFunction
                    ]
                )
                + "\n"
            )

        # Call LattE executable
        polytope_path_abs = os.path.abspath(polytope_path)
        monomial_path_abs = os.path.abspath(monomial_path)
        sub_command = [
            os.path.abspath("../latte-distro/dest/bin/integrate"),
            polytope_path_abs,
            "--cone-decompose",
            "--monomials=" + monomial_path_abs,
            "--valuation=integrate",
        ]
        try:
            with open(os.devnull, "w") as devnull:
                command_ret = subprocess.check_output(
                    sub_command, stderr=devnull, cwd="temp"
                ).split()
            latte_ret = abs(
                float(command_ret[1 + command_ret.index(b"Decimal:")])
            )
            sum_active_exponents = sum(
                weightFunction[0][1][i] for i in active_vars if i < len(weightFunction[0][1])
            )
            num_active_vars = len([i for i in active_vars if i < len(appearing_for_latte) - 1 and appearing_for_latte[i + 1]])
            latte_ret = latte_ret / (
                scaling_factor ** (sum_active_exponents + num_active_vars)
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                "LattE integration failed or LattE executable not found,"
                " assume empty volume."
            )
            latte_ret = 0.0
        
        # Clean up temporary files
        os.remove(polytope_path)
        os.remove(monomial_path)
        
    elif len(active_vars) > 0:
        # Manual integration for simple constraints only
        for i in active_vars:
            lower_bound, upper_bound = simple_bounds[i]
            p = np.longdouble(weightFunction[0][1][i] + 1)
            if upper_bound > lower_bound:
                integration_term = (
                    np.longdouble(upper_bound**p - lower_bound**p) / p
                )
                manual_integration *= integration_term
            else:
                manual_integration = 0
                break
    else:
        # No active variables
        latte_ret = 1.0

    # Manual integration for free variables (always happens)
    for i in free_vars:
        p = np.longdouble(weightFunction[0][1][i] + 1)
        manual_integration *= (
            np.longdouble(
                (
                    (universeReals.upperBound**p)
                    - (universeReals.lowerBound**p)
                )
            )
            / p
        )

    print("Computed volume: " + str(latte_ret * manual_integration))

    return latte_ret * manual_integration