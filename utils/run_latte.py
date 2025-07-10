import subprocess
import numpy as np
import string, os
import subprocess
import numpy as np
import string, os


def integrate(lraAtoms, weightFunction, nbBools, universeReals):
    nbReals = universeReals.nbReals

    random_hash = "".join(
        np.random.choice([c for c in string.ascii_uppercase + string.digits])
        for _ in range(30)
    )

    with open("temp/polytope" + random_hash + ".hrep.latte", "w") as f:
        lines = []
        for _, atom in enumerate(lraAtoms):
            # Convert constraints to >= form for LattE: ax + by + c >= 0
            operator = atom[-1][0]
            constant = atom[-1][1]
            vec = [0] * (nbReals + 1)

            if operator in [">=", ">"]:
                # ax + by >= c becomes ax + by - c >= 0
                vec[0] = -constant
                for i, v in atom[:-1]:
                    vec[i - nbBools + 1] = v
            elif operator in ["<=", "<"]:
                # ax + by <= c becomes -ax - by + c >= 0
                vec[0] = constant
                for i, v in atom[:-1]:
                    vec[i - nbBools + 1] = -v
            elif operator == "=":
                # For equality, we'll handle one direction here
                vec[0] = -constant
                for i, v in atom[:-1]:
                    vec[i - nbBools + 1] = v

            lines.append(" ".join([str(x) for x in vec]))

        appearing = np.array([False] * (nbReals + 1))
        appearing[0] = True
        for line in lines:
            appearing |= np.array([x != "0" for x in line.split()])

        # CORRECTED: Use LattE for ALL variables appearing in constraints, regardless of weight coefficients
        # The weight coefficients only affect the monomial passed to LattE, not constraint handling
        appearing_for_latte = appearing.copy()

        # For reference: active/free variable separation (not used for LattE decision)
        active_vars = np.where(appearing[1:])[
            0
        ]  # Variables that appear in constraints
        free_vars = np.where(~appearing[1:])[0]  # Variables that don't appear

        # Store original constraint lines for bound tightening
        original_lines = lines[:]

        lines = [
            " ".join(list(np.array(line.split())[appearing_for_latte]))
            for line in lines
        ]

        for i in range(nbReals):
            if appearing_for_latte[i + 1]:
                lines.append(
                    " ".join(
                        list(
                            np.array(universeReals.strConstraints[i].split())[
                                appearing_for_latte
                            ]
                        )
                    )
                )
                lines.append(
                    " ".join(
                        list(
                            np.array(
                                universeReals.strConstraints[
                                    i + nbReals
                                ].split()
                            )[appearing_for_latte]
                        )
                    )
                )

        f.write(
            str(len(lines)) + " " + str(int(appearing_for_latte.sum())) + "\n"
        )
        f.write("\n".join(lines))

    with open("temp/monomial" + random_hash + ".txt", "w") as f:
        f.write(
            str(
                [
                    [1, list(np.array(monomial[1])[appearing_for_latte[1:]])]
                    for monomial in weightFunction
                ]
            )
            + "\n"
        )

    # Check if we have complex multi-variable constraints
    # Single-variable constraints like x <= 2 should be handled analytically
    has_complex_constraints = False
    for line in lines:
        var_count = sum(
            1 for i, x in enumerate(line.split()) if i > 0 and x != "0"
        )
        if var_count > 1:
            has_complex_constraints = True
            break

    # Use LattE only if we have variables with non-zero weight and complex constraints
    if appearing_for_latte.sum() > 1 and has_complex_constraints:
        polytope_path = os.path.abspath(
            "temp/polytope" + random_hash + ".hrep.latte"
        )
        monomial_path = os.path.abspath("temp/monomial" + random_hash + ".txt")
        sub_command = [
            os.path.abspath("../latte-distro/dest/bin/integrate"),
            polytope_path,
            "--cone-decompose",
            "--monomials=" + monomial_path,
            "--valuation=integrate",
        ]
        try:
            command_ret = subprocess.check_output(
                sub_command, stderr=open(os.devnull, "w"), cwd="temp"
            ).split()
            latte_ret = abs(
                float(command_ret[1 + command_ret.index(b"Decimal:")])
            )
        except subprocess.CalledProcessError:
            print("LattE integration failed, assume empty volume.")
            latte_ret = 0
    else:
        latte_ret = 1

    manual_integration = np.longdouble(weightFunction[0][0])

    if appearing.sum() > 1 and not has_complex_constraints:
        # We have simple single-variable constraints, tighten bounds accordingly
        for i in active_vars:
            # Variable appears in constraints, compute tightened bounds
            lower_bound = universeReals.lowerBound
            upper_bound = universeReals.upperBound

            # Apply constraints to tighten bounds
            # All constraints are in form: c + a*x + b*y >= 0
            # For single variable constraints, this becomes: c + a*x >= 0, so x >= -c/a (if a > 0) or x <= -c/a (if a < 0)
            for line in original_lines:
                coeffs = [float(x) for x in line.split()]
                if len(coeffs) > i + 1 and coeffs[i + 1] != 0:
                    const = coeffs[0]
                    coeff = coeffs[i + 1]
                    # Constraint is: const + coeff*var >= 0
                    # Rearranging: coeff*var >= -const, so var >= -const/coeff (if coeff > 0) or var <= -const/coeff (if coeff < 0)
                    if coeff > 0:
                        bound = -const / coeff
                        lower_bound = max(lower_bound, bound)
                    else:
                        bound = -const / coeff
                        upper_bound = min(upper_bound, bound)

            # Use tightened bounds for integration
            p = np.longdouble(weightFunction[0][1][i] + 1)
            if upper_bound > lower_bound:
                integration_term = (
                    np.longdouble(upper_bound**p - lower_bound**p) / p
                )
                manual_integration *= integration_term
            else:
                manual_integration = 0  # Infeasible constraint
                break

        # Integrate free variables (those not appearing in constraints)
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
    else:
        # Original behavior: only integrate free variables (those not in constraints)
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

    os.remove("temp/polytope" + random_hash + ".hrep.latte")
    os.remove("temp/monomial" + random_hash + ".txt")
    return latte_ret * manual_integration
