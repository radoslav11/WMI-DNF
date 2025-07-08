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
            # Handle constraint conversion (same logic as in SimpleWMISolver)
            sign = atom[-1][0][0]
            if sign == ">":
                atom = [(x, -y) for x, y in atom]
            elif sign == "<":
                # For <= constraints, negate variable coefficients to convert to >= form
                atom = [(x, -y) for x, y in atom[:-1]] + [
                    (atom[-1][0], atom[-1][1])
                ]

            vec = [0] * (nbReals + 1)
            vec[0] = atom[-1][1]
            for i, v in atom[:-1]:
                vec[i - nbBools + 1] = v
            lines.append(" ".join([str(x) for x in vec]))

        appearing = np.array([False] * (nbReals + 1))
        appearing[0] = True
        for line in lines:
            appearing |= np.array([x != "0" for x in line.split()])

        lines = [
            " ".join(list(np.array(line.split())[appearing])) for line in lines
        ]

        for i in range(nbReals):
            if appearing[i + 1]:
                lines.append(
                    " ".join(
                        list(
                            np.array(universeReals.strConstraints[i].split())[
                                appearing
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
                            )[appearing]
                        )
                    )
                )

        f.write(str(len(lines)) + " " + str(int(appearing.sum())) + "\n")
        f.write("\n".join(lines))

    with open("temp/monomial" + random_hash + ".txt", "w") as f:
        f.write(
            str(
                [
                    [1, list(np.array(monomial[1])[appearing[1:]])]
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

    if appearing.sum() > 1 and has_complex_constraints:
        polytope_path = os.path.abspath("temp/polytope" + random_hash + ".hrep.latte")
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
            latte_ret = abs(float(command_ret[1 + command_ret.index(b"Decimal:")]))
        except subprocess.CalledProcessError:
            print("LattE integration failed, assume empty volume.")
            latte_ret = 0
    else:
        latte_ret = 1

    manual_integration = np.longdouble(weightFunction[0][0])

    if appearing.sum() > 1 and not has_complex_constraints:
        # We have simple single-variable constraints, tighten bounds accordingly
        for i in range(nbReals):
            if appearing[i + 1]:
                # Variable appears in constraints, compute tightened bounds
                lower_bound = universeReals.lowerBound
                upper_bound = universeReals.upperBound

                # Apply constraints to tighten bounds
                for line in lines:
                    coeffs = [float(x) for x in line.split()]
                    if len(coeffs) > i + 1 and coeffs[i + 1] != 0:
                        const = coeffs[0]
                        coeff = coeffs[i + 1]
                        # Constraint is: const + coeff*var >= 0, so var >= -const/coeff
                        if coeff > 0:
                            bound = -const / coeff
                            lower_bound = max(lower_bound, bound)
                        else:
                            bound = -const / coeff
                            upper_bound = min(upper_bound, bound)

                # Use tightened bounds for integration
                p = np.longdouble(weightFunction[0][1][i] + 1)
                if upper_bound > lower_bound:
                    manual_integration *= (
                        np.longdouble(upper_bound**p - lower_bound**p) / p
                    )
                else:
                    manual_integration = 0  # Infeasible constraint
                    break
            else:
                # Variable doesn't appear in constraints, use universe bounds
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
        for i in range(nbReals):
            if appearing[i + 1]:
                continue
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
