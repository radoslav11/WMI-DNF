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

    if appearing.sum() > 1:
        sub_command = [
            "../latte-distro/dest/bin/integrate",
            "polytope" + random_hash + ".hrep.latte",
            "--cone-decompose",
            "--monomials=monomial" + random_hash + ".txt",
            "--valuation=integrate",
        ]
        command_ret = subprocess.check_output(
            sub_command, stderr=open(os.devnull, "w"), cwd="temp"
        ).split()
        latte_ret = abs(float(command_ret[1 + command_ret.index(b"Decimal:")]))
    else:
        latte_ret = 1

    manual_integration = np.float128(weightFunction[0][0])
    for i in range(nbReals):
        if appearing[i + 1]:
            continue
        p = np.float128(weightFunction[0][1][i] + 1)
        manual_integration *= (
            np.float128(
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
