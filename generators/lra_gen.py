import numpy as np
from generators.dnf_gen import generateDNF


def generateAtomLRA(mainVar, nbBools, avgLen, universeReals, x):
    nbReals = universeReals.nbReals
    nbVars = nbReals + nbBools
    if mainVar >= nbVars:
        mainVar -= nbVars

    actualLen = min(nbReals, np.random.geometric(1 / avgLen))

    nullify = np.random.choice(nbReals, nbReals - actualLen, replace=False)

    weights = np.random.normal(1, 1, nbReals)
    weights[nullify] = 0
    weights[mainVar - nbBools] = 1

    b = np.random.normal(0, abs(2 * weights.sum()))

    b *= 100
    weights *= 100

    b = np.ceil(b).astype(int)
    weights = np.ceil(weights).astype(int)

    if x.dot(weights) > b:
        weights *= -1
        b *= -1

    ret = [(v + nbBools, w) for v, w in enumerate(weights) if w != 0]
    ret.append(("<=", b))

    return ret


def fixClause(clause, nbBools, avgLRAAtomLength, universeReals):
    # Generate a point that will definitely satisfy the current clause
    x = np.random.uniform(
        universeReals.lowerBound,
        universeReals.upperBound,
        universeReals.nbReals,
    )
    nbVars = nbBools + universeReals.nbReals
    ret = [
        (
            int(varId)
            if (varId < nbBools or (nbVars <= varId < (nbVars + nbBools)))
            else generateAtomLRA(
                varId, nbBools, avgLRAAtomLength, universeReals, x
            )
        )
        for varId in clause
    ]

    def fix(x, atom):
        if (
            sum([x[idx - nbBools] * coef for idx, coef in atom[:-1]])
            > atom[-1][1]
        ):
            atom[:-1] = [(idx, -coef) for idx, coef in atom[:-1]]
            atom[-1] = ("<=", -atom[-1][1])

        return atom

    cntReals = sum([len(x) - 1 if type(x) == list else 0 for x in ret])
    while cntReals > 10:
        for i in range(len(ret)):
            if type(ret[i]) != list:
                continue

            delta = min(max(0, cntReals - 10), len(ret[i]) - 1)
            if len(ret[i]) == delta + 1:
                ret[i] = []
                cntReals -= delta
            else:
                ret[i] = fix(x, ret[i][delta:])
                cntReals -= delta

    return [x for x in ret if (type(x) != list or len(x) > 0)]


def generateLRA(
    nbBools,
    universeReals,
    nbClauses,
    minCWidth,
    maxCWidth,
    avgLRAAtomLength=1,
    R=0,
    Q=1,
    uniformisePrivilegedVars=True,
):
    nbVars = nbBools + universeReals.nbReals
    dnf = generateDNF(
        nbVars, nbClauses, minCWidth, maxCWidth, R, Q, uniformisePrivilegedVars
    )
    return [
        fixClause(clause, nbBools, avgLRAAtomLength, universeReals)
        for clause in dnf
    ]


def printFormula(conjunctions, nbVars):
    for conjunction in conjunctions:
        print("(", end=" ")
        for index, value in enumerate(conjunction):
            if type(value) == int:
                negated = (value / nbVars) >= 1
                if negated:
                    print("~", end=" ")
                print("v_" + str(value % nbVars), end=" ")
                if index < len(conjunction) - 1:
                    print("^", end=" ")
            else:
                print("(", end="")
                flag = True
                for it in value[:-1]:
                    if it[1] < 0:
                        if flag == True:
                            print(" -", end="")
                        else:
                            print(" - ", end="")
                    elif flag == False:
                        print(" + ", end="")
                    flag = False
                    print(
                        "{0:.2f}".format(abs(it[1]))
                        + " * "
                        + "x_"
                        + str(it[0]),
                        end="",
                    )

                print(
                    " " + value[-1][0] + " " + "{0:.2f}".format(value[-1][1]),
                    end="",
                )
                print(") ", end="")
                if index < len(conjunction) - 1:
                    print("^", end=" ")

        print(")OR", end="")
    print()
