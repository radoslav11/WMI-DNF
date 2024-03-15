import numpy as np
from utils.runLatte import integrate
from utils.polytopeSampling import sample, chebyshev_center


class SimpleWMISolver:
    def __init__(self, clauseList, nbBools, universeReals, weightFunction):
        self.nbBools = nbBools

        self.universeReals = universeReals
        self.nbReals = self.universeReals.nbReals

        self.nbVariables = self.nbBools + self.nbReals
        self.weightFunction = weightFunction

        self.clauseList = clauseList
        self.nbClauses = len(clauseList)

        self.computeClauseWeights()
        self.generateClauseHrep()

    def generateHrep(self, clause):
        lines = []
        for atom in clause:
            if type(atom) == list:
                sign = atom[-1][0][0]
                if sign == "!":
                    continue

                if sign == ">":
                    atom = [(x, -y) for x, y in atom]

                vec = [0] * (1 + self.nbReals)
                vec[0] = atom[-1][1]
                for i, v in atom[:-1]:
                    vec[i - self.nbBools + 1] = v

                lines.append(vec)

                if sign == "=":
                    vec = [-x for x in vec]
                    vec[0] *= -1
                    lines.append(vec)

        universeConstraints = np.zeros(
            (self.universeReals.A.shape[0], self.universeReals.A.shape[1] + 1)
        )
        universeConstraints[:, 1:] = self.universeReals.A
        universeConstraints[:, 0] = self.universeReals.b

        if lines == []:
            return universeConstraints

        return np.append(np.array(lines), universeConstraints, axis=0)

    def generateClauseHrep(self):
        self.hrep = [
            (A[:, 0], A[:, 1:])
            for A in [self.generateHrep(clause) for clause in self.clauseList]
        ]
        self.lastSampled = [chebyshev_center(a, b) for b, a in self.hrep]

    def computeWeightOfClause(self, clause):
        boolLits = np.array(
            [x for x in filter(lambda x: type(x) != list, clause)]
        ).astype(int)
        negWeight = (
            1
            - self.weightFunction.boolWeights[
                (
                    boolLits[boolLits >= self.nbVariables] - self.nbVariables
                ).astype(int)
            ]
        ).prod()
        normWeight = self.weightFunction.boolWeights[
            boolLits[boolLits < self.nbVariables]
        ].prod()

        booleanWeight = negWeight * normWeight
        lraWeight = sum(
            [
                integrate(
                    filter(lambda x: type(x) == list, clause),
                    [one_monomial],
                    self.nbBools,
                    self.universeReals,
                )
                for one_monomial in self.weightFunction.f
            ]
        )

        return booleanWeight * lraWeight

    def computeClauseWeights(self):
        self.clauseWeights = np.array(
            [self.computeWeightOfClause(clause) for clause in self.clauseList]
        )
        self.universeDisjointWeightSum = self.clauseWeights.sum()
        self.clauseProbs = (
            self.clauseWeights / self.universeDisjointWeightSum
        ).astype(float)

    def sampleSolution(self, clause, hrep, idx, epsilon, delta):
        sampledBools = list(
            np.random.uniform(0, 1, size=self.nbBools)
            < self.weightFunction.boolWeights
        )
        for lit in clause:
            if type(lit) == int:
                if lit < self.nbVariables:
                    sampledBools[lit] = True
                else:
                    sampledBools[lit - self.nbVariables] = False

        sampledReals = sample(
            hrep[1],
            hrep[0],
            self.weightFunction,
            self.lastSampled[idx],
            epsilon,
            delta,
            self.universeReals,
        )[:-1]
        self.lastSampled[idx] = sampledReals
        return sampledBools + list(sampledReals)

    def checkClauseSAT(self, sol, clause):
        for lit in clause:
            if type(lit) == int:
                if (lit >= self.nbVariables) and sol[lit - self.nbVariables]:
                    return False
                if (lit < self.nbVariables) and not sol[lit]:
                    return False

            elif (
                type(lit) == list
                and sum([sol[idx] * coef for idx, coef in lit[:-1]])
                > lit[-1][1]
            ):
                return False

        return True

    def simpleCoverage(self, epsilon, delta):
        # If you have a procedure that can sample within some
        # epsilon and delta, you can use that instead of the
        # hit and run sampling.
        SampleEps = epsilon * epsilon / (47 * self.nbClauses)
        SampleDelta = (
            delta
            * (epsilon**2)
            / (2276 * np.log(8 / delta) * self.nbClauses)
        )

        # Note that ~eps = eps, because in this implementation
        # we use Latte which is an exact solver.
        C = 1 + SampleEps
        T = int(
            np.ceil(
                (8 * self.nbClauses * (1 + epsilon) * np.log(8 / delta))
                / ((epsilon**2) - 8 * (C - 1) * self.nbClauses)
            )
        )
        numberSuccesses = 0
        point = None

        for _ in range(T):
            if point is None:
                clauseIdx = np.random.choice(
                    self.nbClauses, p=self.clauseProbs
                )
                point = self.sampleSolution(
                    self.clauseList[clauseIdx],
                    self.hrep[clauseIdx],
                    clauseIdx,
                    SampleEps,
                    SampleDelta,
                )

            checkClauseIdx = np.random.randint(self.nbClauses)
            if self.checkClauseSAT(point, self.clauseList[checkClauseIdx]):
                numberSuccesses += 1
                point = None

        return (
            T
            * self.universeDisjointWeightSum
            / (self.nbClauses * numberSuccesses)
        )
