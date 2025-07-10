import numpy as np
from utils.run_latte import integrate
from utils.polytope_sampling import sample
from utils.polytope_utils import find_interior_point_active_vars
from tqdm import tqdm


class SimpleWMISolver:
    def __init__(self, clauseList, nbBools, universeReals, weightFunction):
        self.nbBools = nbBools

        self.universeReals = universeReals
        self.nbReals = self.universeReals.nbReals

        self.nbVariables = self.nbBools + self.nbReals
        self.weightFunction = weightFunction

        # Normalize constraints to eliminate <= and < operators
        self.clauseList = self.normalizeConstraints(clauseList)
        self.nbClauses = len(self.clauseList)

        self.computeClauseWeights()
        self.generateClauseHrep()

    def normalizeConstraints(self, clauseList):
        """Keep constraints in original form for direct conversion to Ax <= b"""
        return clauseList

    def generateHrep(self, clause):
        lines = []
        for atom in clause:
            if type(atom) == list:
                operator = atom[-1][0]
                constant = atom[-1][1]
                if operator == "!":
                    continue

                # Build constraint vector for Ax <= b format
                vec = [0] * (1 + self.nbReals)

                if operator in [">=", ">"]:
                    # ax + by >= c becomes -ax - by <= -c
                    vec[0] = -constant  # -c (right-hand side)
                    for i, v in atom[:-1]:
                        vec[i - self.nbBools + 1] = (
                            -v
                        )  # -variable coefficients
                elif operator in ["<=", "<"]:
                    # ax + by <= c stays as ax + by <= c
                    vec[0] = constant  # c (right-hand side)
                    for i, v in atom[:-1]:
                        vec[i - self.nbBools + 1] = v  # variable coefficients
                elif operator == "=":
                    # ax + by = c becomes two constraints
                    # First: ax + by <= c
                    vec[0] = constant
                    for i, v in atom[:-1]:
                        vec[i - self.nbBools + 1] = v
                    lines.append(vec)
                    # Second: -ax - by <= -c
                    vec2 = [0] * (1 + self.nbReals)
                    vec2[0] = -constant
                    for i, v in atom[:-1]:
                        vec2[i - self.nbBools + 1] = -v
                    lines.append(vec2)
                    continue

                lines.append(vec)

        universeConstraints = np.zeros(
            (self.universeReals.A.shape[0], self.universeReals.A.shape[1] + 1)
        )
        # Convert universe constraints from Ax >= b to Ax <= b format: -Ax <= -b
        universeConstraints[:, 1:] = -self.universeReals.A
        universeConstraints[:, 0] = -self.universeReals.b

        if lines == []:
            return universeConstraints

        return np.append(np.array(lines), universeConstraints, axis=0)

    def generateClauseHrep(self):
        # Generate constraints in Ax <= b format for sampling
        self.hrep = [
            (A[:, 0], A[:, 1:])
            for A in [self.generateHrep(clause) for clause in self.clauseList]
        ]
        # Initialize lastSampled with interior points computed using LP for active variables
        self.lastSampled = []
        for clause in self.clauseList:
            lraAtoms = list(filter(lambda x: type(x) == list, clause))
            interior_point = find_interior_point_active_vars(
                lraAtoms, self.nbReals, self.nbBools, self.universeReals
            )
            if interior_point is not None:
                self.lastSampled.append(interior_point)
            else:
                # Fallback to center point if no interior point found
                center_point = (
                    self.universeReals.lowerBound
                    + self.universeReals.upperBound
                ) / 2.0
                self.lastSampled.append(np.full(self.nbReals, center_point))

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
        try:
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
        except FileNotFoundError:
            lraWeight = 0.0 # LattE not found, assume 0 LRA weight for this clause

        return booleanWeight * lraWeight

    def computeClauseWeights(self):
        self.clauseWeights = np.array(
            [
                self.computeWeightOfClause(clause)
                for clause in tqdm(
                    self.clauseList,
                    desc="Computing clause weights",
                    unit="clause",
                )
            ]
        )
        self.universeDisjointWeightSum = self.clauseWeights.sum()
        if self.universeDisjointWeightSum == 0:
            self.clauseProbs = np.zeros(self.nbClauses)
        else:
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

            elif type(lit) == list:
                # Check original constraint format
                operator = lit[-1][0]
                constant = lit[-1][1]
                var_sum = sum([sol[idx] * coef for idx, coef in lit[:-1]])

                if operator in [">=", ">"]:
                    if var_sum < constant:
                        return False
                elif operator in ["<=", "<"]:
                    if var_sum > constant:
                        return False
                elif operator == "=":
                    if abs(var_sum - constant) > 1e-9:
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

        if self.universeDisjointWeightSum == 0:
            return 0.0

        for i in tqdm(range(T), desc="WMI Sampling", unit="samples"):
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
            sat_result = self.checkClauseSAT(
                point, self.clauseList[checkClauseIdx]
            )
            if sat_result:
                numberSuccesses += 1
                point = None

        return (
            T
            * self.universeDisjointWeightSum
            / (self.nbClauses * numberSuccesses)
        )
