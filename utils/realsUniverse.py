import numpy as np


class RealsUniverse:
    def __init__(self, nbReals, lowerBound=0, upperBound=10):
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.nbReals = nbReals

        self.generateRepresentations()

    def generateRepresentations(self):
        self.A = [
            [-int(i == j) for j in range(self.nbReals)]
            for i in range(self.nbReals)
        ] + [
            [int(i == j) for j in range(self.nbReals)]
            for i in range(self.nbReals)
        ]

        self.b = ([self.lowerBound] * (self.nbReals)) + (
            [self.upperBound] * (self.nbReals)
        )

        self.strConstraints = [
            " ".join([str(val) for val in ([curr_b] + row)])
            for row, curr_b in zip(self.A, self.b)
        ]
        self.b = np.array(self.b)
        self.A = np.array(self.A)
