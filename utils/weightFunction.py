import numpy as np


class WeightFunction:
    def __init__(self, monomialsList, boolWeights):
        self.f = monomialsList
        self.boolWeights = boolWeights

    def eval(self, x):
        return sum([coef * np.prod(x**powers) for coef, powers in self.f])

    def has_nonzero_coefficient(self, v):
        for _, powers in self.f:
            if powers[v] != 0:
                return True

        return False

    def filter_vars(self, important):
        monomials = []

        for coef, powers in self.f:
            new_powers = []
            for v in important:
                new_powers.append(powers[v])

            monomials.append([coef, new_powers])

        return WeightFunction(monomials, self.boolWeights)
