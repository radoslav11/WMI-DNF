from simple_wmi_solver import SimpleWMISolver
from utils.reals_universe import RealsUniverse
from utils.weight_function import WeightFunction
import numpy as np
import sys, time

if __name__ == "__main__":
    np.random.seed(42)
    with open(sys.argv[1], "r") as f:
        test = eval(f.readline())

        expr = eval(test[0])
        cntReals = int(test[1])
        cntBools = int(test[2])
        cntClauses = int(test[3])
        width = int(test[4])
        avgLraLen = int(test[5])

        uni = RealsUniverse(cntReals)

        cntTerms = np.random.choice([1, 2, 3, 4])
        monomials = []

        C = 2
        for i in range(cntTerms):
            curr = [0] * cntReals

            deg = min(np.random.geometric(0.6), 5)

            currC = 1
            for i in range(deg):
                xi = np.random.choice(cntReals)
                curr[xi] += 1
                currC *= 10

            cnst = 1 + np.random.choice(10)
            currC *= cnst

            C += currC
            if deg >= 2:
                cnst *= -1

            monomials.append([-cnst, curr])

        monomials.append([C, [0] * cntReals])
        poly_wf = WeightFunction(
            monomials, np.random.uniform(0, 1, size=cntBools)
        )

        eps = 0.25
        delta = 0.15

        timestamp_start = time.time()

        task = SimpleWMISolver(expr, cntBools, uni, poly_wf)
        result = task.simpleCoverage(eps, delta)

        timestamp_end = time.time()
        execution_time = timestamp_end - timestamp_start

        print("Report for  " + str(sys.argv[1]) + ": ")
        print()
        print("Eps: " + str(eps))
        print("Delta: " + str(delta))
        print("List form of the test (+ generation parameters): " + str(test))
        print("WF (as list of monomials): " + str(monomials))

        print()
        print("Result: " + str(result))
        print(
            "Execution time: " + str("{0:.2f}".format(execution_time)) + " sec"
        )
