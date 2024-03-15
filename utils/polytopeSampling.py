# DISCLAIMER: Part of the code in this file was reimplemented and 
# is not what was orignially used in the paper.

import numpy as np
from scipy.optimize import linprog


def hit_and_run(a, b, x, w, eps):
    # Part of https://github.com/jonls/dikin_walk is used
    """Generate points with Hit-and-run algorithm."""

    if not (a.dot(x[:-1]) <= b).all():
        print(a.dot(x[:-1]) - b)
        raise Exception("Invalid state: {}".format(x))

    d = np.random.normal(size=(a.shape[1] + 1))
    d /= np.linalg.norm(d)

    dist = np.divide(b - a.dot(x[:-1]), a.dot(d[:-1]))
    closest = dist[dist > 0].min()

    # Instead of binary searching, we could solve 
    # the optimization problem.
    low = 0
    high = closest

    # We can estimate this cnt based on the epsilon we have,
    # instead of making redundant iterations.
    cnt = 32
    for _ in range(cnt):
        mid = (low + high) / 2.0
        curr = mid * d + x
        if curr[-1] >= 0 and (w.eval(curr[:-1]) - curr[-1] < 0):
            low = mid
            closest = mid
        else:
            high = mid

    x += d * closest * np.random.uniform()

    return x


def chebyshev_center(a, b):
    norm_vector = np.reshape(np.linalg.norm(a, axis=1), (a.shape[0], 1))
    c = np.zeros(a.shape[1] + 1)
    c[-1] = -1
    a_lp = np.hstack((a, norm_vector))
    res = linprog(c, A_ub=a_lp, b_ub=b, bounds=(None, None))
    if not res.success:
        raise Exception("Unable to find Chebyshev center")

    return res.x[:-1]


# Actual hit and run sampling
def sample_(a, b, w, x0, eps, delta):
    # Hit and run number of iterations heuristic:
    # Originally in the KR 2020 version of the paper, we used a
    # heuristic for the number of iterations based on the eps and
    # delta, but in the journal version, we did an ablation study
    # for various values for the number of hit and run iterations,
    # which showed that even a small number of iterations was enough.
    c = 32 

    x0 = np.append(x0, np.array([np.random.uniform(w.eval(x0))]))
    for _ in range(c):
        x0 = hit_and_run(a, b, x0, w, eps)
    return x0


# Smarter sampling
def sample(a, b, w, x0, eps, delta, reals_universe):
    """
    Similarly to the volume computation, we will extract the "easy" constraints
    and then run hit-and-run only on a subset of the dimensions.
    """

    n = len(a[0])
    m = len(a)

    important = []
    for i in range(n):
        # We assume that the last 2*n constraints are for the bounds
        if np.any(a[: m - 2 * n, i]) or w.has_nonzero_coefficient(i):
            important.append(i)

    important = np.array(important)

    new_wf = w.filter_vars(important)
    new_x0 = np.array(x0)[important]

    new_a = a[:, important]

    important_rows = ~np.all(new_a == 0, axis=1)

    new_a = new_a[important_rows]
    new_b = b[important_rows]

    # Hit and run
    new_sample = sample_(new_a, new_b, new_wf, new_x0, eps, delta)[:-1]
    pos_new_sample = 0

    ret = np.array([0.0] * n)
    for i in range(n):
        if i in important:
            ret[i] = new_sample[pos_new_sample]
            pos_new_sample += 1
        else:
            ret[i] = np.random.uniform(
                reals_universe.lowerBound, reals_universe.upperBound
            )

    ret = np.append(ret, np.array([np.random.uniform(w.eval(ret))]))
    return ret
