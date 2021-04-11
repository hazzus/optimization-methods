import copy
import math

from testing import TestCase, TaskType

import numpy as np

EPS = 1e-6
IT_LIM = 100


class IterationLimitException(Exception):
    pass


def choose_col(m):
    n = m.shape[1]
    maxi = -1
    for i in range(1, n):
        if m[-1][i] >= EPS and (maxi == -1 or m[-1][i] > m[-1][maxi]):
            maxi = i
    return None if maxi == -1 else maxi


def choose_row(m, col, basis):
    # from leading column select positive minimal by ratio to first column element
    # it's index is leading row
    min_ratio = math.inf
    res = -1
    b = m[:, 0]
    column = m[:, col]
    for i in range(len(b) - 1):
        if column[i] <= EPS:
            continue
        ratio = b[i] / column[i]
        if abs(ratio - min_ratio) < EPS and basis[res] > basis[i]:
            res = i
            min_ratio = ratio
        elif ratio < min_ratio:
            res = i
            min_ratio = ratio
    if res == -1:
        return None
    return res


def simplex(tc: TestCase):
    m = copy.deepcopy(tc.matrix)
    m = np.vstack((m, np.append(np.array([0]), copy.deepcopy(tc.f))))
    sorted_basis = []
    # make basis ordered by rows
    for v in tc.matrix:
        for p in tc.basis:
            if abs(1 - v[p]) <= EPS:
                sorted_basis.append(p)
    # make 0 at basis positions in functional row
    for j in range(len(tc.basis)):
        m[-1] -= m[-1][sorted_basis[j]] * m[j]
    throw = True
    for step in range(IT_LIM):
        # leading column c_l = argmax(c_i) | c_i >= 0
        col = choose_col(m)
        if col is None:
            # all values in functional row are negative => solution reached
            throw = False
            break
        # leading row
        row = choose_row(m, col, tc.basis)
        if row is None:
            # no positive elements => no optimum
            return None
        # gauss to create 1 on leading position and 0 on rows on same column
        m[row] /= m[row][col]
        for i, v in enumerate(m):
            if i != row and abs(v[col]) >= EPS:
                v -= m[row] * v[col]
        # including new vector into basis
        sorted_basis[row] = col
        # print('Iteration', step, 'row', row, 'col', col, 'new matrix\n', m)

    if throw:
        raise IterationLimitException('Iteration limit exceeded')

    res = np.zeros(m.shape[1] - 1)
    for i, p in enumerate(sorted_basis):
        res[p-1] = m[i][0]
    return res, sorted_basis, -m[-1][0]


def find_first_basis(tc: TestCase):
    tc_expanded = copy.deepcopy(tc)
    m, n = tc.matrix.shape
    tc_expanded.filename += "_expanded"
    tc_expanded.f = np.array([0.] * (n - 1) + [-1] * m)
    tc_expanded.matrix = np.concatenate((tc.matrix, np.eye(m)), axis=1)
    tc_expanded.basis = tuple([i + n for i in range(m)])

    optimal_point, new_basis, val = simplex(tc_expanded)
    if abs(val) > EPS:
        return None
    res = optimal_point.tolist()
    return new_basis, res[:-m]


def _solve_max(tc: TestCase):
    tc.prepare()
    result = simplex(tc)
    if result is None:
        return None
    opt_sol, _, val = result
    return opt_sol, val


def solve(tc: TestCase):
    if tc.type == TaskType.MIN:
        tc.make_min()
    if tc.basis is None:
        sol = find_first_basis(tc)
        if sol is None:
            return None
        basis, vec = sol
        print('Found basis:', basis)
        tc.basis = basis
        point, val = _solve_max(tc)
        return point, (-1 if tc.type == TaskType.MIN else 1) * val, vec
    else:
        res = _solve_max(tc)
        if res is None:
            return None
        point, val = res
        return point, (-1 if tc.type == TaskType.MIN else 1) * val, None
