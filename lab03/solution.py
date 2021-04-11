import copy

from testing import TestCase, TaskType

import numpy as np

EPS = 1e-6
IT_LIM = 100


class IterationLimitException(Exception):
    pass


def choose_col(m: np.ndarray):
    n = m.shape[1] - 1
    for i in range(1, n + 1):
        if m[-1][i] >= EPS:
            return i
    return None


def choose_row(m: np.ndarray, col: int, pos):
    max_v = 1e9
    res = -1
    b = m[:, 0]
    column = m[:, col]
    for i in range(len(b) - 1):
        if column[i] > EPS > abs(b[i] / column[i] - max_v) and pos[res] > pos[i]:
            res = i
            max_v = b[i] / column[i]
            continue
        if column[i] > EPS and b[i] / column[i] < max_v:
            res = i
            max_v = b[i] / column[i]
    if res == -1:
        return None
    return res


def simplex(tc: TestCase):
    m = copy.deepcopy(tc.matrix)
    m = np.vstack((m, np.append(np.array([0]), copy.deepcopy(tc.f))))
    sorted_pos = []
    for v in tc.matrix:
        for p in tc.pos:
            if v[p] >= 0.5:
                sorted_pos.append(p)
    for j in range(len(tc.pos)):
        m[-1] -= m[-1][sorted_pos[j]] * m[j]
    throw = True
    for step in range(IT_LIM):
        col = choose_col(m)
        if col is None:
            throw = False
            break
        row = choose_row(m, col, tc.pos)
        if row is None:
            return None
        m[row] /= m[row][col]
        for i, v in enumerate(m):
            if i != row and abs(v[col]) >= 1e-6:
                v -= m[row] * v[col]
        sorted_pos[row] = col

    if throw:
        raise IterationLimitException('Iteration limit exceeded')

    res = np.zeros(m.shape[1] - 1)
    for i, p in enumerate(sorted_pos):
        res[p - 1] = m[i][0]
    return res, sorted_pos, -m[-1][0]


def find_acceptable_solution(tc):
    full_matrix_np = np.array(tc.matrix)
    cur_matrix = copy.deepcopy(full_matrix_np)
    m = full_matrix_np.shape[0]
    n = full_matrix_np.shape[1]
    f = np.array([0] * (full_matrix_np.shape[1] - 1) + [-1] * full_matrix_np.shape[0])
    cur_matrix_list = cur_matrix.tolist()
    for i in range(m):
        additional_line = [0.0] * m
        additional_line[i] = 1.0
        cur_matrix_list[i] += additional_line

    tcc = copy.deepcopy(tc)
    tcc.f = f
    tcc.matrix = np.array(cur_matrix_list)
    tcc.pos = tuple([i + n for i in range(m)])
    opt, positions, val = simplex(tcc)
    if abs(val) > 1e-6:
        return None
    res = opt.tolist()
    return positions, res[:-m]


def _solve_max(tc):
    tc.prepare()
    result = simplex(tc)
    if result is None:
        return None
    opt_sol, pos, val = result
    return opt_sol, val


def solve(tc, tt=TaskType.MIN):
    if tt == TaskType.MIN:
        tc._make_min()
    if tc.pos is None:
        sol = find_acceptable_solution(tc)
        if sol is None:
            return None
        pos, vec = sol
        tc.pos = pos
        point, val = _solve_max(tc)
        return point, (-1 if tt == TaskType.MIN else 1) * val, vec
    else:
        res = _solve_max(tc)
        if res is None:
            return None
        opt, val = res
        return opt, (-1 if tt == TaskType.MIN else 1) * val, None
