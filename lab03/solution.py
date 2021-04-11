import copy

from testing import TestCase, TaskType

import numpy as np

EPS = 1e-6
IT_LIM = 100


class IterationLimitException(Exception):
    pass


def choose_col(m: np.ndarray):
    n = m.shape[1]
    maxi = -1
    for i in range(1, n):
        if m[-1][i] >= EPS and (maxi == -1 or m[-1][i] > m[-1][maxi]):
            maxi = i
    return None if maxi == -1 else maxi


def choose_row(m: np.ndarray, col: int, pos):
    min_v = 1e9
    res = -1
    b = m[:, 0]
    print('B:', b)
    column = m[:, col]
    for i in range(len(b) - 1):
        if column[i] > EPS > abs(b[i] / column[i] - min_v) and pos[res] > pos[i]:
            res = i
            min_v = b[i] / column[i]
            continue
        if column[i] > EPS and b[i] / column[i] < min_v:
            res = i
            min_v = b[i] / column[i]
    if res == -1:
        return None
    return res


def simplex(tc: TestCase):
    m = copy.deepcopy(tc.matrix)
    print(m)
    m = np.vstack((m, np.append(np.array([0]), copy.deepcopy(tc.f))))
    print(m)
    sorted_pos = []
    for v in tc.matrix:
        for p in tc.pos:
            print('element', p, v[p])
            if abs(1 - v[p]) <= EPS:
                sorted_pos.append(p)
    print('Sorted pos', sorted_pos)
    # todo nahuya?
    for j in range(len(tc.pos)):
        m[-1] -= m[-1][sorted_pos[j]] * m[j]
    print('New matrix\n', m)
    throw = True
    for step in range(IT_LIM):
        # leading column
        col = choose_col(m)
        if col is None:
            throw = False
            break
        # leading row
        row = choose_row(m, col, tc.pos)
        if row is None:
            return None
        # gauss to create 1 and 0 column
        m[row] /= m[row][col]
        for i, v in enumerate(m):
            if i != row and abs(v[col]) >= 1e-6:
                v -= m[row] * v[col]
        sorted_pos[row] = col
        print('Iteration', step, 'row', row, 'col', col, 'new matrix\n', m)

    if throw:
        raise IterationLimitException('Iteration limit exceeded')

    res = np.zeros(m.shape[1] - 1)
    for i, p in enumerate(sorted_pos):
        res[p - 1] = m[i][0]
    return res, sorted_pos, -m[-1][0]


def find_acceptable_solution(tc):
    full_matrix_np = tc.matrix
    cur_matrix = copy.deepcopy(full_matrix_np)
    m = full_matrix_np.shape[0]
    n = full_matrix_np.shape[1]
    f = np.array([0] * (n - 1) + [-1] * m)
    cur_matrix_list = cur_matrix.tolist()
    for i in range(m):
        additional_line = [0.0] * m
        additional_line[i] = 1.0
        cur_matrix_list[i] += additional_line

    tcc = copy.deepcopy(tc)
    tcc.f = f
    tcc.matrix = np.array(cur_matrix_list)
    tcc.pos = tuple([i + n for i in range(m)])
    print(tcc)
    opt, positions, val = simplex(tcc)
    print(opt, positions, val)
    if abs(val) > EPS:
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
        print(np.dot(vec, tc.f))

        tc.pos = pos
        point, val = _solve_max(tc)
        print(vec)
        print(point)
        return point, (-1 if tt == TaskType.MIN else 1) * val, vec
    else:
        res = _solve_max(tc)
        if res is None:
            return None
        opt, val = res
        return opt, (-1 if tt == TaskType.MIN else 1) * val, None
