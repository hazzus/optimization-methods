from enum import Enum
import numpy as np


class PreparationException(Exception):
    pass


class TaskType(Enum):
    MIN = 1
    MAX = 2


class TestCase:
    _store_type = float
    EPS = 1e-6

    def __init__(self, filename: str):
        self.filename = filename
        with open(filename, 'r') as file:
            self.f = np.array(list(map(self._store_type, file.readline().split())))
            n = int(file.readline())
            self.matrix = []
            for i in range(n):
                self.matrix.append(list(map(self._store_type, file.readline().split())))
            pos = file.readline()
            self.pos = None if pos == 'no' else tuple(map(int, pos.split()))
        self.matrix = np.array(self.matrix)

    def __str__(self):
        return 'testcase name: {}\nf: {}\nm: {}\npos: {}\n==='.format(self.filename, self.f, self.matrix, self.pos)

    def prepare(self):
        used = {}
        for p in self.pos:
            j = -1
            for i, v in enumerate(self.matrix):
                if v[p] != 0 and (i not in used):
                    j = i
                    break
            used[j] = -1
            if j == -1:
                raise PreparationException('Can\'t build matrix')
            self.matrix[j] /= self.matrix[j][p]
            for i, v in enumerate(self.matrix):
                if i != j and abs(v[p]) >= self.EPS:
                    v -= self.matrix[j] * v[p]

    def _make_min(self):
        self.f *= -1


