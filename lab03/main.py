import os
import numpy as np

from testing import TestCase, TaskType
from solution import solve

TEST_FOLDER = 'tests'

if __name__ == '__main__':
    np.set_printoptions(linewidth=120)
    for filename in sorted(os.listdir(TEST_FOLDER)):
        # if filename != 'test2':
        #     continue
        tc = TestCase(os.path.join(TEST_FOLDER, filename))
        print(tc)
        print('Solution: ')
        solution = solve(tc)
        if solution is None:
            print('No suitable solution!')
            continue
        point, val, start = solution

        print("Optimal point:", point)
        print("Function value:", val)
        print("Basis:", tc.basis)
        if start is not None:
            print("Starting point:", start)
