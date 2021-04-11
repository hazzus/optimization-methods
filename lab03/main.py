import os
import numpy as np

from testing import TestCase, TaskType
from solution import solve

TEST_FOLDER = 'tests'

if __name__ == '__main__':
    np.set_printoptions(linewidth=120)
    for filename in os.listdir(TEST_FOLDER):
        # if filename != 'test2':
        #     continue
        tc = TestCase(os.path.join(TEST_FOLDER, filename))
        print(tc)
        print('Solution: ')

        point, val, _ = solve(tc, TaskType.MIN)
        print("Resulting point:", point)
        print("Function value:", val)
        print("Positions:", tc.pos)
