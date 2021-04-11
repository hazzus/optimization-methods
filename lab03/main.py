import os

from testing import TestCase
from solution import solve

TEST_FOLDER = 'tests'

if __name__ == '__main__':
    for filename in os.listdir(TEST_FOLDER):
        if filename == 'test6':
            continue
        tc = TestCase(os.path.join(TEST_FOLDER, filename))
        print(tc)
        print('Solution: ')

        point, val, start = solve(tc)
        print("Resulting point:", point)
        print("Function value:", val)
        print("Positions:", tc.pos)
        print("Starting point:", start)
