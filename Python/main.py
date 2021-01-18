"""
Jacobi's Algorithm implementation for solve Eigen problem
Author: Robert Limas
Year: 2021
"""

from JacobiAlgorithm import eigen
import numpy as np


def main():
    a = np.array([[7, -1, -1],
                  [-1, 5, 1],
                  [-1, 1, 5]]
                 )
    values, vectors = eigen(a, 20)
    print(values)
    print(vectors)
    values, vectors = np.linalg.eig(a)
    print(values)
    print(vectors)


if __name__ == '__main__':
    main()
