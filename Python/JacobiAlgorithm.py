"""
Jacobi's Algorithm implementation for solve Eigen problem
Author: Robert Limas
Year: 2021

Inputs:
    matrix (required -> square matrix), iterations (optional -> default 20 iterations)
Outputs:
    eigen_values (list with the matrix's eigen values), eigen_vectors (matrix square with eigen vectors)

Description:
Jacobi's algorithm is based on the existence of the orthogonal matrices (P), such that
they transform a matrix A into matrix whose main diagonal is formed by the eigenvalues

For example in 3x3 matrix

                |A1 0   0 |
P * A * P^(t) = |0  A2  0 |
                |0  0   A3|

Where:
    P is the orthogonal matrix
    A is a symmetric matrix which we want to found its eigen values
    (t) indicates transpose
    Ax correspond to the eigen value in x position

The result of this operation contain the new values of A.
This process is executing in n iterations, to more iterations, less error

The matrix P has this shape
    |1                                  |
    |   1                               |
P = |       Cos(theta)  Sin(theta)      |
    |       -Sin(theta) Cos(theta)      |
    |                   1               |
    |                               1   |
"""
import numpy as np


def eigen(matrix, iterations=20):
    # Verify that the matrix is square
    [rows, columns] = matrix.shape
    if rows != columns:
        print("******************************************************************************\n"
              "Warning\n"
              "The matrix must be square\n"
              "******************************************************************************")
        return [], []

    # Initialization eigen vectors and eigen values
    eigen_values = []
    eigen_vectors = np.identity(rows)

    # Loop Jacobi Algorithm
    for iteration in range(iterations):

        # Look for the maximum value in main diagonal
        arg_max = 0
        arg_max_row, arg_max_column = 0, 0
        for row in range(rows):
            for column in range(row + 1, columns):
                if row != column:
                    absolute = np.absolute(matrix[row, column])
                    if absolute > arg_max:
                        arg_max = absolute
                        arg_max_row, arg_max_column = row, column

        # initialization orthogonal matrix (p) as identity matrix
        p = np.identity(rows)

        # Calculating values x, y, z, sin and cos
        y = matrix[arg_max_row, arg_max_row] - matrix[arg_max_column, arg_max_column]
        if np.absolute(y) == 0:
            # if |y| < 0 then cos(theta) and sin (theta) are equals, so theta is pi / 4
            cos, sin = np.sin(np.pi / 4), np.sin(np.pi / 4)
        else:
            x = 2 * matrix[arg_max_row, arg_max_column]
            z = np.sqrt(np.power(x, 2) + np.power(y, 2))
            cos = np.sqrt((z + y) / (2 * z))
            sin = ((x / y) / (np.absolute(x / y))) * (np.sqrt((z - y) / (2 * z)))

        # Update orthogonal matrix values
        p[arg_max_row, arg_max_row] = cos
        p[arg_max_column, arg_max_column] = cos
        p[arg_max_row, arg_max_column] = sin
        p[arg_max_column, arg_max_row] = -sin

        # Solve Jacobi Algorithm
        matrix = np.dot(p, np.dot(matrix, np.transpose(p)))
        eigen_vectors = np.dot(eigen_vectors, np.transpose(p))

    for row in range(rows):
        eigen_values.append(matrix[row, row])

    return eigen_values, eigen_vectors
