"""

Questions:
    • Using scipy, write a code to:
• Finding the inverse of an n x n matrix
• Solve a linear system involving an n x n matrix
• Find the determinant of an n x n matrix
• Solve the eigenvalue-eigenvector problem involving an n x n matrix
"""
import numpy as np
import scipy.linalg
from scipy import linalg


def prep_zeros(n):
    first_col = np.zeros((1, n))
    first_col[0, 0] = 2
    first_col[0, 1] = -1
    return first_col


def make_inverse(n=3):
    first_col = prep_zeros(n)
    matr = linalg.toeplitz(first_col)
    return linalg.inv(matr)


def make_linear_eqn(n=3):
    first_col = prep_zeros(n)
    b = np.ones(n)
    b[0] = 2
    b[1] = 1
    a = linalg.toeplitz(first_col)
    return linalg.solve(a, b)


print(
    f"""
inv: 
    {make_inverse()}

eqn: 
    {make_linear_eqn()}
"""
)
