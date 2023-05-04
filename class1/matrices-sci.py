import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz

"""
    Class summary:
    
    I made matrices k, t, b, and c

    docs and exploration come after as appendices, but main functions are:
        make_k
        make_t
        make_b
        make_c
    Todo: Make
"""


def make_t(n=4):
    """
    Mutative version of the t matrix based on example

    """
    first_col = np.zeros((1, n))
    first_col[0, 0] = 2
    first_col[0, 1] = -1
    toe = linalg.toeplitz(first_col)
    toe[0, 0] = 1

    return toe


def make_b(n=4):
    """
    Mutative version of these based on example
    """
    first_col = np.zeros((1, n))
    first_col[0, 0] = 2
    first_col[0, 1] = -1
    toe = linalg.toeplitz(first_col)
    toe[0, 0] = 1
    toe[n - 1, n - 1] = 1
    return toe


def make_c(n=4):
    first_col = np.zeros((1, n))
    first_col[0, 0] = 2
    first_col[0, 1] = -1
    first_col[0, n - 1] = -1
    some_circ = linalg.circulant(first_col)
    return some_circ


def make_k(n=4):
    first_col = np.zeros((1, n))
    first_col[0, 0] = 2
    first_col[0, 1] = -1
    toe = linalg.toeplitz(first_col)
    return toe


"""
    Exec:
"""
k = make_k()
t = make_t()
b = make_b()
c = make_c()
print(f"""
  Matrix k:
    {k}
  Matrix t:
    {t}
  Matrix b:
    {b}
  Matrix c:
    {c}
""")

"""
    APPENDIX:
"""


def example_k_constant_diag():
    """
    1. Create a toeplitz matrix (Has a constant diagonal)

    What is a toeplitz matrix?

    A matrix with constant diagonals, first studied by toeplitz.

    Aka time-invariant.
    Note:
        The boundary condition is the last and first row.

    What are the fns provided?
        - Zeros, gives matrix filled with zeros with shape passed in
        - Ones, gives same as above, with ones

    A toeplitz matrix, when symmetric, only needs the first row.
    When asymmetric, would need the first row and column as those will be constant.
    """

    first_row = [2, -1, 0, 0, 0, 0]
    first_column = [2, -1, 0, 0]

    toe_asymmetric = toeplitz(first_column, first_row)
    toe_symmetric = toeplitz(first_column)
    print(toe_asymmetric, toe_asymmetric.shape)  # 4x6

    print(toe_symmetric, toe_symmetric.shape)  # 4x4


"""

2. Create a kroncker sum of a matrix

What is a kroncker sum? [Ref](https://youtu.be/jmX4FOUEfgU?t=4416)

It takes a nxn matrix, and converts it into a n^2 matrix,

Kroncker sum is defined for some nxn matrix X as: 
    kron_product(X, identity(dim(X))) + kron_product(identity(dim(X)), X).
    
    Using straightforward approach:
"""


def example_2_straight():
    x = np.array(
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]]
    )

    n = x.shape[0]
    x_kron_sum = linalg.kron(x, np.identity(n)) + linalg.kron(np.identity(n), x)

    print(x_kron_sum)


def circulant_matrix():
    """
       Print default circulant example

       Ex: https://youtu.be/CgfkEUOFAj0?list=PLF706B428FB7BD52C&t=2101

       It's circulant because the diagonals loop around.
       It's also not invertible.

       because it solves Cu = 0;
       With: u = ones(dim(C))

       Reason it's not invertible?
       Multiply both sides by C^-1, and we find inequality. As C-1 suggests that the only possible value of u is zero,
       while we solved already with u as ones
    """

    some_circ = linalg.circulant([2, -1, 0, -1])
    print(some_circ)


def fun_zero_ones():
    """
    Test out zeros and ones fns
    """
    zeros = np.zeros([2, 3])
    ones = np.ones([2, 3])

    print(zeros, zeros.shape)
    print(ones, ones.shape)


def others_from_class():
    """
        For this, I assume B and T are the last two examples he gave, concluding.

        T: Free-fixed matrix; Aka Top.
        B: Both free matrix
    More
        Fixed: The displacement is zero. I.e u makes it zero.
        Free: The fifth guy is same as the fourth. I.e slope is zero.
        Both b_matrix and t_matrix are not invertible, and the u vector proves in both cases as the rows add to 0

    More_more:
        - K and T are positive definite matrices

        - C, B are positive semidefinite matrices (because they hit zero somehow)

        If there is a symmetric matrix and the pivots are all positive, it is not only invertible,
            More than that, the matrix is positive definite.
        We'll see that all the eigen values are positive definite. It has to do with least squares, etc.

    """
    # Start with toeplitz
    # Sub 1 from top-left corner (free one end)
    first_col = [2, -1, 0, 0]
    toe = linalg.toeplitz(first_col)
    min_1_top = [-1, 0, 0, 0]
    zeros = np.zeros([3, 4])

    combined = np.array([min_1_top, zeros[0], zeros[1], zeros[2]])

    t_matrix = toe + combined
    print(t_matrix)

    # Start with T_matrix
    # Sub 1 from bottom-right corner (free both ends)
    min_1_bottom = [0, 0, 0, 1]
    combined = np.array([zeros[0], zeros[1], zeros[2], min_1_bottom])
    b_matrix = t_matrix - combined
    print(b_matrix)
