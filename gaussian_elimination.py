import numpy as np
from typing import Union

np.set_printoptions(suppress=True, linewidth=500)

FLAG = True


def swap_rows_to_make_pivot_non_zero(augmented_matrix, piv_row, piv_col) -> bool:
    for r in range(piv_row + 1, augmented_matrix.shape[0]):
        if not augmented_matrix[r, piv_col] == 0:
            # set of rows from right array assigned to respective set of rows of the left array
            augmented_matrix[[piv_row, r]] = augmented_matrix[[r, piv_row]]
            return True
    else:
        return False


def forward_elimination(A: np.ndarray, B: np.ndarray, d: bool) -> np.ndarray:
    """
    Manipulates the matrices to form an upper diagonal matrix from A and B.
    1. Subtract from each elem[r, c], r = [1:], the values elem[0, c] * elem[r, c] / elem[0, 0]
    2. Repeat step 1 for values [1:] instead of 0 with column value c incrementing each iteration. .
    :param A: Matrix A passed into the GaussianElimination function.
    :param B: Matrix B passed into the GaussianElimination function.
    :param d: Flag d passed into the GaussianElimination function.
    :return: Augmented Matrix consisting A and B transformed to upper diagonal matrix.
    """
    augmented_matrix = np.hstack((A, B))
    N_ROW = augmented_matrix.shape[0]

    n_step = 0
    if d:
        print("Step ", n_step, " : \n", augmented_matrix, "\n")
        n_step += 1

    piv_col = 0
    for piv_row in range(0, N_ROW - 1):
        if augmented_matrix[piv_row, piv_col] == 0:
            is_found = swap_rows_to_make_pivot_non_zero(augmented_matrix, piv_row, piv_col)

            if not is_found:
                raise ArithmeticError("Number equations don't match the number of variables")

        for r in range(piv_row + 1, N_ROW):
            augmented_matrix[r] -= augmented_matrix[piv_row] * (augmented_matrix[r, piv_col] /
                                                                augmented_matrix[piv_row, piv_col])

            if d:
                print("Step ", n_step, " : \n", augmented_matrix, "\n")
                n_step += 1

        piv_col += 1

    return augmented_matrix


def back_substitution(augmented_matrix: np.ndarray) -> tuple:
    """
    Extracts the solutions of the variables from the upper diagonal matrix from
    forward elimination.
    :param augmented_matrix: Augmented matrix returned from forward_elimination.
    :return: tuple consisting of the solution of the linear equations.
    """
    N_ROW, N_COL = augmented_matrix.shape
    solutions = []

    col = 1
    for row in range(1, N_ROW + 1):
        # sol is the result of backpropagation for each variable starting from the end
        sol = (augmented_matrix[-row, N_COL - 1] - np.sum(augmented_matrix[-row, -col: -1] * solutions)) / \
              augmented_matrix[-row, -col - 1]
        solutions.insert(0, sol)
        col += 1

    return tuple(solutions)


def GaussianElimination(A: Union[list, np.ndarray], B: Union[list, np.ndarray], d: bool = True) -> tuple:
    """
    Uses Gaussian Elimination to solve a system of linear equations.
    :param A: Matrix A, the coefficient matrix. Has to be in n x n shape.
    :param B: Matrix B, the constant matrix. Has to be in n x 1 shape.
    :param d: Flag, to determine if matrices will be printed after forward elimination sub-steps
    :return: Tuple, denoting the a column vector with the solutions in sequence.
    """
    if isinstance(A, list):
        A = np.array(A, dtype="float64")

    if isinstance(B, list):
        B = np.array(B, dtype="float64").reshape(A.shape[0], 1)

    if not A.shape[0] == B.shape[0] or not A.shape[0] == A.shape[1] or not B.shape[1] == 1:
        raise ValueError("Shapes aren't consistent")

    upper_diagonal_matrix = forward_elimination(A, B, d)
    solutions = back_substitution(upper_diagonal_matrix)

    return solutions


# client
if __name__ == '__main__':
    num_var = int(input())
    a = []
    for _ in range(0, num_var):
        a.append([float(elem) for elem in input().split()])

    input()

    b = []
    for _ in range(0, num_var):
        b.append((float(input())))

    for sol in GaussianElimination(a, b, FLAG):
        print("{:.4f}".format(sol))
