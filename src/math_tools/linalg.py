# ---------------------------------------------------------------------------- #
#                              Package and Modules                             #
# ---------------------------------------------------------------------------- #

import numpy as np
import numpy.linalg as la
from sympy import Matrix

# ----------------------------- Standard library ----------------------------- #

from typing import Union, List, Tuple

# --------------- Express vector in terms of new basis vectors --------------- #


def cob(vector: Union[np.ndarray, List, Tuple], *args: Union[np.ndarray, List, Tuple]) -> np.ndarray:
    """
    Express a vector in a new basis spanned by a pairwise orthogonal (not orthonomal) set basis vectors.

    Parameters
    ----------
    vector : Union[np.ndarray, List, Tuple]
        An n-dimensional vector.
    args : Union[np.ndarray, List, Tuple]
        An arbitrary number of pairwise orthogonal n-dimensional vectors.

    Returns
    -------
    np.ndarray
        The original vector expressed in a new basis.
    """
    # Cast to np.ndarray
    vector = np.array(vector)
    # Projection
    scalar_projections = [np.dot(vector, np.array(basis)) / (la.norm(np.array(basis)))
                          ** 2 for basis in args]
    return np.array(scalar_projections)

# ---------- Scalar or vector projection of one vector onto another ---------- #


def proj(v1: Union[np.ndarray, List, Tuple], v2: Union[np.ndarray, List, Tuple], proj_type: str = 'vector') -> Union[np.ndarray, np.float64]:
    """
    Obtain the scalar or vector projection of vector `v1` onto vector `v2`.

    Parameters
    ----------
    v1 : Union[np.ndarray, List, Tuple]
        An n-dimensional vector.
    v2 : Union[np.ndarray, List, Tuple]
        An n-dimensional vector.
    proj_type : str, optional
        Type of projection, by default 'vector'.


    Returns
    -------
    np.ndarray
        The scalar or vector projection of vector `v1` onto vector `v2`.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    if (proj_type == 'scalar'):
        out = (np.dot(v1, v2) / (la.norm(v2)))
    elif (proj_type == 'vector'):
        out = (np.dot(v1, v2) / (la.norm(v2)) ** 2) * v2
    else:
        raise ValueError("'proj_type' must either be 'vector' or 'scalar")

    return out

# ----------------------- Check for linear independence ---------------------- #


def lin_ind(*args: Union[np.ndarray, List, Tuple]) -> Union[None, np.ndarray]:
    """
    Check if the input vectors are linearly independent. If not, return the redundant
    column vectors as an `np.ndarray`.

    Returns
    -------
    None or np.ndarray
        An `np.ndarray` of redundant columns or None if the input vectors are linearly independent.
    """
    # Convert input tuple of vectors into an ndarray for indexing later
    args = np.array(args)

    # Need to transpose the matrix so each vector in 'args' is a column vector
    # Pivot column indices are stored as the second element of the tuple returned by rref()
    pivot_indices = Matrix(args).T.rref()[1]

    # If number of  elements in 'pivot_indices' equals the number of input vectors, then the matrix has full rank
    if (len(pivot_indices) == args.shape[0]):
        print("The matrix with input vectors as columns is full column rank, and so the input vectors are linearly independent")
        return None
    else:
        # Return non-pivot columns
        return np.delete(args, pivot_indices, axis=0)

# ------------------------------- Gram-schmidt ------------------------------- #


def gs(X):
    """
    This function creates an orthogonal matrix (a set of orthonormal basis vectors)
    given an input matrix `X`.

    Parameters
    ----------
    X : np.ndarray
        An input matrix.

    Returns
    -------
    np.ndarray
        An orthogonal matrix.
    """
    return la.qr(X)[0]


if __name__ == '__main__':

    # ------------------------------- Tests for cob ------------------------------ #

    # Run one
    cob(np.array([10, -5]), np.array([3, 4]), np.array([4, -3]))

    # Run two
    cob(
        np.array([2, 2]),
        np.array([-3, 1]),
        np.array([1, 3])
    )

    # Run three
    cob(
        np.array([1, 1, 1]),
        np.array([2, 1, 0]),
        np.array([1, -2, -1]),
        np.array([-1, 2, -5])
    )

    # Run four
    cob(
        np.array([1, 1, 2, 3]),
        np.array([1, 0, 0, 0]),
        np.array([0, 2, -1, 0]),
        np.array([0, 1, 2, 0]),
        np.array([0, 0, 0, 3])
    )

    # Run five
    cob(
        np.array([-4, -3, 8]),
        np.array([1, 2, 3]),
        np.array([-2, 1, 0]),
        np.array([-3, -6, 5])
    )

    # ------------------------------ Tests for proj ------------------------------ #

    # One
    proj(np.array([2, 4, 0]), np.array([4, 2, 4]), 'scalar')

    # Two
    proj(np.array([2, 1]), np.array([3, -4]), 'scalar')

    # ---------------------------- Tests for lin_indep --------------------------- #

    # Linearly independent inputs
    lin_ind([1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0])

    # Linearly dependent inputs
    lin_ind([1, 0, 0, 0], [1, 9, 3, 0], [
        0, 0, 4, 1], [0, 1, 0, 0], [2, 3, 4, 9])

    lin_ind([1, 2], [2, 4], [2, 1])

    # ------------------------------- Tests for gs ------------------------------- #

    # Singular
    gs(np.array([[1, 0],
                 [0, 0]], dtype=np.float_))

    # Non-square
    gs(np.array([[3, 2, 3],
                 [2, 5, -1],
                 [2, 4, 8],
                 [12, 2, 1]], dtype=np.float_))
