import numpy as np
import numpy.linalg as la

# --------------- Express vector in terms of new basis vectors --------------- #


def cob(vector: np.array, *args: np.array) -> np.array:
    """
    Express a vector in terms of an orthogonal (not orthonomal) basis.

    Parameters
    ----------
    vector : np.array
        An n-dimensional vector.
    args : np.array
        An arbitrary number of n-dimensional vector.

    Returns
    -------
    np.array
        The original vector expressed in a new basis.
    """
    scalar_proj = [np.dot(vector, basis) / (la.norm(basis))
                   ** 2 for basis in args]
    return np.array(scalar_proj)

# --------------- Vector projection of one vector onto another --------------- #


def proj(v1: np.array, v2: np.array, type='vector'):
    """
    Obtain the vector or scalar project of vector `v1` onto vector `v2`.

    Parameters
    ----------
    v1 : np.array
        An n-dimensional vector.
    v2 : np.array
        An n-dimensional vector.

    Returns
    -------
    np.array
        The scalar or vector project of vector `v1` onto vector `v2`.
    """

    if (type == 'scalar'):
        out = (np.dot(v1, v2) / (la.norm(v2)) ** 2)
    elif (type == 'vector'):
        out = (np.dot(v1, v2) / (la.norm(v2)) ** 2) * v2
    else:
        ValueError("'type' must either be 'vector' or 'scalar")

    return out


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

    # ------------------------------ tests for proj ------------------------------ #

    # One
    proj(np.array([1, 2]), np.array([1, 1]))

    # Two
    proj(np.array([2, 1]), np.array([3, -4]))
