# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import numpy as np
from scipy.stats import ortho_group
import pytest

# ------------------------------- Intra-package ------------------------------ #

import math_tools.linalg as la

# ---------------------------------------------------------------------------- #
#                            Tests for cob function                            #
# ---------------------------------------------------------------------------- #


def test_cob():
    """
    Test the cob function.
    """
    # Test 1:
    new_vec = la.cob(np.array([10, -5]), np.array([3, 4]), np.array([4, -3]))
    assert la.MyMatrix.is_coord_vec(np.array([10, -5]), new_vec,
                                    np.array([3, 4]), np.array([4, -3]))

    # Test 2:
    new_vec = la.cob(
        (1, 1, 1),
        (2, 1, 0),
        (1, -2, -1),
        (-1, 2, -5)
    )
    assert la.MyMatrix.is_coord_vec((1, 1, 1), new_vec,
                                    np.array([2, 1, 0]), np.array([1, -2, -1]),
                                    np.array([-1, 2, -5]))

    # Test 3:
    new_vec = la.cob(
        [1, 1, 2, 3],
        [1, 0, 0, 0],
        [0, 2, -1, 0],
        [0, 1, 2, 0],
        [0, 0, 0, 3]
    )
    assert la.MyMatrix.is_coord_vec([1, 1, 2, 3], new_vec,
                                    np.array([1, 0, 0, 0]), np.array(
                                        [0, 2, -1, 0]),
                                    np.array([0, 1, 2, 0]), np.array([0, 0, 0, 3]))

# ---------------------------------------------------------------------------- #
#                            Tests for proj function                           #
# ---------------------------------------------------------------------------- #


def test_proj():
    """
    Test the proj function.
    """

    # Test 1:
    assert isinstance(
        la.proj(np.array([2, 4, 0]), np.array([4, 2, 4]), 'scalar'),
        np.float64
    )

    # Test 2:
    assert isinstance(
        la.proj(np.array([2, 1]), np.array([3, -4]), 'vector'),
        np.ndarray
    )

    # Test 3:
    assert isinstance(la.proj([2, 1, 5, 6, 10, 5],
                              (3, -4, 0, 3, 0, 0), 'vector'),
                      np.ndarray)

    # Test 4:
    assert isinstance(la.proj([2, 1, 5, 6, 10, 5],
                              (3, -4, 0, 3, 0, 0), 'scalar'), np.float64)


# ---------------------------------------------------------------------------- #
#                          Tests for lin_ind function                          #
# ---------------------------------------------------------------------------- #

def test_lin_ind(capfd):
    """
    Test the lin_ind function.
    """

    # Test 1 (Linear independent)
    la.lin_ind([1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0])
    captured = capfd.readouterr()
    assert captured.out == "The matrix with input vectors as columns has full column rank, and so the input vectors are linearly independent\n"

    # Test 2 (Linear dependent)
    assert np.allclose(la.lin_ind([1, 0, 0, 0], [1, 9, 3, 0], [0, 0, 4, 1], [
                       0, 1, 0, 0], [2, 3, 4, 9]), np.array([[2, 3, 4, 9]]))

    # Test 3
    assert np.allclose(la.lin_ind([1, 2], [2, 4], [2, 1]), np.array([[2, 4]]))

# ---------------------------------------------------------------------------- #
#                           Tests for MyMatrix class                           #
# ---------------------------------------------------------------------------- #


class TestMyMatrix:
    """
    Tests for MyMatrix class.
    """

    # ---------------------- Tests for default instantiation --------------------- #

    @pytest.mark.parametrize(
        "data, axis, expected",
        [
            # List of lists
            ([[1, 2], [1, 2]], 1, np.array([[1, 1], [2, 2]])),
            # Tuple of tuple
            (((1, 2), [1, 2]), 0, np.array([[1, 2], [1, 2]])),
            # Non-square matrices
            ([[1, 2, 3], [1, 2, 4]], 1, np.stack([[1, 2, 3], [1, 2, 4]], axis=1)),
            ([[1, 2], [1, 2], [7, 7]], 1, np.stack(
                [[1, 2], [1, 2], [7, 7]], axis=1))
        ],
        scope='function'
    )
    def test_default(self, data, axis, expected):
        """
        Test default instantiation.
        """

        assert np.allclose(la.MyMatrix.default(data, axis).X, expected)

        assert hasattr(la.MyMatrix.default(data, axis), 'X')
        assert hasattr(la.MyMatrix.default(data, axis), 'det')
        assert hasattr(la.MyMatrix.default(data, axis), 'row')
        assert hasattr(la.MyMatrix.default(data, axis), 'col')

    # ----------------- Tests for change of basis matrix function ---------------- #

    def test_cob_mat(self):
        """
        Tests for change of basis matrix function.
        """

        # Test 1:
        assert np.allclose(
            # Change of basis from [[1, 2, -3], [4, -1, -3]] to [[0, 1, -1], [1, 0, -1]]
            la.MyMatrix.cob_mat([[1, 2, -3], [4, -1, -3]],
                                [[0, 1, -1], [1, 0, -1]], False),
            # Inverse of change of basis matrix above
            np.linalg.inv(la.MyMatrix.cob_mat([[1, 2, -3], [4, -1, -3]],
                                              [[0, 1, -1], [1, 0, -1]], True))
        )

        # Test 2: Change of basis matrix from standard basis to an arbitrary basis
        mat = la.MyMatrix.cob_mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  [[2, 1, 0], [1, -2, -1], [-1, 2, -5]],
                                  False)
        # This should be equal
        np.allclose(
            # The change of basis matrix should translate [1, 1, 1] from standard basis to new basis
            a=mat @ np.array([1, 1, 1]),
            # The above should be equivalent to finding the coordinate vector of [1, 1, 1] with respect to the new basis vectors
            b=la.cob(
                np.array([1, 1, 1]),
                np.array([2, 1, 0]),
                np.array([1, -2, -1]),
                np.array([-1, 2, -5])
            )
        )
