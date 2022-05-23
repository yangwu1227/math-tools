# ---------------------------------------------------------------------------- #
#                              Package and Modules                             #
# ---------------------------------------------------------------------------- #

import numpy as np
import numpy.linalg as la
from sympy import Matrix
from sympy.core.expr import Expr
from scipy.linalg import eig

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
        An arbitrary number of pairwise orthogonal n-dimensional basis vectors.

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

# ------------------------------ MyMatrix class ------------------------------ #


class MyMatrix:
    """
    A utility class for matrix.

    Attributes:
            X (np.ndarray): A matrix.
            row (int): Number of rows.
            col (int): Number of columns.
    """
    # ---------------------------------------------------------------------------- #
    #                                 Constructors                                 #
    # ---------------------------------------------------------------------------- #

    def __init__(self, X: Union[List[List], Tuple[Tuple], np.ndarray]) -> None:
        """
        A parameterized class constructor.

        Parameters
        ----------
        X : Union[List[List], Tuple[Tuple], np.ndarray]
            A matrix.
        """
        self.X = X
        # Other instance attributes
        self.row = self.X.shape[0]
        self.col = self.X.shape[1]

    @classmethod
    def default(cls, X: Union[List[List], Tuple[Tuple]], axis: int = 1) -> None:
        """
        Instantiate a class instance from a list of lists or a tuple of tuples, whose
        elements can either be the rows (axis=0) or the columns (axis=1) of the matrix.

        Parameters
        ----------
        X : Union[List[List], Tuple[Tuple]]
            Input vectors.
        axis : int, optional
            Whether to process input elements as rows vectors or column vectors, by default 1.

        Returns
        -------
        MyMatrix
            An instance of of class MyMatrix.

        Raises
        ------
        ValueError
            The argument 'axis' must either be 1 or 0.
        """
        # Input as column vectors
        if (axis == 1):
            X = np.stack(X, axis=1)
        elif (axis == 0):  # Input as row vectors
            X = np.array(X)
        else:
            raise ValueError("The argument 'axis' must either be 1 or 0")

        return cls(X)

    @classmethod
    def from_numpy(cls, X: np.ndarray) -> None:
        """
        Instantiate a class instance from a numpy n-dimensional array. The input
        must be already be in the desired shape since this constructor is a no-op.

        Parameters
        ----------
        X : np.ndarray
            A matrix.

        Returns
        -------
        MyMatrix
            An instance of of class MyMatrix.
        """
        return cls(X)

    @classmethod
    def zeros(cls, shape: Tuple[int], dtype: np.dtype = np.float64) -> None:
        """
        Instantiate a class instance with a zero matrix.

        Parameters
        ----------
        shape : Tuple[int]
            Shape of the zero matrix, e.g. (2, 4).
        dtype : np.dtype, optional
            The desired data-type for the matrix, by default np.float64.

        Returns
        -------
        MyMatrix
            An instance of of class MyMatrix.
        """
        return cls(np.zeros(shape=shape, dtype=dtype))

    @classmethod
    def eye(cls, N: int, M: int = None, k: int = 0, dtype: np.dtype = np.int8) -> None:
        """
        Instantiate a class instance with an identity matrix.

        Parameters
        ----------
        N : int
            Number of rows.
        M : int, optional
            Number of columns, by default None, which takes the value of N.
        k : int, optional
            Index of the diagonal where 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal, and a negative value to a 
            lower diagonal, by default 0.
        dtype : np.dtype, optional
            Data-type of identity matrix, by default np.int8.

        Returns
        -------
        MyMatrix
            An instance of of class MyMatrix.
        """
        return cls(np.eye(N=N, M=M, k=k, dtype=dtype))

    # ---------------------------------------------------------------------------- #
    #                                Static methods                                #
    # ---------------------------------------------------------------------------- #

    # -------------------------- Change of basis matrix -------------------------- #

    @staticmethod
    def cob_mat(original_basis: List[List], new_basis: List[List], inv: bool = False) -> np.ndarray:
        """
        Find the change of basis matrix `S` from `original_basis` to `new_basis`. The change of basis
        matrix from basis A to basis B is defined to be:

        .. math::
            S_{A \rightarrow B}=\left[\begin{array}{ccc} \mid & & \mid \\
            {\left[a_{1}\right]_{B}} & \cdots & {\left[a_{n}\right]_{B}} \\
                \mid & & \mid
                \end{array}\right]

        Parameters
        ----------
        original_basis : List[List]
            A set of `n` basis vectors for :math:`R^{n}` or `m` basis vectors for a subspace of :math:`R^{n}`.
        new_basis : List[List]
            A set of `n` basis vectors for :math:`R^{n}` or `m` basis vectors for a subspace of :math:`R^{n}`.
        inv : bool, optional
            Whether to return the inverse of the change of basis matrix, by default False.

        Returns
        -------
        np.ndarray
            The change of basis matrix or the inverse of it, which translates from `new_basis` back to `original_basis`.
        """
        # Matrix with new basis vectors as columns
        mat_new_basis = np.stack(new_basis, axis=1)

        # Convert each element (vector) in 'original_basis' to np.ndarray
        original_basis = [np.array(vec) for vec in original_basis]

        # Find coordinate vectors of original basis vectors with respect to new basis vectors
        list_of_coord_vectors = [la.lstsq(mat_new_basis, vec, rcond=None)[0]
                                 for vec in original_basis]

        # Concatenate coordinate vectors (np.ndarray) to form change of basis vectors
        if (inv):
            cob = np.stack(list_of_coord_vectors, axis=1)
            return la.inv(cob)
        else:
            return np.stack(list_of_coord_vectors, axis=1)

    # ---------------------------------------------------------------------------- #
    #                               Instance methods                               #
    # ---------------------------------------------------------------------------- #

    # ------------------------- Check if matrix is square ------------------------ #

    def is_sq(self) -> bool:
        """
        Check if matrix `X` is square.

        Returns
        -------
        bool
            True if square or False if non-square.
        """
        return self.row == self.col

    # ------------------------------- Gram-schmidt ------------------------------- #

    def gs(self, ret_type: str = 'matrix') -> np.ndarray:
        """
        This function creates an orthogonal matrix (a set of orthonormal basis vectors)
        for matrix `X`.

        Parameters
        ----------
        ret_type : str, optional
            Whether to return a set of orthonormal basis vectors or an orthogonal matrix, by default 'matrix'.

        Returns
        -------
        np.ndarray
            A set of orthonormal basis vectors or an orthogonal matrix, depending on the `ret_type` argument.
        """
        if (ret_type == 'vector'):
           # Create orthogonal matrix
            orthogonal_mat = la.qr(self.X)[0]

            # Return a list of orthonormal basis vectors
            orthonormal_vectors = [orthogonal_mat[:, col_index]
                                   for col_index in range(self.col)]
            return orthonormal_vectors
        elif (ret_type == 'matrix'):
            return la.qr(self.X)[0]

    # --------------------- Find eigenvalues and eigenvectors -------------------- #

    def eigen(self, real: bool = True) -> Tuple:
        """
        Find the eigenvalues and eigenvectors of the matrix `X`.

        Parameters
        ----------
        real : bool, optional
            Whether to convert the array of eigenvalues to real numbers, by default True.

        Returns
        -------
        w (…, M) array 
            The eigenvalues, each repeated according to its multiplicity. The eigenvalues are not necessarily ordered. The resulting array will be of complex type, unless the imaginary part is zero in which case it will be cast to a real type. 
            When `X` is real the resulting eigenvalues will be real (0 imaginary part) or occur in conjugate pairs.
        vl(…, M, M) array
            The normalized (unit “length”) left eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        vr(…, M, M) array
            The normalized (unit “length”) right eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

        Raises
        ------
        la.LinAlgError
            The matrix X must be square.
        """
        # Check if square
        if (not self.is_sq()):
            raise la.LinAlgError('The matrix X is not square')

        # Compute
        eig_vals, left_eig_vecs, right_eig_vecs = eig(
            self.X, left=True, right=True)

        if (real):
            return eig_vals.real, left_eig_vecs, right_eig_vecs
        else:
            return eig_vals, left_eig_vecs, right_eig_vecs

    # -------------- Find the characteristic polynomial of a matrix -------------- #

    def char_poly(self):
        """
        Computes characteristic polynomial det(lambda*I - M) where I is
        the identity matrix.


        Returns
        -------
        sympy expression
            A class instance inheriting from the Base class for algebraic expressions `Expr`.
        """
        return Matrix(self.X).charpoly().as_expr()


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
    MyMatrix.from_numpy(np.array([[1, 0],
                                  [0, 0]], dtype=np.float_)).gs(ret_type='matrix')
    MyMatrix.from_numpy(np.array([[1, 0],
                                  [0, 0]], dtype=np.float_)).gs(ret_type='vector')

    # Non-square
    MyMatrix.from_numpy(np.array([[3, 2, 3],
                                  [2, 5, -1],
                                  [2, 4, 8],
                                  [12, 2, 1]], dtype=np.float_)).gs(ret_type='matrix')
    MyMatrix.from_numpy(np.array([[3, 2, 3],
                                  [2, 5, -1],
                                  [2, 4, 8],
                                  [12, 2, 1]], dtype=np.float_)).gs(ret_type='vector')

    # Pass a list of vectors
    MyMatrix.default([[3, 2, 2, 12],
                      [2, 5, 4, 2],
                      [3, -1, 8, 1]]).gs(ret_type='matrix')
    MyMatrix.default([[3, 2, 2, 12],
                      [2, 5, 4, 2],
                      [3, -1, 8, 1]]).gs(ret_type='vector')

    # ----------------------------- Tests for cob_mat ---------------------------- #

    # Subspace (a plane) of R3
    MyMatrix.cob_mat([[1, 2, -3], [4, -1, -3]],
                     [[0, 1, -1], [1, 0, -1]], False)
    MyMatrix.cob_mat([[1, 2, -3], [4, -1, -3]], [[0, 1, -1], [1, 0, -1]], True)

    # Bases for R3
    mat = MyMatrix.cob_mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                           [[2, 1, 0], [1, -2, -1], [-1, 2, -5]],
                           False)
    # This should be equal
    np.allclose(
        a=mat @ np.array([1, 1, 1]),
        b=cob(
            np.array([1, 1, 1]),
            np.array([2, 1, 0]),
            np.array([1, -2, -1]),
            np.array([-1, 2, -5])
        )
    )

    # ------------------------------ Tests for eigen ----------------------------- #

    # First example
    MyMatrix.default([[1, 0], [0, 2]]).eigen()

    # Second example
    MyMatrix.default([[3, 0], [4, 5]]).eigen()

    # Third example
    MyMatrix.default([[1, -1], [0, 4]]).eigen()

    # Fourth example
    MyMatrix.default([[-3, 2], [8, 3]]).eigen()

    # Fifth example
    MyMatrix.default([[5, -4], [4, -3]]).eigen()

    # Sixth example
    MyMatrix.default([[-2, 1], [-3, 1]]).eigen()

    # ---------------------------- Tests for char_poly --------------------------- #

    # First example
    MyMatrix.default([[3, 0], [4, 5]]).char_poly()

    # Second example
    MyMatrix.default([[3, 0], [4, 5]]).char_poly()

    # Third example
    MyMatrix.default([[1, -1], [0, 4]]).char_poly()

    # Fourth example
    MyMatrix.default([[-3, 2], [8, 3]]).char_poly()

    # Fifth example
    MyMatrix.default([[5, -4], [4, -3]]).char_poly()

    # Sixth example
    MyMatrix.default([[-2, 1], [-3, 1]]).char_poly()
