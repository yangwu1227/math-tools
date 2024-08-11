import numpy as np
import numpy.linalg as la
from scipy.linalg import eig, svdvals  # type: ignore
from sympy import Matrix  # type: ignore
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional, Self, Any


def cob(
    vector: Union[np.ndarray, List, Tuple], *args: Union[np.ndarray, List, Tuple]
) -> np.ndarray:
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

    Examples
    --------
    >>> import linalg as la
    >>> # Express [10, -5] in the basis spanned by [3, 4] and [4, -3]
    >>> la.cob(vector=np.array([10, -5]), np.array([3, 4]), np.array([4, -3]))
    """
    # Cast to np.ndarray
    vector = np.array(vector)
    # Projection
    scalar_projections = [
        np.dot(vector, np.array(basis)) / (la.norm(np.array(basis))) ** 2
        for basis in args
    ]
    return np.array(scalar_projections)


def proj(
    v1: Union[np.ndarray, List, Tuple],
    v2: Union[np.ndarray, List, Tuple],
    proj_type: str = "vector",
) -> Union[np.ndarray, np.float64]:
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

    Examples
    --------
    >>> import linalg as la
    >>> la.proj(np.array([2, 4, 0]), np.array([4, 2, 4]), 'scalar')
    np.float64(2.6666666666666665)
    >>> la.proj(np.array([2, 1]), np.array([3, -4]), 'vector')
    array([ 0.24, -0.32])
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    if proj_type == "scalar":
        out = np.dot(v1, v2) / (la.norm(v2))
    elif proj_type == "vector":
        out = (np.dot(v1, v2) / (la.norm(v2)) ** 2) * v2
    else:
        raise ValueError("'proj_type' must either be 'vector' or 'scalar")

    return out


def lin_ind(*args: Union[np.ndarray, List, Tuple]) -> Union[None, np.ndarray]:
    """
    Check if the input vectors are linearly independent. If not, return the redundant
    column vectors as an `np.ndarray`.

    Parameters
    ----------
    args : Union[np.ndarray, List, Tuple]
        An arbitrary number of vectors.

    Returns
    -------
    None or np.ndarray
        An `np.ndarray` of redundant columns or None if the input vectors are linearly independent.

    Examples
    --------
    >>> import linalg as la
    >>> # Linear independent
    >>> la.lin_ind([1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0])
    The matrix with input vectors as columns has full column rank, and so the input vectors are linearly independent
    >>> # Linear dependent
    >>> lin_ind([1, 0, 0, 0], [1, 9, 3, 0], [0, 0, 4, 1], [0, 1, 0, 0], [2, 3, 4, 9])
    array([[2, 3, 4, 9]])
    """
    # Convert input tuple of vectors into an ndarray for indexing later
    input_vectors = np.array(args)

    # Need to transpose the matrix so each vector in 'args' is a column vector
    # Pivot column indices are stored as the second element of the tuple returned by rref()
    pivot_indices = Matrix(input_vectors).T.rref()[1]

    # If number of elements in 'pivot_indices' equals the number of input vectors, then the matrix has full rank
    if len(pivot_indices) == len(args):
        print(
            "The matrix with input vectors as columns has full column rank, and so the input vectors are linearly independent"
        )
        return None
    else:
        # Return non-pivot columns
        return np.delete(input_vectors, pivot_indices, axis=0)


class MyMatrix:
    """
    A matrix utility class that provides a set of methods for matrix operations.

    Attributes
    ----------
    X (np.ndarray): A matrix.
    row (int): Number of rows.
    col (int): Number of columns.
    det (np.ndarray): The determinant of X, if it is square.
    """

    def __init__(
        self,
        X: Union[np.ndarray, List[List[Any]], Tuple[Tuple[Any]]],
        axis: Optional[int] = None,
    ) -> None:
        """
        Initialize the MyMatrix instance.

        Parameters
        ----------
        X : Union[np.ndarray, List[List[Any]], Tuple[Tuple[Any]]]
            A matrix or a sequence that represents the matrix.
        axis : Optional[int]
            The axis along which the sequence is stacked into a matrix. Required if X is a sequence.

        Returns
        -------
        None
        """
        if isinstance(X, np.ndarray):
            self._init_from_ndarray(X)
        elif isinstance(X, (list, tuple)):
            if axis is None:
                raise ValueError(
                    "The argument 'axis' must be provided when input is a sequence."
                )
            self._init_from_seq(X, axis)
        else:
            raise TypeError(
                "Input must be a numpy.ndarray or a sequence (list or tuple)."
            )

        self.row = self.X.shape[0]
        self.col = self.X.shape[1]
        self.det = la.det(self.X) if self.is_sq() else None

    def _init_from_ndarray(self, X: np.ndarray) -> None:
        """
        Initialize from an ndarray.

        Parameters
        ----------
        X : np.ndarray
            A matrix.

        Returns
        -------
        None
        """
        self.X = X

    def _init_from_seq(
        self, X: Union[List[List[Any]], Tuple[Tuple[Any]]], axis: int
    ) -> None:
        """
        Initialize from a sequence of lists or tuples.

        Parameters
        ----------
        X : Union[List[List[Any]], Tuple[Tuple[Any]]]
            Input vectors.
        axis : int
            Whether to process input elements as row vectors or column vectors.

        Returns
        -------
        None
        """
        if axis == 1:
            self.X = np.stack(X, axis=1)
        elif axis == 0:
            self.X = np.array(X)
        else:
            raise ValueError("The argument 'axis' must either be 1 or 0")

    @classmethod
    def zeros(
        cls, shape: Tuple[int], dtype: Optional[np.dtype] = np.dtype(np.float64)
    ) -> Self:
        """
        Instantiate a class instance with a zero matrix.

        Parameters
        ----------
        shape : Tuple[int]
            Shape of the zero matrix, e.g. (2, 4).
        dtype : Optional[np.dtype]
            The desired data-type for the matrix, by default np.float64.

        Returns
        -------
        MyMatrix
            An instance of of class MyMatrix.

        Examples
        --------
        >>> import linalg as la
        >>> la.MyMatrix.zeros((2, 3))
        array([[0., 0., 0.],
               [0., 0., 0.]])
        """
        return cls(np.zeros(shape=shape, dtype=dtype))

    @classmethod
    def eye(
        cls,
        N: int,
        M: Optional[int] = None,
        k: int = 0,
        dtype: Optional[np.dtype] = np.dtype(np.int8),
    ) -> Self:
        """
        Instantiate a class instance with an identity matrix.

        Parameters
        ----------
        N : int
            Number of rows.
        M : Optional[int]
            Number of columns, by default None, which takes the value of N.
        k : int
            Index of the diagonal where 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal, and a negative value to a
            lower diagonal, by default 0.
        dtype : Optional[np.dtype]
            Data-type of identity matrix, by default np.int8.

        Returns
        -------
        MyMatrix
            An instance of of class MyMatrix.

        Examples
        --------
        >>> import linalg as la
        >>> la.MyMatrix.eye(3)
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]], dtype=int8)
        >>> la.MyMatrix.eye(3, 4, 1)
        array([[0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], dtype=int8)
        """
        return cls(np.eye(N=N, M=M, k=k, dtype=dtype))

    def __repr__(self) -> str:
        """
        Delegate to numpy's representation.
        """
        return repr(self.X)

    # ------------------------------ Infix Operators ----------------------------- #

    def __add__(self, other: Any) -> "MyMatrix":
        if isinstance(other, MyMatrix):
            X = self.X + other.X
            return MyMatrix(X)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other: Any) -> "MyMatrix":
        if isinstance(other, MyMatrix):
            X = self.X - other.X
            return MyMatrix(X)
        return NotImplemented

    def __rsub__(self, other):
        return -1 * (self - other)

    def __matmul__(self, other: Any) -> "MyMatrix":
        if isinstance(other, MyMatrix):
            X = self.X @ other.X
            return MyMatrix(X)
        return NotImplemented

    def __rmatmul__(self, other):
        return self @ other

    def __mul__(self, other: Any) -> "MyMatrix":
        if isinstance(other, (int, float)):
            X = self.X * other
            return MyMatrix(X)
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    # --------------------------- Comparison Operators --------------------------- #

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MyMatrix):
            return np.array_equal(self.X, other.X)
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        if isinstance(other, MyMatrix):
            return not self.__eq__(other)
        return NotImplemented

    def __gt__(self, other: Any) -> np.ndarray:
        if isinstance(other, MyMatrix):
            return self.X > other.X
        return NotImplemented

    def __lt__(self, other: Any) -> np.ndarray:
        if isinstance(other, MyMatrix):
            return self.X < other.X
        return NotImplemented

    def __ge__(self, other: Any) -> np.ndarray:
        if isinstance(other, MyMatrix):
            return self.X >= other.X
        return NotImplemented

    def __le__(self, other: Any) -> np.ndarray:
        if isinstance(other, MyMatrix):
            return self.X <= other.X
        return NotImplemented

    @staticmethod
    def cob_mat(
        original_basis: List[List], new_basis: List[List], inv: bool = False
    ) -> np.ndarray:
        """
        Find the change of basis matrix `S` from `original_basis` to `new_basis`. The change of basis
        matrix from basis A to basis B is defined to be: `S_{A -> B} = [ [a_1]_B, ..., [a_n]_B ]`.

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

        Examples
        --------
        >>> import linalg as la
        >>> # Change of basis from [[1, 2, -3], [4, -1, -3]] to [[0, 1, -1], [1, 0, -1]]
        >>> la.MyMatrix.cob_mat([[1, 2, -3], [4, -1, -3]], [[0, 1, -1], [1, 0, -1]], False)
        array([[ 2., -1.],
               [ 1.,  4.]])
        """
        # Matrix with new basis vectors as columns
        mat_new_basis = np.stack(new_basis, axis=1)

        # Convert each element (vector) in 'original_basis' to np.ndarray
        original_basis_arrays = [np.array(vec) for vec in original_basis]

        # Find coordinate vectors of original basis vectors with respect to new basis vectors
        list_of_coord_vectors = [
            la.lstsq(mat_new_basis, vec, rcond=None)[0] for vec in original_basis_arrays
        ]

        # Concatenate coordinate vectors (np.ndarray) to form change of basis vectors
        if inv:
            cob = np.stack(list_of_coord_vectors, axis=1)
            return la.inv(cob)
        else:
            return np.stack(list_of_coord_vectors, axis=1)

    @staticmethod
    def is_coord_vec(
        original_vec: Union[np.ndarray, List, Tuple],
        coord_vec: Union[np.ndarray, List, Tuple],
        *args: Union[np.ndarray, List, Tuple],
    ) -> bool:
        """
        Test if `coord_vec` is the coordinate vector with respect to a set of basis vectors `basis_vecs`.

        Parameters
        ----------
        original_vec : Union[np.ndarray, List, Tuple]
            The original vector in its original coordinate.
        coord_vec : Union[np.ndarray, List, Tuple]
            A coordinate vector.
        args : Union[np.ndarray, List, Tuple]
            An arbitrary number of basis vectors.

        Returns
        -------
        bool
            Whether `coord_vec` is the coordinate vector with respect to the basis vectors.

        Examples
        --------
        >>> import linalg as la
        >>> original_vec = np.array([5, 11])
        >>> coord_vec = [1, 2]
        >>> basis_vecs = [np.array([1, 3]), np.array([2, 4])]
        >>> la.MyMatrix.is_coord_vec(original_vec, coord_vec, *basis_vecs)
        True
        """
        sum_vec = np.zeros_like(original_vec)
        # Sum the basis vectors scaled by their corresponding coefficients in the coordinate vector
        for basis_vec, coef in zip(args, coord_vec):
            sum_vec = np.add(sum_vec, coef * np.array(basis_vec))

        return np.allclose(sum_vec, np.array(original_vec))

    def is_sq(self) -> bool:
        """
        Check if matrix `X` is square.

        Returns
        -------
        bool
            True if square or False if non-square.

        Examples
        --------
        >>> import linalg as la
        >>> la.MyMatrix(np.array([[1, 2], [3, 4]]), axis=0).is_sq()
        True
        >>> la.MyMatrix(np.array([[1, 2], [3, 4], [5, 6]]), axis=0).is_sq()
        False
        """
        return self.row == self.col

    def is_pos_def(self) -> bool:
        """
        Check if matrix `X` is positive definite. A symmetric matrix A is positive definite if
        (and only if) all of its eigenvalues are positive. The matrix A is positive sem-definite
        if  (and only if) all of its eigenvalues are non-negative (positive or zero).

        Returns
        -------
        bool
            True if positive definite or False if non-positive definite.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix.eye(3)
        >>> m.is_pos_def()
        True
        """
        try:
            # Will raise an exception if matrix is not positive definite
            np.linalg.cholesky(self.X)
            return True
        except la.LinAlgError:
            return False

    # ------------------------------- Matrix power ------------------------------- #

    def power(self, n) -> "MyMatrix":
        """
        Raise a square matrix to the (integer) power n.

        Parameters
        ----------
        n : int
            The exponent can be any integer or long integer, positive, negative, or zero.

        Returns
        -------
        MyMatrix
            An instance of class MyMatrix.

        Raises
        ------
        LinAlgError
           For matrices that are not square or that (for negative powers) cannot be inverted numerically.

        Examples
        --------
        >>> import linalg as la
        >>> la.MyMatrix(np.array([[1, 2], [12, 17]]), axis=0).power(2)
        array([[ 25,  36],
               [216, 313]])
        """
        X = la.matrix_power(self.X, n)
        return MyMatrix(X)

    def rref(self, pivots: bool = False) -> Union['MyMatrix', Tuple['MyMatrix', tuple]]:
        """
        Return the reduced row-echelon form of the matrix. Use `pivots` to return
        the indices of pivot columns.

        Parameters
        ----------
        pivots : bool, optional
            Whether to return a tuple of pivot indices, by default False.

        Returns
        -------
        'MyMatrix' or tuple
            The reduced row-echelon from or a tuple of pivot column indices.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], axis=1)
        >>> m.rref()
        array([[1., 0., -1.],
               [0., 1.,  2.],
               [0., 0.,  0.]])
        >>> m.rref(pivots=True)
        (array([[1., 0., -1.],
                [0., 1.,  2.],
                [0., 0.,  0.]]), (0, 1))
        """
        if pivots:
            mat, piv = Matrix(self.X).rref(pivots=pivots)
            return MyMatrix(np.array(mat, dtype=np.float64)), piv
        else:
            mat = Matrix(self.X).rref(pivots=pivots)
            return MyMatrix(np.array(mat, dtype=np.float64))

    def col_space(self) -> List[np.ndarray]:
        """
        Returns a list of vectors (np.ndarray objects) that span the column-space or image of X.

        Returns
        -------
        List[np.ndarray]
            A list of columns vectors.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 2], [3, 4], [5, 6]], axis=0)
        >>> [array([[1.],
                    [3.],
                    [5.]]),
            array([[2.],
                    [4.],
                    [6.]])]
        """
        return [np.array(col, dtype=np.float64) for col in Matrix(self.X).columnspace()]

    def null_space(self) -> List[np.ndarray]:
        """
        Returns a list of vectors (np.ndarray objects) that span the null-space or kernel of X.

        Returns
        -------
        List[np.ndarray]
            A list of columns vectors.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 2], [3, 6]], axis=0)
        >>> m.null_space()
        >>> [array([[-2.],
                    [ 1.]])]
        """
        return [np.array(col, dtype=np.float64) for col in Matrix(self.X).nullspace()]

    def inv(self) -> "MyMatrix":
        """
        Given a square matrix `X`, return the matrix `Xinv` satisfying `dot(X, Xinv) = dot(Xinv, X) = eye(X.shape[0])`.

        Returns
        -------
        MyMatrix
            The inverse of X.

        Raises
        ------
        LinAlgError
            If `X` is not square or inversion fails.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 2], [3, 4]], axis=0)
        >>> m.inv()
        >>> [array([-2.,  1.])]
                   [ 1.5, -0.5]])
        """
        return MyMatrix(la.inv(self.X))

    def pinv(self) -> "MyMatrix":
        """
        Compute the (Moore-Penrose) pseudo-inverse of a matrix. Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD)
        and including all large singular values.

        Returns
        -------
        B : (..., N, M) MyMatrix
            The pseudo-inverse of `X`.

        Raises
        ------
        LinAlgError
            If the SVD computation does not converge.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 2], [3, 4], [5, 6]], axis=1)
        >>> m.pinv()
        array([[-1.33333333,  1.08333333],
            [-0.33333333,  0.33333333],
            [ 0.66666667, -0.41666667]])
        """
        return MyMatrix(la.pinv(a=self.X, rcond=1e-15, hermitian=False))

    def gs(self, ret_type: str = "matrix") -> Union[List[np.ndarray], "MyMatrix"]:
        """
        Create an orthogonal matrix (a set of orthonormal basis vectors)
        for matrix `X`.

        Parameters
        ----------
        ret_type : str, optional
            Whether to return a set of orthonormal basis vectors or an orthogonal matrix, by default 'matrix'.

        Returns
        -------
        np.ndarray or MyMatrix
            A set of orthonormal basis vectors or an orthogonal matrix, depending on the `ret_type` argument.

        Raises
        ------
        LinAlgError
            The matrix X must be square.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 0], [0, 1], [1, 1]], axis=0)
        >>> m.gs(ret_type="vector")
        [array([-0.70710678, -0.        , -0.70710678]), array([ 0.40824829, -0.81649658, -0.40824829])]
        >>> m.gs(ret_type="matrix")
        array([[-0.70710678,  0.40824829],
               [-0.        , -0.81649658],
               [-0.70710678, -0.40824829]])
        """
        if not self.is_sq():
            raise la.LinAlgError("The matrix X is not square")

        if ret_type == "vector":
            # Create orthogonal matrix
            orthogonal_mat = la.qr(self.X)[0]
            # Return a list of orthonormal basis vectors
            orthonormal_vectors = [
                orthogonal_mat[:, col_index] for col_index in range(self.col)
            ]
            return orthonormal_vectors
        elif ret_type == "matrix":
            return MyMatrix(la.qr(self.X)[0])
        else:
            raise ValueError(
                "The argument 'ret_type' must either be 'matrix' or 'vector'"
            )

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
        LinAlgError
            The matrix X must be square.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 2], [2, 1]], axis=0)
        >>> m.eigen()
        (array([ 3., -1.]),
         array([[ 0.70710678, -0.70710678],
                [ 0.70710678,  0.70710678]]),
         array([[ 0.70710678, -0.70710678],
                [ 0.70710678,  0.70710678]]))
        """
        if not self.is_sq():
            raise la.LinAlgError("The matrix X is not square")

        # Compute
        eig_vals, left_eig_vecs, right_eig_vecs = eig(self.X, left=True, right=True)

        if real:
            return eig_vals.real, left_eig_vecs, right_eig_vecs
        else:
            return eig_vals, left_eig_vecs, right_eig_vecs

    def char_poly(self):
        """
        Computes characteristic polynomial det(lambda*I - M) where I is
        the identity matrix.

        Returns
        -------
        sympy expression
            A class instance inheriting from the Base class for algebraic expressions `Expr`.

        Raises
        ------
        LinAlgError
            The matrix X must be square.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 2], [3, 4]], axis=0)
        >>> m.char_poly()
        lambda**2 - 5*lambda - 2
        """
        if not self.is_sq():
            raise la.LinAlgError("The matrix X is not square")

        return Matrix(self.X).charpoly().as_expr()

    def diagonalize(
        self, reals_only: bool = False, sort: bool = False, normalize: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (P, D), where D is a diagonal D = P^-1 * M * P where M is current matrix.

        Parameters
        ----------
        reals_only : bool, optional
            Whether to throw an error if complex numbers are need to diagonalize, by default False.
        sort : bool, optional
            Whether to sort the eigenvalues along the diagonal, by default False.
        normalize : bool, optional
            Whether to normalize the eigenvector columns of P, by default False.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple (P, D).

        Raises
        ------
        LinAlgError
            The matrix X must be square.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 2], [2, 1]], axis=1)
        >>> m.diagonalize()
        (array([[-1.,  1.],
                [ 1.,  1.]]),
         array([[-1.,  0.],
                [ 0.,  3.]]))
        """
        if not self.is_sq():
            raise la.LinAlgError("The matrix X is not square")

        P, D = Matrix(self.X).diagonalize(
            reals_only=reals_only, sort=sort, normalize=normalize
        )
        return np.array(P, dtype=np.float64), np.array(D, dtype=np.float64)

    # ----------------------- Singular value decomposition ----------------------- #

    def svd(
        self, full_matrices: Optional[bool] = True, sigma_only: Optional[bool] = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Factorizes the matrix X into two unitary matrices U and Vh, and a 1-D array s of singular values (real, non-negative)
        such that a == U @ S @ Vh, where S is a suitably shaped matrix of zeros with main diagonal s.

        Parameters
        ----------
        full_matrices : Optional[bool]
            If True, u and vh have the shapes (..., M, M) and (..., N, N), respectively.
            Otherwise, the shapes are (..., M, K) and (..., K, N), respectively, where
            K = min(M, N)., by default True.
        sigma_only : Optional[bool]
            Whether to return the singular values only, by default False, which constructs
            the sigma matrix in SVD from the singular values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple of three matrices: U (m x m), S (m x n), V^T (n x n).

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 0], [71, 1], [12, 1]], axis=1)
        >>> m.svd()
        (array([[-0.99987192, -0.01600465],
                [-0.01600465,  0.99987192]]),
         array([[72.02311126+0.j,  0.        +0.j,  0.        +0.j],
                [ 0.        +0.j,  0.81941679+0.j,  0.        +0.j]]),
         array([[-0.01388265, -0.98589063, -0.16681406],
                [-0.01953176, -0.16653093,  0.98584277],
                [-0.99971285,  0.01694429, -0.01694429]]))
        """
        if not isinstance(full_matrices, bool):
            raise ValueError("The argument 'full_matrices' must be a boolean")

        # The vector s contains the singular values of self.X arranged in descending order of size
        u, s, vh = la.svd(
            a=self.X, full_matrices=full_matrices, compute_uv=True, hermitian=False
        )

        if sigma_only:
            return u, s, vh

        # Number of singular values
        m_or_n = len(s)

        # Create a diagonal matrix with the singular values on the diagonal
        if m_or_n == self.row:
            # For m < n, this is a higher dimensional (domain R^n) vector space mapping to lower dimensional (co-domain R^m) vector space
            # The zero matrix (m, n - m) is concatenated to diag(s) along last axis '-1' (column-bind)
            s_matrix = np.r_[
                "-1",
                np.diag(s),
                np.zeros((self.row, self.col - self.row), dtype=complex),
            ]
            return u, s_matrix, vh
        else:
            # For m > n, this is a lower dimensional (domain R^n) vector space mapping to higher dimensional (co-domain R^m) vector space
            # The zero matrix (m - n, n) is stacked below diag(s) along the first axis '0' (row-bind)
            s_matrix = np.r_[
                np.diag(s), np.zeros((self.row - self.col, self.col), dtype=complex)
            ]
            return u, s_matrix, vh

    # ------------------------ Rank-k matrix approximation ----------------------- #

    def approx_rank_k(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the rank-k approximation of the matrix X. The matrix approximation can be
        constructed as the matrix product `u @ np.diag(s) @ vh` where (u, s, vh) are the
        tuple returned by calling this method.

        Parameters
        ----------
        k : int
            The rank of the approximation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The singular values, left, right singular vectors needed to construct the rank-k approximation of the matrix X.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 0, 12], [7, 12, 17], [9, 77, 27], [8, 7, 16]], axis=0)
        >>> m.approx_rank_k(1)
        (array([[-0.05593827, -0.49093673],
                [-0.2166436 , -0.55228019],
                [-0.96140112,  0.25677546],
                [-0.16013852, -0.62292382]]),
         array([85.19359971, 21.49336595]),
         array([[-0.13505899, -0.91261056, -0.38587696],
                [-0.32704554,  0.40867874, -0.85206978]]))
        """
        # Compute the singular values
        u, s, vh = self.svd(sigma_only=True)

        # Return the necessary values and vectors to construct the rank-k approximation
        # The first k left and right singular vectors
        # And also the first k singular values, sorted in descending order
        return u[:, :k], s[:k], vh[:k, :]

        # ------------------ Randomized rank-k matrix approximation ------------------ #

    def rapprox_rank_k(
        self,
        k: int,
        n_oversamples: Optional[int] = None,
        n_iters: Optional[int] = None,
        return_onb: Optional[bool] = True,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Computes the rank-k approximation of the matrix X using randomized SVD. The matrix approximation can be
        constructed as the matrix product `u @ np.diag(s) @ vh` where (u, s, vh) are the tuple returned by calling this method.

        Parameters
        ----------
        k : int
            The rank of the approximation.
        n_oversamples : Optional[int]
            Additional number of random vectors to sample the column space of X so as to ensure proper conditioning.
        n_iters : Optional[int]
            Number of power iterations.
        return_onb : Optional[bool]
            Whether to return the orthonormal basis Q for the approximated column space of X.

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            The singular values, left, right singular vectors needed to construct the rank-k approximation of the matrix X.
            Optionally, the orthonormal basis Q for the approximated column space of X can be returned.

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 0, 12], [7, 12, 17], [9, 77, 27], [8, 7, 16]], axis=0)
        >>> m.rapprox_rank_k(2)
        (array([[-0.05593827,  0.49093673],
                [-0.2166436 ,  0.55228019],
                [-0.96140112, -0.25677546],
                [-0.16013852,  0.62292382]]),
         array([85.19359971, 21.49336595]),
         array([[-0.13505899, -0.91261056, -0.38587696],
                [ 0.32704554, -0.40867874,  0.85206978]]),
         array([[-0.58506352, -0.35851766, -0.70637316,  0.17378931],
                [-0.31208547, -0.44629115,  0.29152198, -0.78641071],
                [ 0.67509565, -0.71187709, -0.18086095,  0.06903766],
                [-0.3233407 , -0.40684188,  0.61914555,  0.58871833]]))
        """
        if n_oversamples is None:
            # If no oversampling parameter is specified, use the default value
            n_samples = 2 * k
        else:
            n_samples = k + n_oversamples

        # Random projection matrix P (sampled from the column space of X)
        P = np.random.randn(self.col, n_samples)
        Z = self.X @ P

        # If number of power iterations is specified
        if isinstance(n_iters, int):
            for _ in range(n_iters):
                Z = self.X @ (self.X.T @ Z)
            Q, R = la.qr(Z)
        else:
            # If no power iteration, simply find the orthonormal basis of Z
            Q, R = la.qr(Z)

        # Compute SVD on projected low-rank Y = Q.T @ self.X
        Y = Q.T @ self.X
        U_tilde, S, Vt = la.svd(Y)
        U = Q @ U_tilde

        # Q is useful for computing the actual error of approximation
        if return_onb:
            return U[:, :k], S[:k], Vt[:k, :], Q
        return U[:, :k], S[:k], Vt[:k, :]

        # --------------------------- Singular value plots --------------------------- #

    def sv_plot(self) -> None:
        """
        Plot the log and cumulative sum of singular values of the matrix X. This can be
        used to visually assess how much information is captured by the first k-rank of
        the matrix X, so that a sensible number can be selected for `k` for rank-k matrix
        approximation.

        Returns
        -------
        None

        Examples
        --------
        >>> import linalg as la
        >>> m = la.MyMatrix([[1, 0, 12], [7, 12, 17], [9, 77, 27], [8, 7, 16]], axis=0)
        >>> m.sv_plot()
        """
        s = svdvals(self.X)

        # Plot of the log of singular values
        plt.figure(num="Log of Singular Values")
        plt.semilogy(s)
        plt.title("Log of Singular Values")
        plt.xlabel("Rank")
        plt.show()

        # Plot the cumulative sum of the singular values
        plt.figure(num="Singular Values: Cumulative Sum")
        plt.plot(np.cumsum(s) / np.sum(s))
        plt.title("Singular Values: Cumulative Sum")
        plt.xlabel("Rank")
        plt.show()
