# ---------------------------------------------------------------------------- #
#                              Package and Modules                             #
# ---------------------------------------------------------------------------- #

import numpy as np
import numpy.linalg as la
from scipy.linalg import eig, svdvals
from sympy import Matrix
from sympy.core.expr import Expr
import matplotlib.pyplot as plt

# ----------------------------- Standard library ----------------------------- #

from typing import Union, List, Tuple, Optional

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
        print("The matrix with input vectors as columns has full column rank, and so the input vectors are linearly independent")
        return None
    else:
        # Return non-pivot columns
        return np.delete(args, pivot_indices, axis=0)

# ------------------------------ MyMatrix class ------------------------------ #


class MyMatrix:
    """
    A utility class for matrices.

    Attributes:
            X (np.ndarray): A matrix.
            row (int): Number of rows.
            col (int): Number of columns.
            det (np.ndarray): The determinant of X, if it is square.
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
        # If square, compute determinant
        if (self.row == self.col):
            self.det = la.det(self.X)
        else:
            self.det = None

    @classmethod
    def default(cls, X: Union[List[List], Tuple[Tuple]], axis: int = 1) -> 'MyMatrix':
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
    def from_numpy(cls, X: np.ndarray) -> 'MyMatrix':
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
    def zeros(cls, shape: Tuple[int], dtype: np.dtype = np.float64) -> 'MyMatrix':
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
    def eye(cls, N: int, M: int = None, k: int = 0, dtype: np.dtype = np.int8) -> 'MyMatrix':
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

    # --------------------------------- Printing --------------------------------- #

    def __repr__(self) -> str:
        return repr(self.X)

    # ---------------------------------------------------------------------------- #
    #                             Overloading operators                            #
    # ---------------------------------------------------------------------------- #

    # ------------------------------ Infix Operators ----------------------------- #

    def __add__(self, other: 'MyMatrix') -> 'MyMatrix':
        try:
            X = self.X.copy().__add__(other.X.copy())
            return MyMatrix.from_numpy(X)
        # Catch AttributesError since not all objects have 'X' attribute
        # ValueError may need to be caught as well, depending on the type of 'other'
        # Other exceptions may slip through
        except (TypeError, AttributeError, ValueError):
            return NotImplemented

    # If the left operand is not a MyMatrix instance, then invoke the __add__ method with the operands swapped
    def __radd__(self, other):
        return self + other

    def __sub__(self, other: 'MyMatrix') -> 'MyMatrix':
        try:
            X = self.X.copy().__sub__(other.X.copy())
            return MyMatrix.from_numpy(X)
        except (TypeError, AttributeError, ValueError):
            return NotImplemented

    def __rsub__(self, other):
        return self - other

    def __matmul__(self, other: 'MyMatrix') -> 'MyMatrix':
        try:
            X = self.X.copy() @ other.X.copy()
            return MyMatrix.from_numpy(X)
        except (TypeError, AttributeError, ValueError):
            return NotImplemented

    def __rmatmul__(self, other):
        return self @ other

    # Left operand is an instance of MyMatrix, and right operand is a scalar
    def __mul__(self, other: Union[int, float]) -> 'MyMatrix':
        try:
            X = self.X.copy() * other
            return MyMatrix.from_numpy(X)
        except (TypeError, AttributeError, ValueError):
            return NotImplemented

    # If the left operand is not a MyMatrix instance, just invoke the __mul__ method with MyMatrix as the left operand
    def __rmul__(self, other):
        return self * other

    # --------------------------- Comparison Operators --------------------------- #

    def __eq__(self, other: 'MyMatrix') -> np.ndarray:
        try:
            return self.X == other.X  # Delegate to numpy
        except (TypeError, AttributeError, ValueError):
            return NotImplemented

    def __ne__(self, other: 'MyMatrix') -> np.ndarray:
        try:
            return self.X != other.X
        except (TypeError, AttributeError, ValueError):
            return NotImplemented

    def __gt__(self, other: 'MyMatrix') -> np.ndarray:
        try:
            return self.X > other.X
        except (TypeError, AttributeError, ValueError):
            return NotImplemented

    def __lt__(self, other: 'MyMatrix') -> np.ndarray:
        try:
            return self.X < other.X
        except (TypeError, AttributeError, ValueError):
            return NotImplemented

    def __ge__(self, other: 'MyMatrix') -> np.ndarray:
        try:
            return self.X >= other.X
        except (TypeError, AttributeError, ValueError):
            return NotImplemented

    def __le__(self, other: 'MyMatrix') -> np.ndarray:
        return self.X <= other.X

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

    # ----------- Test for coordinate vector with respect basis vectors ---------- #

    @staticmethod
    def is_coord_vec(original_vec: Union[np.ndarray, List, Tuple], coord_vec: Union[np.ndarray, List, Tuple], *args: Union[np.ndarray, List, Tuple]) -> bool:
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
        """
        sum_vec = np.zeros_like(original_vec)
        # Sum the basis vectors scaled by their corresponding coefficients in the coordinate vector
        for basis_vec, coef in zip(args, coord_vec):
            sum_vec = np.add(sum_vec, coef * np.array(basis_vec))

        return np.allclose(sum_vec, np.array(original_vec))

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

    # ------------------- Check if matrix is positive definite ------------------- #

    def is_pos_def(self) -> bool:
        """
        Check if matrix `X` is positive definite. A symmetric matrix A is positive definite if
        (and only if) all of its eigenvalues are positive. The matrix A is positive sem-definite
        if  (and only if) all of its eigenvalues are non-negative (positive or zero).

        Returns
        -------
        bool
            True if positive definite or False if non-positive definite.
        """
        try:
            # Will raise an exception if matrix is not positive definite
            np.linalg.cholesky(self.X)
            return True
        except la.LinAlgError:
            return False

    # ------------------------------- Matrix power ------------------------------- #

    def power(self, n) -> 'MyMatrix':
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
        """
        X = la.matrix_power(self.X.copy(), n)
        return MyMatrix.from_numpy(X)

    # ------------------------------- Compute rref ------------------------------- #

    def rref(self, pivots: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, tuple]]:
        """
        Return the reduced row-echelon form of the matrix. Use `pivots` to return
        the indices of pivot columns.

        Parameters
        ----------
        pivots : bool, optional
            Whether to return a tuple of pivot indices, by default False.

        Returns
        -------
        np.ndarray or tuple
            The reduced row-echelon from or a tuple of pivot column indices.
        """
        if pivots:
            mat, piv = Matrix(self.X).rref(pivots=pivots)
            return np.array(mat, dtype=np.float_), piv
        else:
            mat = Matrix(self.X).rref(pivots=pivots)
            return np.array(mat, dtype=np.float_)

    # -------------------- Column space (image) of the matrix -------------------- #

    def col_space(self) -> List[np.ndarray]:
        """
        Returns a list of vectors (np.ndarray objects) that span the column-space or image of X.

        Returns
        -------
        List[np.ndarray]
            A list of columns vectors.
        """
        return [np.array(col, dtype=np.float_)
                for col in Matrix(self.X).columnspace()]

    # --------------------- Null space (kernel) of the matrix -------------------- #

    def null_space(self) -> List[np.ndarray]:
        """
        Returns a list of vectors (np.ndarray objects) that span the null-space or kernel of X.

        Returns
        -------
        List[np.ndarray]
            A list of columns vectors.
        """
        return [np.array(col, dtype=np.float_)
                for col in Matrix(self.X).nullspace()]

    # ---------------------------------- Inverse --------------------------------- #

    def inv(self) -> 'MyMatrix':
        """
        Given a square matrix `X`, return the matrix `Xinv` satisfying `dot(X, Xinv) = dot(Xinv, X) = eye(X.shape[0])`.

        Returns
        -------
        MyMatrix
            The inverse of X.

        Raises
        ------
        la.LinAlgError
            If `X` is not square or inversion fails..
        """
        return MyMatrix.from_numpy(la.inv(self.X))

    # ----------------------- Moore-Penrose Pseudo-Inverse ----------------------- #

    def pinv(self) -> 'MyMatrix':
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

        Notes
        -----
        The pseudo-inverse of a matrix A, denoted :math:`A^+`, is
        defined as: "the matrix that 'solves' [the least-squares problem]
        :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
        :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.
        It can be shown that if :math:`Q_1 \\Sigma Q_2^T = A` is the singular
        value decomposition of A, then
        :math:`A^+ = Q_2 \\Sigma^+ Q_1^T`, where :math:`Q_{1,2}` are
        orthogonal matrices, :math:`\\Sigma` is a diagonal matrix consisting
        of A's so-called singular values, (followed, typically, by
        zeros), and then :math:`\\Sigma^+` is simply the diagonal matrix
        consisting of the reciprocals of A's singular values
        (again, followed by zeros). [1]_

        References
        ----------
        .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
            FL, Academic Press, Inc., 1980, pp. 139-142.
        """
        return MyMatrix.from_numpy(la.pinv(a=self.X, rcond=1e-15, hermitian=False))

    # ------------------------------- Gram-schmidt ------------------------------- #

    def gs(self, ret_type: str = 'matrix') -> Union[List[np.ndarray], 'MyMatrix']:
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
        """
        if (ret_type == 'vector'):
            # Create orthogonal matrix
            orthogonal_mat = la.qr(self.X)[0]

            # Return a list of orthonormal basis vectors
            orthonormal_vectors = [orthogonal_mat[:, col_index]
                                   for col_index in range(self.col)]
            return orthonormal_vectors
        elif (ret_type == 'matrix'):
            return MyMatrix.from_numpy(la.qr(self.X)[0])

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

    # ------------------------------ Diagonalization ----------------------------- #

    def diagonalize(self, reals_only: bool = False, sort: bool = False, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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
        """
        P, D = Matrix(self.X).diagonalize(
            reals_only=reals_only, sort=sort, normalize=normalize)
        return np.array(P, dtype=np.float_), np.array(D, dtype=np.float_)

    # ----------------------- Singular value decomposition ----------------------- #

    def svd(self, full_matrices: bool = True, sigma_only: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Factorizes the matrix X into two unitary matrices U and Vh, and a 1-D array s of singular values (real, non-negative)
        such that a == U @ S @ Vh, where S is a suitably shaped matrix of zeros with main diagonal s.

        Parameters
        ----------
        full_matrices : bool, optional
            If True, u and vh have the shapes (..., M, M) and (..., N, N), respectively.
            Otherwise, the shapes are (..., M, K) and (..., K, N), respectively, where
            K = min(M, N)., by default True.
        sigma_only : bool, optional
            Whether to return the singular values only, by default False, which constructs
            the sigma matrix in SVD from the singular values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple of three matrices: U (m x m), S (m x n), V^T (n x n).
        """
        # The vector s contains the singular values of self.X arranged in descending order of size
        u, s, vh = la.svd(
            a=self.X,
            full_matrices=full_matrices,
            compute_uv=True,
            hermitian=False
        )

        if sigma_only:
            return u, s, vh
        else:
            # Number of singular values
            m_or_n = len(s)

            # Create a diagonal matrix with the singular values on the diagonal
            if m_or_n == self.row:
                # For m < n, this is a higher dimensional (domain R^n) vector space mapping to lower dimensional (co-domain R^m) vector space
                # The zero matrix (m, n - m) is concatenated to diag(s) along last axis '-1' (column-bind)
                return u, np.r_['-1', np.diag(s), np.zeros((self.row, self.col - self.row), dtype=complex)], vh
            elif m_or_n == self.col:
                # For m > n, this is a lower dimensional (domain R^n) vector space mapping to higher dimensional (co-domain R^m) vector space
                # The zero matrix (m - n, n) is stacked below diag(s) along the first axis '0' (row-bind)
                return u, np.r_[np.diag(s), np.zeros((self.row - self.col, self.col), dtype=complex)]

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
        """
        # Compute the singular values
        u, s, vh = self.svd(sigma_only=True)

        # Return the necessary values and vectors to construct the rank-k approximation
        # The first k left and right singular vectors
        # And also the first k singular values, sorted in descending order
        return u[:, :k], s[:k], vh[:k, :]

        # ------------------ Randomized rank-k matrix approximation ------------------ #

    def rapprox_rank_k(self,
                       k: int,
                       n_oversamples: Optional[int] = None,
                       n_iters: Optional[int] = None,
                       return_onb: Optional[bool] = True) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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
            Q, R = la.linalg.qr(Z)
        else:
            # If no power iteration, simply find the orthonormal basis of Z
            Q, R = la.linalg.qr(Z)

        # Compute SVD on projected low-rank Y = Q.T @ self.X
        Y = Q.T @ self.X
        U_tilde, S, Vt = la.svd(Y)
        U = Q @ U_tilde

        # Q is useful for computing the actual error of approximation
        if return_onb:
            return U[:, :k], S[:k], Vt[:k, :], Q
        return U[:, :k], S[:k], Vt[:k, :]

        # --------------------------- Singular value plots --------------------------- #

    def sv_plot(self):
        """
        Plot the log and cumulative sum of singular values of the matrix X. This can be
        used to visually assess how much information is captured by the first k-rank of 
        the matrix X, so that a sensible number can be selected for `k` for rank-k matrix 
        approximation.
        """
        s = svdvals(self.X)

        # Plot of the log of singular values
        plt.figure(num='Log of Singular Values')
        plt.semilogy(s)
        plt.title('Log of Singular Values')
        plt.xlabel('Rank')
        plt.show()

        # Plot the cumulative sum of the singular values
        plt.figure(num='Singular Values: Cumulative Sum')
        plt.plot(np.cumsum(s) / np.sum(s))
        plt.title('Singular Values: Cumulative Sum')
        plt.xlabel('Rank')
        plt.show()
