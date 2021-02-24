"""Python implementation of the symmetric rank-1 up- and downdate algorithms from
sections 2 and 3 in [1]_.

References
----------
.. [1] M. Seeger, "Low Rank Updates for the Cholesky Decomposition", 2008.
"""

import numpy as np
import scipy.linalg


def update(L: np.ndarray, v: np.ndarray) -> None:
    """Python implementation of the rank-1 update algorithm from section 2 in [1]_.

    Warning: The validity of the arguments will not be checked by this method, so
    passing invalid argument will result in undefined behavior.

    Parameters
    ----------
    L : (N, N) numpy.ndarray, dtype=numpy.double
        The lower-triangular Cholesky factor of the matrix to be updated.
        Must have shape `(N, N)` and dtype `np.double`.
        Must not contain zeros on the diagonal.
        The entries in the strict upper triangular part of :code:`L` can contain
        arbitrary values, since the algorithm neither reads from nor writes to this part
        of the matrix
        Will be overridden with the Cholesky factor of the matrix to be updated.
    v : (N,) numpy.ndarray, dtype=numpy.double
        The vector :math:`v` with shape :code:`(N, N)` and dtype :class:`numpy.double`
        defining the symmetric rank-1 update :math:`v v^T`.
        Will be reused as an internal memory buffer to store intermediate results, and
        thus modified.

    References
    ----------
    .. [1] M. Seeger, "Low Rank Updates for the Cholesky Decomposition", 2008.
    """

    N = L.shape[0]

    # Generate a contiguous view of the underling memory buffer of L, emulating raw
    # pointer access
    L_buf = L.ravel(order="K")

    assert np.may_share_memory(L, L_buf)

    if L.flags.f_contiguous:
        # In column-major memory layout, moving to the next row means moving the pointer
        # by 1 entry, while moving to the next column means moving the pointer by N
        # entries, i.e. the number of entries per column
        row_inc = 1
        column_inc = N
    else:
        assert L.flags.c_contiguous

        # In row-major memory layout, moving to the next column means moving the pointer
        # by 1 entry, while moving to the next row means moving the pointer by N
        # entries, i.e. the number of entries per row
        row_inc = N
        column_inc = 1

    # Create a "pointer" into the contiguous view of L's memory buffer
    # Points to the k-th diagonal entry of L at the beginning of the loop body
    L_buf_off = 0

    for k in range(N):
        # At this point L/L_buf contains a lower triangular matrix and the first k
        # entries of v are zeros

        # Generate Givens rotation which eliminates the k-th entry of v by rotating onto
        # the k-th diagonal entry of L and apply it only to these entries of (L|v)
        # Note: The following two operations will be performed by a single call to
        # `drotg` in C/Fortran. However, Python can not modify `Float` arguments.
        c, s = scipy.linalg.blas.drotg(L_buf[L_buf_off], v[k])
        L_buf[L_buf_off], v[k] = scipy.linalg.blas.drot(L_buf[L_buf_off], v[k], c, s)

        # Givens rotations generated by BLAS' `drotg` might rotate the diagonal entry to
        # a negative value. However, by convention, the diagonal entries of a Cholesky
        # factor are positive. As a remedy, we add another 180 degree rotation to the
        # Givens rotation matrix. This flips the sign of the diagonal entry while
        # ensuring that the resulting transformation is still a Givens rotation.
        if L_buf[L_buf_off] < 0.0:
            L_buf[L_buf_off] = -L_buf[L_buf_off]
            c = -c
            s = -s

        # Apply (modified) Givens rotation to the remaining entries in the k-th column
        # of L and the remaining entries in v

        # The first k entries in the k-th column of L are zero, since L is lower
        # triangular, so we only need to consider indices larger than or equal to i
        i = k + 1

        if i < N:
            # Move the pointer to the entry of L at index (i, k)
            L_buf_off += row_inc

            scipy.linalg.blas.drot(
                # We only need to rotate the last N - i entries
                n=N - i,
                # This constructs the memory adresses of the last N - i entries of the
                # k-th column in L
                x=L_buf,
                offx=L_buf_off,
                incx=row_inc,
                # This constructs the memory adresses of the last N - i entries of v
                y=v,
                offy=i,
                incy=1,
                c=c,
                s=s,
                overwrite_x=True,
                overwrite_y=True,
            )

            # In the beginning of the next iteration, the buffer offset must point to
            # the (k + 1)-th diagonal entry of L
            L_buf_off += column_inc


def downdate(L: np.ndarray, v: np.ndarray) -> None:
    """Python implementation of the rank-1 downdate algorithm from section 3 in [1]_.

    Warning: The validity of the arguments will not be checked by this method, so
    passing invalid argument will result in undefined behavior.

    Parameters
    ----------
    L : (N, N) numpy.ndarray, dtype=numpy.double
        The lower-triangular Cholesky factor of the matrix to be downdated.
        Must have shape `(N, N)` and dtype `np.double`.
        Must not contain zeros on the diagonal.
        The entries in the strict upper triangular part of :code:`L` can contain
        arbitrary values, since the algorithm neither reads from nor writes to this part
        of the matrix
        Will be overridden with the Cholesky factor of the matrix to be downdated.
    v : (N,) numpy.ndarray, dtype=numpy.double
        The vector :math:`v` with shape :code:`(N, N)` and dtype :class:`numpy.double`
        defining the symmetric rank-1 downdate :math:`v v^T`.
        Will be reused as an internal memory buffer to store intermediate results, and
        thus modified.

    Raises
    ------
    ValueError
        If the memory layout of `L` is neither Fortan- nor C-contiguous.
    numpy.linalg.LinAlgError
        If the downdate results in a matrix that is not positive definite.

    Notes
    -----
    This method allocates an auxiliary buffer of shape :code:`(N,)` and dtype
    :class:`np.double`.

    References
    ----------
    .. [1] M. Seeger, "Low Rank Updates for the Cholesky Decomposition", 2008.
    """

    # pylint: disable=too-many-locals

    N = L.shape[0]

    # Compute p by solving L @ p = v
    if L.flags.f_contiguous:
        # Solve L @ p = v
        scipy.linalg.blas.dtrsv(
            a=L,
            x=v,
            overwrite_x=True,
            lower=1,
            trans=0,
            diag=0,
        )
    elif L.flags.c_contiguous:
        # Solve (L^T)^T @ p = v
        # This is necessary because `dtrsv` expects column-major matrices and a
        # row-major L buffer can be interpreted as a column-major L^T buffer
        scipy.linalg.blas.dtrsv(
            a=L.T,  # L.T is column-major
            x=v,
            overwrite_x=True,
            lower=0,
            trans=1,
            diag=0,
        )
    else:
        raise ValueError(
            "Unsupported memory layout. L should either be Fortran- or C-contiguous."
        )

    p = v  # `v` now contains p

    # Compute ρ = √(1 - p^T @ p)
    rho_sq = 1 - scipy.linalg.blas.ddot(p, p)

    if rho_sq <= 0.0:
        # The downdated matrix is positive definite if and only if rho ** 2 is positive
        raise np.linalg.LinAlgError("The downdated matrix is not positive definite.")

    rho = np.sqrt(rho_sq)

    # "Append" `rho` to `p` to form `q`
    q_1n = p  # contains q[:-1]
    q_np1 = rho  # contains q[-1]

    # "Append" a column of zeros to `L` to form the augmented matrix `L_aug`
    L_aug_cols_1n = L  # contains L_aug[:, :-1]
    L_aug_col_np1 = np.zeros(N, dtype=L.dtype)  # contains L_aug[:, -1]

    # We implement different versions of the algorithm for row- and column-major
    # Cholesky factors, since we found this to be faster in Python
    if L_aug_cols_1n.flags.f_contiguous:
        for k in range(N - 1, -1, -1):
            # Generate Givens rotation which eliminates the k-th entry of `q` with the
            # (n + 1)-th entry of `q` and apply it to q.
            # Note: The following two operations will be performed by a single call to
            # `drotg` in C/Fortran. However, Python can not modify `Float` arguments.
            c, s = scipy.linalg.blas.drotg(q_np1, q_1n[k])
            q_np1, q_1n[k] = scipy.linalg.blas.drot(q_np1, q_1n[k], c, s)

            # Givens rotations generated by BLAS' `drotg` might rotate `q_np1` to a
            # negative value. However, for the algorithm to work, it is important that
            # `q_np1` remains positive. As a remedy, we add another 180 degree rotation
            # to the Givens rotation matrix. This flips the sign of `q_np1` while
            # ensuring that the resulting transformation is still a Givens rotation.
            if q_np1 < 0.0:
                q_np1 = -q_np1
                c = -c
                s = -s

            # Apply the transpose of the (modified) Givens rotation matrix to the
            # augmented matrix `L_aug` from the right, i.e. compute L_aug @ Q_{c, s}^T
            scipy.linalg.blas.drot(
                x=L_aug_col_np1[k:],
                overwrite_x=True,
                y=L_aug_cols_1n[k:, k],
                overwrite_y=True,
                c=c,
                s=s,
            )

            # Applying the Givens rotation might lead to a negative diagonal element in
            # `L_aug`. However, by convention, the diagonal entries of a Cholesky factor
            # are positive. As a remedy, we simply rescale the whole row. Note that this
            # is possible, since rescaling a row is equivalent to a mirroring along one
            # dimension which is in turn an orthogonal transformation.
            if L_aug_cols_1n[k, k] < 0.0:
                L_aug_cols_1n[k:, k] = -L_aug_cols_1n[k:, k]
    elif L_aug_cols_1n.flags.c_contiguous:
        # Generate a contiguous view of the underling memory buffer of L, emulating raw
        # pointer access
        L_aug_cols_1n_buf = L_aug_cols_1n.ravel(order="K")

        assert np.may_share_memory(L, L_aug_cols_1n_buf)

        # In row-major memory layout, moving to the next column means moving the pointer
        # by 1 entry, while moving to the next row means moving the pointer by N
        # entries, i.e. the number of entries per row
        row_stride = N
        col_stride = 1

        for k in range(N - 1, -1, -1):
            # Generate Givens rotation which eliminates the k-th entry of `q` with the
            # (n + 1)-th entry of `q` and apply it to q.
            # Note: The following two operations will be performed by a single call to
            # `drotg` in C/Fortran. However, Python can not modify `Float` arguments.
            c, s = scipy.linalg.blas.drotg(q_np1, q_1n[k])
            q_np1, q_1n[k] = scipy.linalg.blas.drot(q_np1, q_1n[k], c, s)

            # Givens rotations generated by BLAS' `drotg` might rotate `q_np1` to a
            # negative value. However, for the algorithm to work, it is important that
            # `q_np1` remains positive. As a remedy, we add another 180 degree rotation
            # to the Givens rotation matrix. This flips the sign of `q_np1` while
            # ensuring that the resulting transformation is still a Givens rotation.
            if q_np1 < 0.0:
                q_np1 = -q_np1
                c = -c
                s = -s

            # Apply the transpose of the (modified) Givens rotation matrix to the
            # augmented matrix `L_aug` from the right, i.e. compute L_aug @ Q_{c, s}^T
            scipy.linalg.blas.drot(
                x=L_aug_col_np1[k:],
                overwrite_x=True,
                # The following 3 lines describe the memory adresses of the slice
                # `L_aug_cols_1n[k:, k]`
                y=L_aug_cols_1n_buf,
                offy=k * (row_stride + col_stride),
                incy=row_stride,
                overwrite_y=True,
                c=c,
                s=s,
            )

            # Applying the Givens rotation might lead to a negative diagonal element in
            # `L_aug`. However, by convention, the diagonal entries of a Cholesky factor
            # are positive. As a remedy, we simply flip the sign of the whole row. Note
            # that this is possible, since rescaling a row by -1.0 is equivalent to a
            # mirroring along one dimension which is in turn an orthogonal
            # transformation.
            if L_aug_cols_1n[k, k] < 0.0:
                L_aug_cols_1n[k:, k] = -L_aug_cols_1n[k:, k]