"""Python adaptations of some useful Matlab functions."""

from typing import Tuple

import numpy as np
import scipy.linalg as la
from scipy.linalg.lapack import dlartg


def chol(a: np.ndarray, lower: bool = False) -> Tuple[np.ndarray, int]:
    """Cholesky factorization inspired by Matlab's chol function

    Uses LAPACK's POTRF subroutine to compute the factorization.

    Parameters
    ----------
    A : (M, M) ndarray
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper- or lower-triangular Cholesky
        factorization.  Default is upper-triangular.

    Returns
    -------
    c : (M, M) ndarray
        Upper- or lower-triangular Cholesky factor of a.
    info : int
        Status information on the POTRF call:
            * -i < 0 : the i-th argument had an illegal value.
            *    = 0 : successful execution.
            *  i > 0 : the leading minor of order i is not positive definite,
                         and the factorization could not be completed.
    """
    a1 = np.atleast_2d(np.asarray_chkfinite(a))

    # Dimension check
    if a1.ndim != 2:
        raise ValueError('Input array needs to be 2D but received '
                         f'a {a1.ndim}d-array.')
    # Squareness check
    if a1.shape[0] != a1.shape[1]:
        raise ValueError('Input array is expected to be square but has '
                         f'the shape: {a1.shape}.')

    # Quick return for square empty array
    if a1.size == 0:
        raise ValueError("Empty input array.")

    potrf, = la.lapack.get_lapack_funcs(('potrf',), (a1,))
    c, info = potrf(a1, lower=lower, overwrite_a=False, clean=True)

    return c, info


def planerot(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Givens plane rotation matrix G and the transformed vector y

    Uses LAPACK's DLARTG to compute the result.

    Parameters
    ----------
    x : ndarray
        2-component vector on which to apply the transformation

    Returns
    -------
    G : ndarray
        2 x 2 Givens plane rotation matrix
    y : ndarray
        2-component transformed vector y = G @ x
    """
    x = np.asarray_chkfinite(x)

    if x.shape != (2,):
        raise ValueError(
            "Input x must be a one-dimensional 2 element array")

    cs, sn, r = dlartg(*x)

    G = np.array([[cs, sn], [-sn, cs]])
    y = np.array([r, 0])

    return G, y
