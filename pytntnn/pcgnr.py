"""
Left-preconditioned CG solver for the normal equations system.

Derived from Erich Frahm and Joseph Myre's Matlab implementation,
available at https://github.com/ProfMyre/tnt-nn.
"""
from typing import Tuple

import numpy as np
import scipy.linalg as la


def pcgnr(A: np.ndarray, b: np.ndarray,
          R: np.ndarray) -> Tuple[np.ndarray, int]:
    """Left-Preconditioned CGNR.

    Solves the normal equations A^T A x = A^T b using a left-preconditioned
    conjugate gradient method. See Algorithm 9.7 from "Iterative Methods
    for Sparse Linear Systems" (2nd Ed.), by Yousef Saad.

    Parameters
    ----------
    A : (m, n) ndarray
        Left-hand side matrix
    b : (m,) ndarray
        Right-hand side vector
    R : (n, n) ndarray
        Left-preconditioner matrix

    Returns
    -------
    x : (n,) ndarray
        Solution of the normal equations system
    k : int
        Number of iterations
    """
    n = A.shape[1]
    x = np.zeros(n)
    r = b.copy()
    r_hat = A.T @ r
    y = la.solve_triangular(R, r_hat, trans='T', lower=False)
    z = la.solve_triangular(R, y, lower=False)
    p = z.copy()

    gamma = np.dot(z, r_hat)
    prev_rr = -1

    for k in range(1, n+1):
        w = A @ p
        ww = np.dot(w,w)

        if ww == 0:
            break

        alpha = gamma/ww
        x_prev = x.copy()
        x += alpha*p
        r = b - A @ x
        r_hat = A.T @ r

        # Enforce continuous improvement in the score.
        rr = np.dot(r_hat, r_hat)
        if 0 <= prev_rr <= rr:
            x = x_prev
            break
        prev_rr = rr

        y = la.solve_triangular(R, r_hat, trans='T', lower=False)
        z = la.solve_triangular(R, y, lower=False)
        gamma_new = np.dot(z, r_hat)
        beta = gamma_new/gamma
        p = z + beta*p
        gamma = gamma_new

        if gamma == 0:
            break

    return x, k
