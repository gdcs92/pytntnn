"""
Non-negative Least Squares feasible solution.

Derived from Erich Frahm and Joseph Myre's Matlab implementation
available at https://github.com/ProfMyre/tnt-nn.
"""
from typing import List
from dataclasses import dataclass

import numpy as np

from .matlab_functions import chol, planerot
from .pcgnr import pcgnr


@dataclass
class LSQResult:
    score: float
    x: np.ndarray
    residual: float
    free_set: List[int]
    binding_set: List[int]
    AA: np.ndarray
    epsilon: float
    del_hist: List[int]
    dels: int
    loops: int
    lsq_loops: int


def lsq_solve(A: np.ndarray, b: np.ndarray, lmbda: float, AA: np.ndarray,
              epsilon: float, free_set: List[int], binding_set: List[int],
              deletions_per_loop: int) -> LSQResult:
    """Computes a feasible solution non-negative least-squares problem.

    Uses an active-set strategy to handle the non-negativity constraints,
    combined with a preconditioned conjugate gradient solver for the
    unconstrained least-squares problem.
    """
    free_set = sorted(free_set, reverse=True)
    binding_set = sorted(binding_set, reverse=True)

    # Reduce A to B.
    # B is a matrix that has all of the rows of A, but its
    # columns are a subset of the columns of A. The free_set
    # provides a map from the columns of B to the columns of A.
    B = A[:, free_set].copy()

    # Reduce AA to BB.
    # BB is a symmetric matrix that has a subset of rows and
    # columns of AA. The free_set provides a map from the rows
    # and columns of BB to rows and columns of AA.
    BB = AA[np.ix_(free_set, free_set)].copy()

    # Adjust with Tikhonov regularization parameter lmbda.
    if lmbda > 0:
        for i, _ in enumerate(free_set):
            B[i, i] += lmbda
            BB[i, i] += lmbda**2

    # Cholesky decomposition
    n = AA.shape[0]
    R, p = chol(BB)
    while p > 0:
        # It may be necessary to add to the diagonal of B'B to avoid
        # taking the sqare root of a negative number when there are
        # rounding errors on a nearly singular matrix. That's still OK
        # because we just use the Cholesky factor as a preconditioner.
        epsilon = epsilon * 10

        AA = AA + (epsilon * np.eye(n))
        BB = AA[np.ix_(free_set, free_set)].copy()

        if lmbda > 0:
            for i, _ in enumerate(free_set):
                BB[i,i] += lmbda**2

        R, p = chol(BB)

    # Loops until solution is feasible
    dels = 0
    loops = 0
    lsq_loops = 0
    del_hist = []

    while True:
        loops += 1

        # Use PCGNR to find the unconstrained optimum in the "free" variables.
        reduced_x, k = pcgnr(B, b, R)

        lsq_loops = max(k, lsq_loops)

        # Get a list of variables that must be deleted
        deletion_set = [i for i,_ in enumerate(free_set) if reduced_x[i] <= 0]

        # If the current solution is feasible then quit.
        if not deletion_set:
            break

        # Sort the possible deletions by their reduced_x values to
        # find the worst violators.
        x_score = reduced_x[deletion_set]
        set_index = np.argsort(x_score)
        deletion_set = [deletion_set[i] for i in set_index]

        # Limit the number of deletions per loop
        if len(deletion_set) > deletions_per_loop:
            deletion_set = deletion_set[:deletions_per_loop]

        deletion_set = sorted(deletion_set, reverse=True)
        del_hist.extend(deletion_set)
        dels += len(deletion_set)

        # Move the variables from "free" to "binding".
        # bound_variables = free_set[deletion_set]
        bound_variables = [free_set[i] for i in deletion_set]
        binding_set.extend(bound_variables)
        free_set = [j for j in free_set if j not in bound_variables]

        # Reduce A to B
        # B is a matrix that has all of the rows of A, but its
        # columns are a subset of the columns of A. The free_set
        # provides a map from the columns of B to the columns of A.
        B = A[:, free_set].copy()

        # Reduce AA to BB.
        # BB is a symmetric matrix that has a subset of rows and
        # columns of AA. The free_set provides a map from the rows
        # and columns of BB to rows and columns of AA.
        BB = AA[np.ix_(free_set, free_set)].copy()

        # Adjust with Tikhonov regularization parameter lambda
        if lmbda > 0:
            for i, _ in enumerate(free_set):
                B[i, i] += lmbda
                BB[i, i] += lmbda**2

        # Compute R, the Cholesky factor.
        R = _cholesky_delete(R, BB, deletion_set)

    # Unscramble the column indices to get the full (unreduced) x.
    n = A.shape[1]
    x = np.zeros(n)
    x[free_set] = reduced_x

    # Compute the full (unreduced) residual.
    residual = b - (A @ x)

    # Compute the norm of the residual.
    score = np.sqrt(np.dot(residual,residual))

    return LSQResult(score, x, residual, free_set, binding_set, AA,
                           epsilon, del_hist, dels, loops, lsq_loops)


def _cholesky_delete(R: np.ndarray, BB: np.ndarray,
                    deletion_set: List[int]) -> np.ndarray:
    n = R.shape[1]
    num_deletions = len(deletion_set)

    speed_fudge_factor = 0.001
    if num_deletions > speed_fudge_factor * n:
        R, p = chol(BB)
        if p > 0:
            # This should never happen because we have already added
            # a sufficiently large "epsilon" to AA to do the
            # nonnegativity tests required to create the deleted_set.
            raise ValueError(
                f"Could not compute Cholesky factorization ({p = })"
            )
    else:

        for i in range(num_deletions):
            j = deletion_set[i]

            # This function is just a stripped version of Matlab's qrdelete.
            # http://pmtksupport.googlecode.com/svn/trunk/lars/larsen.m

            R = np.delete(R, j, axis=1)
            n = R.shape[1]

            for k in range(j, n):

                p = [k, k+1]
                G, r = planerot(R[p, k])  # remove extra element in col
                R[p, k] = r

                if k < n-1:  # adjust rest of row
                    R[p, k+1:] = G @ R[p, k+1:]

            R = np.delete(R, -1, axis=0)

    return R
