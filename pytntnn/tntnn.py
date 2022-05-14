"""
Non-negative Least Squares solution using the TNT-NN active-set method.

Derived from Erich Frahm and Joseph Myre's Matlab implementation
available at https://github.com/ProfMyre/tnt-nn.
"""
from typing import Union, List
from dataclasses import dataclass
import math

import numpy as np
import scipy.linalg as la

from .lsq_solve import lsq_solve


@dataclass
class TNTNNResult:
    x: Union[np.ndarray, None]
    AA: Union[np.ndarray, None]
    OuterLoop: int
    TotalInnerLoops: int
    hist: Union[List[List[int]], None]


def tntnn(A: np.ndarray, b: np.ndarray,
          lmbda: float = 0,
          rel_tol: float = 0,
          AA: Union[np.ndarray, None] = None,
          use_AA: bool = False,
          red_c: float = 0.2,
          exp_c: float = 1.2,
          verbose: bool = False) -> TNTNNResult:

    hist = []

    m, n = A.shape

    if b.shape != (m,):
        raise ValueError(
            "A must have the same number of rows as the dimension of b."
            f" Got {A.shape = }, {b.shape = }.")

    if use_AA:
        if AA.shape != (n, n):
            raise ValueError(
                "When use_AA = True, AA.shape must equal (n, n),"
                " where n is the number of columns of A")
    else:
        AA = A.T @ A

    # AA is a symmetric and positive definite (probably) n x n matrix.
    # If A did not have full rank, then AA is positive semi-definite.
    # Also, if A is very ill-conditioned, then rounding errors can make
    # AA appear to be indefinite. Modify AA a little to make it more
    # positive definite.
    epsilon = 10 * np.spacing(1) * la.norm(AA,1)
    AA = AA + (epsilon * np.eye(n))

    # In this routine A will never be changed, but AA might be adjusted
    # with a larger "epsilon" if needed. Working copies called B and BB
    # will be used to perform the computations using the "free" set
    # of variables.

    # Initialize sets.
    free_set = list(range(n))
    binding_set = []

    # This sets up the unconstrained, core LS solver
    lsq_result = lsq_solve(A, b, lmbda, AA, epsilon, free_set, binding_set, n)

    score = lsq_result.score
    x = lsq_result.x
    residual = lsq_result.residual
    free_set = lsq_result.free_set
    binding_set = lsq_result.binding_set
    AA = lsq_result.AA
    epsilon = lsq_result.epsilon
    dels = lsq_result.dels
    lps = lsq_result.loops

    # Outer loop
    OuterLoop = 0
    TotalInnerLoops = 0
    insertions = n
    continue_outer_loop = True  # used to return from inner loop

    while continue_outer_loop:

        OuterLoop += 1

        # Save this solution
        best_score = score
        best_x = x.copy()
        best_free_set = free_set[:]
        best_binding_set = binding_set[:]
        best_insertions = insertions
        max_insertions = math.floor(exp_c * best_insertions)

        # Compute the gradient of the "Normal Equations"
        gradient = A.T @ residual

        # Check the gradient components
        insertion_set = [
            i
            for i, bi in enumerate(binding_set)
            if gradient[bi] > 0
        ]
        insertions = len(insertion_set)

        # Are we done ?
        if insertions == 0:
            # There were no changes that were feasible.
            # We are done.
            hist.append([0 for i in range(6)])
            break

        # Sort the possible insertions by their gradients to find the
        # most attractive variables to insert.
        grad_score_coordinates = [binding_set[i] for i in insertion_set]
        grad_score = gradient[grad_score_coordinates]
        set_index = np.argsort(grad_score)[::-1]  # use descending order
        insertion_set = [insertion_set[i] for i in set_index]

        # Inner loop
        InnerLoop = 0
        while True:

            InnerLoop += 1
            TotalInnerLoops += 1

            # Adjust the number of insertions.
            insertions = math.floor(red_c * insertions)
            if insertions == 0:
                insertions = 1
            insertions = min(insertions, max_insertions)
            insertion_set = insertion_set[:insertions]

            # Move variables from "binding" to "free"
            free_variables = [binding_set[i] for i in insertion_set]
            free_set.extend(free_variables)
            binding_set = [j for j in binding_set if j not in free_variables]

            # Compute a feasible solution using the unconstrained
            # least-squares solver of your choice.
            lsq_result = lsq_solve(A, b, lmbda, AA, epsilon,
                                   free_set, binding_set, insertions)

            score = lsq_result.score
            x = lsq_result.x
            residual = lsq_result.residual
            free_set = lsq_result.free_set
            binding_set = lsq_result.binding_set
            AA = lsq_result.AA
            epsilon = lsq_result.epsilon
            dels = lsq_result.dels
            lps = lsq_result.loops

            # Accumulate history info for algorithm tuning.
            # Each row has 6 values:
            # 1) Outer loop number
            # 2) Inner loop number
            # 3) Total number of inner loops
            # 4) Insertions in this inner loop
            # 5) Deletions required to make the insertions feasible
            # 6) Number of deletion loops required for these insertions
            hist.append(
                [OuterLoop,InnerLoop,TotalInnerLoops,insertions,dels,lps]
            )

            if verbose:
                print(
                    f"{OuterLoop = }, {InnerLoop = }, {TotalInnerLoops = }, "
                    f"{insertions = }, {dels = }, {lps = }"
                )

            # Check for new best solution
            if score < best_score * (1-rel_tol):
                break

            # Restore the best solution
            score = best_score
            x = best_x.copy()
            free_set = best_free_set[:]
            binding_set = best_binding_set[:]
            max_insertions = math.floor(exp_c * best_insertions)

            # Are we done ?
            if insertions == 1:
                # The best feasible change did not improve the score.
                # We are done.
                hist.append([1 for i in range(6)])
                continue_outer_loop = False
                break

    return TNTNNResult(x, AA, OuterLoop, TotalInnerLoops, hist)
