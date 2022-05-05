import numpy as np
import scipy.linalg as la
from scipy.stats import ortho_group

from pytntnn.matlab_functions import chol, planerot

ABS_TOL = 1e-12  # arbitrarily chosen, seems reasonable


def test_chol_simple():
    """Tests chol with a simple 3 x 3 matrix."""
    A = [[  4,  12, -16],
         [ 12,  37, -43],
         [-16, -43,  98]]

    c, p = chol(A)

    U = np.array(
        [[ 2.,  6., -8.],
         [ 0.,  1.,  5.],
         [ 0.,  0.,  3.]]
    )
    assert p == 0
    assert np.allclose(c, U, atol=ABS_TOL)


def _test_chol_random(n: int, random_state: int = None, verbose: bool = False):
    """Tests chol with a randomly generated SPD matrix."""
    # Generates a n x n SPD test matrix
    spectrum_A = np.arange(n) + 1
    Q = ortho_group.rvs(dim=n, random_state=random_state)
    A = Q @ np.diag(spectrum_A) @ Q.T

    c, p = chol(A)

    if verbose:
        print(f"{p = }")
        print(f"{la.norm(A - c.T @ c) = }")

    assert p == 0
    assert np.allclose(c.T @ c, A, atol=ABS_TOL)


def test_chol_random():
    """Tests chol with multiple randomly generated SPD matrices."""
    rng_seed = 555132
    for n in [10, 20, 50, 100, 200]:
        _test_chol_random(n, random_state=rng_seed, verbose=False)


def test_planerot(verbose=True):
    """Tests planerot with a simple example."""
    x = np.array([3, 1])
    G, y = planerot(x)

    if verbose:
        print("G = \n", G, '\n')
        print('y = ', y)

    assert np.allclose(y, G @ x, atol=ABS_TOL)
