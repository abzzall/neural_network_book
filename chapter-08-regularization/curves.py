"""Chapter 8: shared data for the scripts -- noisy samples of sin(2 pi x)."""

import numpy as np


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    return x, np.sin(2 * np.pi * x) + rng.normal(0.0, 0.2, n)


def design(x, degree):
    """Columns 1, x, x^2, ..., x^degree."""
    return np.vander(x, degree + 1, increasing=True)


def ridge(X, y, lam):
    """Least squares with an L2 penalty lam on every coefficient except the intercept.

    Minimising |X theta - y|^2 + lam |theta[1:]|^2 is ordinary least squares on
    X stacked above sqrt(lam) times the identity (without the intercept row),
    with zeros appended to y; lstsq solves that stably even when lam = 0 and
    X^T X is nearly singular.
    """
    d = X.shape[1]
    P = np.sqrt(lam) * np.eye(d)[1:]
    return np.linalg.lstsq(np.vstack([X, P]), np.concatenate([y, np.zeros(d - 1)]), rcond=None)[0]
