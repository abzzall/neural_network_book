"""Chapter 2 applied exercise: linear regression by gradient descent and by the normal equations."""

import numpy as np


def make_data(n=50, seed=42):
    """A noisy line y = 3 + 2x + noise, with x between 0 and 10."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    y = 3.0 + 2.0 * x + rng.normal(0.0, 1.0, n)
    return x, y


# Step 1: a column of ones carries the bias b; theta = (b, w).
def design_matrix(x):
    return np.column_stack([np.ones_like(x), x])


# Step 2: mean squared error.
def mse(X, y, theta):
    r = X @ theta - y
    return r @ r / len(y)


# Step 3: its gradient, (2/n) X^T (X theta - y).
def gradient(X, y, theta):
    return 2 / len(y) * X.T @ (X @ theta - y)


# Step 5: gradient descent converges only for eta < 2 / lambda_max.
def learning_rate_bound(X):
    hessian = 2 / X.shape[0] * X.T @ X
    return 2 / np.linalg.eigvalsh(hessian)[-1]


# Step 6: plain gradient descent from theta = 0.
def gradient_descent(X, y, eta, steps):
    theta = np.zeros(X.shape[1])
    for _ in range(steps):
        theta = theta - eta * gradient(X, y, theta)
    return theta


# Step 7: the exact minimiser, from X^T X theta = X^T y.
def normal_equations(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)
