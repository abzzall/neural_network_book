"""Chapter 2 applied exercise (starting point): linear regression by gradient descent and by the normal equations."""

import numpy as np


def make_data(n=50, seed=42):
    """A noisy line y = 3 + 2x + noise, with x between 0 and 10."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    y = 3.0 + 2.0 * x + rng.normal(0.0, 1.0, n)
    return x, y


# Step 1: a column of ones carries the bias b; theta = (b, w).
def design_matrix(x):
    raise NotImplementedError("Step 1: stack a column of ones and x")


# Step 2: mean squared error.
def mse(X, y, theta):
    raise NotImplementedError("Step 2: mean squared error, eq. (2.23)")


# Step 3: its gradient, (2/n) X^T (X theta - y).
def gradient(X, y, theta):
    raise NotImplementedError("Step 3: gradient of the MSE, eq. (2.24)")


# Step 5: gradient descent converges only for eta < 2 / lambda_max.
def learning_rate_bound(X):
    raise NotImplementedError("Step 5: 2 / largest eigenvalue of the Hessian, eq. (2.22)")


# Step 6: plain gradient descent from theta = 0.
def gradient_descent(X, y, eta, steps):
    raise NotImplementedError("Step 6: repeat the update of eq. (2.21)")


# Step 7: the exact minimiser, from X^T X theta = X^T y.
def normal_equations(X, y):
    raise NotImplementedError("Step 7: solve the normal equations, eq. (2.25)")
