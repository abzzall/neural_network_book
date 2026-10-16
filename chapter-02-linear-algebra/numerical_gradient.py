"""Chapter 2: estimate a gradient by central differences and check a formula."""

import numpy as np


def numerical_gradient(f, x, h=1e-5):
    """Central-difference estimate of the gradient of f at x."""
    grad = np.zeros_like(x)
    for j in range(x.size):
        step = np.zeros_like(x)
        step[j] = h
        grad[j] = (f(x + step) - f(x - step)) / (2 * h)
    return grad


A = np.array([[2.0, 1.0],
              [0.0, 3.0]])
a = np.array([1.0, -1.0])


def f(x):
    return x @ A @ x + a @ x


def analytic_gradient(x):
    return (A + A.T) @ x + a


if __name__ == "__main__":
    x = np.array([0.5, -2.0])
    numeric = numerical_gradient(f, x)
    exact = analytic_gradient(x)
    print("numerical:", np.round(numeric, 6))
    print("analytic: ", exact)
    print("agree to 1e-6:", bool(np.allclose(numeric, exact, atol=1e-6)))
