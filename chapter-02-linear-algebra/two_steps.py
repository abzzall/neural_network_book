"""Chapter 2: two gradient-descent steps for a line through three points."""

import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 5.0])
X = np.column_stack([np.ones_like(x), x])   # design matrix: columns 1 and x
n = len(y)
eta = 0.1


def mse(theta):
    r = X @ theta - y
    return r @ r / n


def gradient(theta):
    return 2 / n * X.T @ (X @ theta - y)


theta = np.zeros(2)                          # theta = (b, w)
if __name__ == "__main__":
    print(f"step 0: theta = {np.round(theta, 4)}, loss = {mse(theta):.4f}")
    for step in (1, 2):
        g = gradient(theta)
        theta = theta - eta * g
        print(f"step {step}: gradient = {np.round(g, 4)}, "
              f"theta = {np.round(theta, 4)}, loss = {mse(theta):.4f}")
    exact = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"normal equations: theta = {np.round(exact, 4)}, loss = {mse(exact):.4f}")
    bound = 2 / np.linalg.eigvalsh(2 / n * X.T @ X)[-1]
    print(f"learning-rate bound = {bound:.4f}")
