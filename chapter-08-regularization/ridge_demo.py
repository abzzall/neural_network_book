"""Chapter 8: an L2 penalty tames a degree-12 polynomial fitted to 20 points."""

import numpy as np

from curves import design, make_data, ridge

x_train, y_train = make_data(20, seed=0)
x_val, y_val = make_data(200, seed=1)
X, X_val = design(x_train, 12), design(x_val, 12)

if __name__ == "__main__":
    for lam in (0.0, 1e-6, 1e-4, 1e-2, 1.0):
        theta = ridge(X, y_train, lam)
        train = np.mean((X @ theta - y_train) ** 2)
        val = np.mean((X_val @ theta - y_val) ** 2)
        print(f"lambda = {lam:<7g} training MSE = {train:.4f}, validation MSE = {val:.4g}, "
              f"largest |coefficient| = {np.abs(theta).max():.3g}")
