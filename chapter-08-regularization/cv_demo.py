"""Chapter 8: five-fold cross-validation chooses the L2 penalty."""

import numpy as np

from curves import design, make_data, ridge
from nnbook.model_selection import kfold

x, y = make_data(60, seed=2)
X = design(x, 12)
lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]

if __name__ == "__main__":
    for lam in lambdas:
        errors = []
        for train, val in kfold(len(y), k=5, seed=0):
            theta = ridge(X[train], y[train], lam)
            errors.append(np.mean((X[val] @ theta - y[val]) ** 2))
        print(f"lambda = {lam:<7g} cross-validated MSE = {np.mean(errors):.4f} "
              f"(fold std {np.std(errors):.4f})")
