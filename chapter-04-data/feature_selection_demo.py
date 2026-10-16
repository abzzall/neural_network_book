"""Chapter 4: correlation misses a nonlinear feature; mutual information finds it."""

import numpy as np
from sklearn.feature_selection import mutual_info_regression

rng = np.random.default_rng(1)
n = 1000
x_linear = rng.normal(size=n)
x_square = rng.uniform(-1.0, 1.0, n)
x_noise = rng.normal(size=n)
y = x_linear + 3 * x_square ** 2 + rng.normal(0.0, 0.3, n)
X = np.column_stack([x_linear, x_square, x_noise])

corr = [abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(3)]
mi = mutual_info_regression(X, y, random_state=0)

if __name__ == "__main__":
    for name, c, m in zip(["linear", "square", "noise"], corr, mi):
        print(f"{name:7s} |correlation| = {c:.3f}, mutual information = {m:.3f} nats")
