"""Chapter 3: sample statistics by formula, checked against NumPy."""

import numpy as np

rng = np.random.default_rng(42)
height = rng.normal(170.0, 10.0, 200)                  # cm
weight = 0.9 * height - 85.0 + rng.normal(0.0, 6.0, 200)  # kg, depends on height
X = np.column_stack([height, weight])                  # one sample per row
n = X.shape[0]

mean = X.sum(axis=0) / n
centred = X - mean
cov = centred.T @ centred / (n - 1)                    # sample covariance matrix
std = np.sqrt(np.diag(cov))
corr = cov / np.outer(std, std)                        # correlation matrix

# A perfect but nonlinear dependence can have zero correlation.
x = np.linspace(-1.0, 1.0, 201)
r_square = np.corrcoef(x, x ** 2)[0, 1]

if __name__ == "__main__":
    print("mean =", np.round(mean, 2))
    print("covariance =", np.round(cov, 2).tolist())
    print(f"correlation = {corr[0, 1]:.4f}")
    print("matches NumPy:", bool(np.allclose(cov, np.cov(X.T)) and np.allclose(corr, np.corrcoef(X.T))))
    print(f"correlation of x and x^2 on [-1, 1] = {abs(r_square):.4f}")
