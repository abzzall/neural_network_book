"""Chapter 4: data leakage through feature selection on pure noise."""

import numpy as np

n, d, k, repeats = 100, 2000, 20, 50
train, val = np.arange(70), np.arange(70, 100)


def top_k_correlated(X, y, rows):
    """Indices of the k features most correlated with y on the given rows."""
    Xc = X[rows] - X[rows].mean(axis=0)
    yc = y[rows] - y[rows].mean()
    corr = Xc.T @ yc / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc))
    return np.argsort(-np.abs(corr))[:k]


def validation_r2(X, y, features):
    """Fit least squares on the training rows; R^2 on the validation rows."""
    A = np.column_stack([np.ones(n), X[:, features]])
    theta = np.linalg.lstsq(A[train], y[train], rcond=None)[0]
    residual = y[val] - A[val] @ theta
    baseline = y[val] - y[train].mean()
    return 1 - residual @ residual / (baseline @ baseline)


leaky, honest = [], []
for seed in range(repeats):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))        # 2000 features of pure noise
    y = rng.normal(size=n)             # a target unrelated to all of them
    leaky.append(validation_r2(X, y, top_k_correlated(X, y, np.arange(n))))  # ALL rows
    honest.append(validation_r2(X, y, top_k_correlated(X, y, train)))        # training rows

if __name__ == "__main__":
    print(f"mean validation R^2 over {repeats} datasets of pure noise:")
    print(f"  features chosen on all data:      {np.mean(leaky):.3f}")
    print(f"  features chosen on training data: {np.mean(honest):.3f}")
