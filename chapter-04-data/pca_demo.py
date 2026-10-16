"""Chapter 4: principal components from the eigenvectors of the covariance matrix."""

import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)
t = rng.normal(size=300)                          # one hidden factor drives three features
X = np.column_stack([t + 0.1 * rng.normal(size=300),
                     2 * t + 0.2 * rng.normal(size=300),
                     rng.normal(size=300)])
Xc = X - X.mean(axis=0)
cov = Xc.T @ Xc / (len(X) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
order = np.argsort(eigenvalues)[::-1]             # largest variance first
explained = eigenvalues[order] / eigenvalues.sum()
Z = Xc @ eigenvectors[:, order[:2]]               # keep two components

if __name__ == "__main__":
    print("explained variance ratio:", np.round(explained, 4))
    sk = PCA(n_components=3).fit(X)
    print("matches scikit-learn:", bool(np.allclose(explained, sk.explained_variance_ratio_)))
    print("shape before and after:", X.shape, Z.shape)
