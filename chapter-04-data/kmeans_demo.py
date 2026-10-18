"""Chapter 4: k-means from scratch on simulated customers, and choosing k."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

rng = np.random.default_rng(0)
# Three simulated groups of customers: yearly spending (thousands) and visits per month.
centres = np.array([[200.0, 2.0], [600.0, 3.0], [400.0, 12.0]])
group = rng.integers(0, 3, 600)
X = centres[group] + rng.normal(0, 1, (600, 2)) * np.array([60.0, 1.5])


def kmeans(X, k, seed=0, steps=100):
    """Lloyd's algorithm: assign each point to the nearest centre, move centres to the means."""
    centres = X[np.random.default_rng(seed).choice(len(X), k, replace=False)]
    for _ in range(steps):
        labels = ((X[:, None, :] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)
        new = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    inertia = ((X - centres[labels]) ** 2).sum()           # within-cluster squared distances
    return labels, centres, inertia


def best_of(X, k, starts=10):
    return min((kmeans(X, k, seed) for seed in range(starts)), key=lambda r: r[2])


Z = (X - X.mean(axis=0)) / X.std(axis=0)                     # standardised, as in section 4.4

if __name__ == "__main__":
    raw = best_of(X, 3)[0]
    std_labels, _, inertia = best_of(Z, 3)
    reference = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z)
    print(f"agreement with the true groups: raw features {adjusted_rand_score(group, raw):.3f}, "
          f"standardised {adjusted_rand_score(group, std_labels):.3f}")
    print(f"inertia {inertia:.2f}, scikit-learn {reference.inertia_:.2f}")
    for k in (2, 3, 4, 5, 6):
        labels, _, inertia = best_of(Z, k)
        print(f"k = {k}: inertia {inertia:7.2f}, silhouette {silhouette_score(Z, labels):.3f}")
