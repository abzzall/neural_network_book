"""One GCN layer on an undirected four-node path, in NumPy."""

import numpy as np


def normalise(adjacency):
    adjacency = np.asarray(adjacency, dtype=float)
    if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
            or not np.isfinite(adjacency).all() or (adjacency < 0).any()
            or not np.allclose(adjacency, adjacency.T)):
        raise ValueError('expected a finite nonnegative symmetric adjacency')
    augmented = adjacency + np.eye(len(adjacency))
    inverse = augmented.sum(axis=1) ** -.5
    return inverse[:, None] * augmented * inverse[None, :]


def example():
    adjacency = np.array([[0, 1, 0, 0], [1, 0, 1, 0],
                          [0, 1, 0, 1], [0, 0, 1, 0]])
    features = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 0.]])
    weights = np.array([[1.], [2.]])
    return adjacency, features, weights


if __name__ == '__main__':
    adjacency, features, weights = example()
    operator = normalise(adjacency)
    print('normalised adjacency:')
    print(np.array2string(operator, precision=6))
    print('GCN output:', np.round(np.maximum(operator @ features @ weights, 0), 6).ravel())
