"""Chapter 2: a layer of neurons as a matrix-vector product."""

import numpy as np

# Three inputs, two neurons: row k of W holds the weights of neuron k.
W = np.array([[0.5, -1.0, 2.0],
              [1.5, 0.0, -0.5]])
b = np.array([0.1, -0.2])

x = np.array([1.0, 2.0, 3.0])         # one sample, shape (3,)
z = W @ x + b                         # one output per neuron, shape (2,)

X = np.array([[1.0, 2.0, 3.0],        # a batch: one sample per row,
              [0.0, 1.0, -1.0]])      # shape (2, 3)
Z = X @ W.T + b                       # one row of outputs per sample, shape (2, 2)

if __name__ == "__main__":
    print("z =", np.round(z, 4), "shape", z.shape)
    print("Z =", np.round(Z, 4).tolist(), "shape", Z.shape)
