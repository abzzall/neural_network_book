"""Chapter 6: a network with two hidden units learns XOR.

Run with an output folder to write the hidden representation for the book's figure.
"""

import sys
from pathlib import Path

import numpy as np

from nnbook import Dense, Network, SoftmaxCrossEntropy, Tanh

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])

rng = np.random.default_rng(3)
net = Network([Dense(2, 2, rng), Tanh(), Dense(2, 2, rng)], SoftmaxCrossEntropy())
net.fit(X, y, learning_rate=0.5, epochs=3000)
hidden = np.tanh(X @ net.layers[0].W.T + net.layers[0].b)   # the learned features

if __name__ == "__main__":
    print(f"loss: first epoch {net.losses[0]:.4f}, last {net.losses[-1]:.4f}")
    print("predictions:", net.predict(X), " targets:", y)
    print("hidden features:", np.round(hidden, 3).tolist())
    if len(sys.argv) > 1:
        np.savetxt(Path(sys.argv[1]) / "xor_hidden.dat", np.column_stack([hidden, y]),
                   fmt="%.4f", header="h1 h2 label", comments="")
