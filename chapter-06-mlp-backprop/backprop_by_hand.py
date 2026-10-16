"""Chapter 6: one forward and one backward pass through a 2-2-1 network, every number shown."""

import numpy as np

from nnbook.linear import sigmoid

x, y = np.array([1.0, 0.5]), 1.0                 # one sample, target 1
W1 = np.array([[0.5, -0.3], [0.2, 0.8]])         # hidden layer: row k = neuron k
b1 = np.array([0.0, 0.1])
w2, b2 = np.array([1.0, -1.5]), 0.2              # output neuron

# Forward pass.
z1 = W1 @ x + b1
a1 = sigmoid(z1)
z2 = w2 @ a1 + b2
p = sigmoid(z2)
loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))

# Backward pass.
delta2 = p - y                                   # dL/dz2 for sigmoid + cross-entropy
dw2, db2 = delta2 * a1, delta2
delta1 = w2 * delta2 * a1 * (1 - a1)             # back through w2 and the sigmoid
dW1, db1 = np.outer(delta1, x), delta1

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("z1 =", z1, " a1 =", a1)
    print(f"z2 = {z2:.4f}, p = {p:.4f}, loss = {loss:.4f}")
    print(f"delta2 = {delta2:.4f}, dw2 =", dw2, f", db2 = {db2:.4f}")
    print("delta1 =", delta1)
    print("dW1 =", np.round(dW1, 4).tolist())
