"""Chapter 5: every update of the perceptron learning rule on logical AND."""

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([-1, -1, -1, 1])
w, b, eta = np.zeros(2), 0.0, 1.0

for epoch in range(1, 20):
    mistakes = 0
    for x_i, y_i in zip(X, y):
        if y_i * (x_i @ w + b) <= 0:          # misclassified (or on the boundary)
            w, b = w + eta * y_i * x_i, b + eta * y_i
            mistakes += 1
            if epoch <= 2:
                print(f"epoch {epoch}: x = {x_i.astype(int)}, y = {y_i:+d} -> w = {w}, b = {b}")
    if mistakes == 0:
        print(f"epoch {epoch}: no mistakes; final w = {w}, b = {b}")
        break
