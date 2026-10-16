"""Chapter 4: train a perceptron on logical AND and on a random separable dataset."""

import numpy as np

from nnbook import Perceptron

# Logical AND with labels in {-1, +1}
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([-1, -1, -1, 1])

model = Perceptron().fit(X_and, y_and)
print("AND: w =", model.w, "b =", model.b)
print("AND: predictions =", model.predict(X_and))
print("AND: errors per epoch =", model.errors_per_epoch)

# Two Gaussian clouds that are linearly separable
rng = np.random.default_rng(42)
X = np.vstack([rng.normal(-2.0, 1.0, (50, 2)), rng.normal(2.0, 1.0, (50, 2))])
y = np.array([-1] * 50 + [1] * 50)

model = Perceptron(max_epochs=1000).fit(X, y)
accuracy = np.mean(model.predict(X) == y)
print(f"Clouds: accuracy = {accuracy:.2f}, epochs = {len(model.errors_per_epoch)}")
