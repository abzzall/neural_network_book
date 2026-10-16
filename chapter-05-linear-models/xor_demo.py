"""Chapter 5: no linear model separates XOR; one engineered feature does."""

import numpy as np

from nnbook import LogisticRegression, Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y01 = np.array([0, 1, 1, 0])                     # XOR

perceptron = Perceptron(max_epochs=100).fit(X, 2 * y01 - 1)
plain = LogisticRegression(learning_rate=0.5, epochs=5000).fit(X, y01)
X_product = np.column_stack([X, X[:, 0] * X[:, 1]])   # add the feature x1 * x2
engineered = LogisticRegression(learning_rate=0.5, epochs=5000).fit(X_product, y01)

if __name__ == "__main__":
    print(f"perceptron mistakes in epoch 100: {perceptron.errors_per_epoch[-1]}")
    print("logistic regression probabilities:", np.round(plain.predict_proba(X), 3))
    print("with the feature x1*x2:          ", np.round(engineered.predict_proba(X_product), 3))
