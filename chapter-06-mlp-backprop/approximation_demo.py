"""Chapter 6: wider hidden layers can approximate sin(2x) more closely.

To measure what a layer of a given width CAN represent, separately from how
well gradient descent trains it, the hidden weights are drawn at random and
only the output layer is fitted, exactly, by least squares.
"""

import numpy as np

x = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
target = np.sin(2 * x).ravel()

for width in (1, 3, 10, 30, 100):
    rng = np.random.default_rng(0)
    W, b = rng.normal(0.0, 2.0, (1, width)), rng.uniform(-np.pi, np.pi, width)
    H = np.column_stack([np.ones(len(x)), np.tanh(x @ W + b)])   # hidden features + bias
    out = np.linalg.lstsq(H, target, rcond=None)[0]
    mse = np.mean((H @ out - target) ** 2)
    print(f"hidden units = {width:3d}: mean squared error = {mse:.2e}")
