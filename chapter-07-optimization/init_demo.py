"""Chapter 7: how the scale of random weights changes the signal through ten layers.

Run with an output folder to write the per-layer standard deviations for the book.
"""

import sys
from pathlib import Path

import numpy as np

n, width, depth = 1000, 256, 10


def activation_stds(init_std, activation):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n, width))
    stds, saturated = [], []
    for _ in range(depth):
        W = rng.normal(0.0, init_std, (width, width))
        a = activation(a @ W.T)
        stds.append(a.std())
        saturated.append(np.mean(np.abs(a) > 0.99))   # tanh units with almost no gradient
    return stds, saturated


relu = lambda z: np.maximum(z, 0.0)  # noqa: E731
cases = {
    "tanh, std 0.01": activation_stds(0.01, np.tanh),
    "tanh, std 1": activation_stds(1.0, np.tanh),
    "tanh, Glorot": activation_stds(np.sqrt(2.0 / (2 * width)), np.tanh),
    "ReLU, Glorot": activation_stds(np.sqrt(2.0 / (2 * width)), relu),
    "ReLU, He": activation_stds(np.sqrt(2.0 / width), relu),
}

if __name__ == "__main__":
    for name, (stds, saturated) in cases.items():
        note = f"; saturated at layer 10: {saturated[9]:.0%}" if name.startswith("tanh") else ""
        print(f"{name:15s} std after layers 1, 5, 10: "
              f"{stds[0]:.4f}, {stds[4]:.4f}, {stds[9]:.4f}{note}")
    if len(sys.argv) > 1:
        table = np.column_stack([np.arange(1, depth + 1)] + [cases[k][0] for k in cases])
        np.savetxt(Path(sys.argv[1]) / "init_stds.dat", table, fmt="%.6g",
                   header="layer tanh001 tanh1 tanhglorot reluglorot reluhe", comments="")
