"""Chapter 7: four optimisers on a narrow valley, f(t) = (t1^2 + 25 t2^2) / 2.

Run with an output folder to write the paths drawn in the book's figure.
"""

import sys
from pathlib import Path

import numpy as np

from nnbook.optimizers import SGD, Adam, RMSProp

A = np.array([1.0, 25.0])                  # curvatures along the two axes


def f(t):
    return 0.5 * np.sum(A * t ** 2)


def run(optimizer, steps=40):
    t = np.array([-8.0, 1.5])
    path = [t.copy()]
    for _ in range(steps):
        optimizer.step([(t, A * t)])       # the gradient of f is A * t
        path.append(t.copy())
    return np.array(path)


# Each optimiser gets its best learning rate from the same small grid.
GRID = [0.01, 0.03, 0.07, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0]
makers = {
    "gradient descent": lambda lr: SGD(lr=lr),
    "momentum": lambda lr: SGD(lr=lr, momentum=0.8),
    "RMSProp": lambda lr: RMSProp(lr=lr),
    "Adam": lambda lr: Adam(lr=lr),
}


def best(make):
    results = [(f(run(make(lr))[-1]), lr) for lr in GRID]
    value, lr = min(r for r in results if np.isfinite(r[0]))
    return lr, run(make(lr))


if __name__ == "__main__":
    for name, make in makers.items():
        lr, path = best(make)
        print(f"{name:16s} best lr = {lr:<4} f after 40 steps = {f(path[-1]):.2e}")
        if len(sys.argv) > 1:
            out = Path(sys.argv[1]) / f"path_{name.split()[0].lower()}.dat"
            np.savetxt(out, path, fmt="%.5f", header="t1 t2", comments="")
