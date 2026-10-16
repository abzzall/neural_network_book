"""Chapter 2: gradient descent on f(t) = (t1^2 + 10 t2^2) / 2 at three learning rates.

Run with an output folder to write the paths used by the book's figure:
    python chapter-02-linear-algebra/gd_paths.py OUTPUT_DIR
"""

import sys
from pathlib import Path

import numpy as np

A = np.diag([1.0, 10.0])               # f(t) = t @ A @ t / 2, gradient A @ t
START = np.array([-9.0, 1.5])
RATES = [0.05, 0.18, 0.205]
STEPS = 20


def f(t):
    return 0.5 * t @ A @ t


def descend(eta, start=START, steps=STEPS):
    path = [start]
    for _ in range(steps):
        path.append(path[-1] - eta * (A @ path[-1]))
    return np.array(path)


if __name__ == "__main__":
    bound = 2 / np.linalg.eigvalsh(A)[-1]
    print(f"learning-rate bound 2/lambda_max = {bound:.3f}")
    for eta in RATES:
        path = descend(eta)
        print(f"eta = {eta}: f(start) = {f(path[0]):.2f}, f(after {STEPS} steps) = {f(path[-1]):.4f}")
        if len(sys.argv) > 1:
            out = Path(sys.argv[1]) / f"gd_eta_{eta}.dat"
            np.savetxt(out, path, fmt="%.5f", header="t1 t2", comments="")
