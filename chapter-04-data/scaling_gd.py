"""Chapter 4: why scaling features speeds up gradient descent.

Uses the line-fitting data of the chapter 2 applied exercise and counts the
steps gradient descent needs with the raw, centred, and standardised feature.
"""

import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)                 # the chapter 2 data
x = rng.uniform(0.0, 10.0, 50)
y = 3.0 + 2.0 * x + rng.normal(0.0, 1.0, 50)


def steps_to_converge(feature, tol=1e-6, max_steps=100000):
    X = np.column_stack([np.ones_like(feature), feature])
    H = 2 / len(y) * X.T @ X                    # Hessian of the MSE
    lam = np.linalg.eigvalsh(H)
    eta = 2 / (lam[0] + lam[-1])                # the best fixed learning rate
    best = np.linalg.solve(X.T @ X, X.T @ y)
    theta, losses = np.zeros(2), []
    for step in range(max_steps):
        losses.append(np.mean((X @ theta - y) ** 2))
        if np.max(np.abs(theta - best)) < tol:
            return step, np.linalg.cond(H), losses
        theta -= eta * 2 / len(y) * X.T @ (X @ theta - y)
    return max_steps, np.linalg.cond(H), losses


features = {"raw": x, "centred": x - x.mean(), "standardised": (x - x.mean()) / x.std()}
if __name__ == "__main__":
    for name, f in features.items():
        steps, kappa, losses = steps_to_converge(f)
        print(f"{name:13s} condition number = {kappa:8.2f}, steps = {steps}")
        if len(sys.argv) > 1:                   # loss curves for the book's figure
            np.savetxt(Path(sys.argv[1]) / f"loss_{name}.dat",
                       np.column_stack([np.arange(len(losses[:400])), losses[:400]]),
                       fmt="%.6g", header="step loss", comments="")
