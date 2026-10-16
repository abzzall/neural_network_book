"""Chapter 8: polynomials of rising degree fitted to 20 noisy points.

Run with an output folder to write the training and validation errors for the figure.
"""

import sys
from pathlib import Path

import numpy as np

from curves import design, make_data

x_train, y_train = make_data(20, seed=0)
x_val, y_val = make_data(200, seed=1)

rows = []
for degree in range(16):
    theta = np.linalg.lstsq(design(x_train, degree), y_train, rcond=None)[0]
    train = np.mean((design(x_train, degree) @ theta - y_train) ** 2)
    val = np.mean((design(x_val, degree) @ theta - y_val) ** 2)
    rows.append((degree, train, val))

if __name__ == "__main__":
    for degree, train, val in rows:
        if degree in (0, 1, 3, 5, 9, 15):
            print(f"degree {degree:2d}: training MSE = {train:.4f}, validation MSE = {val:.4g}")
    if len(sys.argv) > 1:
        np.savetxt(Path(sys.argv[1]) / "degree_errors.dat", np.array(rows), fmt="%.6g",
                   header="degree train val", comments="")
