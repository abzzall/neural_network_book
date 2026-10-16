"""Check the values recorded in RESULTS.md for the chapter 2 applied exercise.

    python verify_results.py               print the reference values
    python verify_results.py RESULTS.md    check a filled-in results file
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import linreg  # noqa: E402


def reference_values():
    x, y = linreg.make_data()
    X = linreg.design_matrix(x)
    bound = linreg.learning_rate_bound(X)
    eigenvalues = np.linalg.eigvalsh(2 / len(y) * X.T @ X)
    early = linreg.gradient_descent(X, y, 0.9 * bound, 100)
    gd = linreg.gradient_descent(X, y, 0.9 * bound, 2000)
    exact = linreg.normal_equations(X, y)
    diverged = linreg.mse(X, y, linreg.gradient_descent(X, y, 1.1 * bound, 100))
    return {
        "lambda_min": eigenvalues[0], "lambda_max": eigenvalues[-1],
        "learning_rate_bound": bound,
        "b_after_100": early[0], "w_after_100": early[1],
        "b_gd": gd[0], "w_gd": gd[1],
        "b_normal": exact[0], "w_normal": exact[1],
        "mse_min": linreg.mse(X, y, exact),
        "mse_above_bound_exceeds_1e6": "yes" if diverged > 1e6 else "no",
    }


def recorded_values(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[1] and not set(cells[1]) <= {"-"}:
            values[cells[0].strip("`")] = cells[1]
    return values


def check(path):
    expected, recorded, ok = reference_values(), recorded_values(path), True
    for name, value in expected.items():
        got = recorded.get(name)
        if isinstance(value, str):
            good = got is not None and got.lower() == value
        else:
            try:
                good = got is not None and np.isclose(float(got), value, rtol=1e-3)
            except ValueError:
                good = False
        print(f"{'OK      ' if good else 'MISMATCH'} {name}: recorded {got}")
        ok = ok and good
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(0 if check(sys.argv[1]) else 1)
    for name, value in reference_values().items():
        print(f"{name}: {value if isinstance(value, str) else f'{value:.4g}'}")
