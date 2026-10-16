"""Check the values recorded in RESULTS.md for the chapter 3 applied exercise.

    python verify_results.py               print the reference values
    python verify_results.py RESULTS.md    check a filled-in results file
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import estimate  # noqa: E402


def reference_values():
    clicks, times = estimate.make_data()
    p = estimate.bernoulli_mle(clicks)
    mu, var = estimate.gaussian_mle(times)
    mu_gd, var_gd = estimate.gaussian_gd(times)
    return {
        "clicks": float(clicks.sum()),
        "p_mle": p,
        "p_gd": estimate.bernoulli_gd(clicks),
        "min_cross_entropy_nats": estimate.bernoulli_nll(clicks, p),
        "entropy_of_p_mle_nats": -(p * np.log(p) + (1 - p) * np.log(1 - p)),
        "mu_mle": mu,
        "var_mle": var,
        "var_unbiased": times.var(ddof=1),
        "mu_gd": mu_gd,
        "var_gd": var_gd,
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
        print(f"{name}: {value:.4g}")
