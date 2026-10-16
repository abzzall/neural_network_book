"""Check the values recorded in RESULTS.md for the chapter 4 applied exercise.

    python verify_results.py               print the reference values
    python verify_results.py RESULTS.md    check a filled-in results file
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import prepare  # noqa: E402


def reference_values():
    raw = prepare.make_listings()
    df, preparation, rmse = prepare.run()
    medians = preparation[0].median_
    return {
        "rows_before": float(len(raw)),
        "district_spellings_before": float(raw["district"].nunique()),
        "rows_after_cleaning": float(len(df)),
        "missing_area_after_cleaning": float(df["area_m2"].isna().sum()),
        "median_area_train": medians[0],
        "median_year_train": medians[2],
        "validation_rmse": rmse,
        "validation_rmse_with_valuation": prepare.run(leak=True)[2],
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
