"""Check the values recorded in RESULTS.md for the chapter 1 applied exercise.

    python verify_results.py               print the reference values
    python verify_results.py RESULTS.md    check a filled-in results file
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import neuron  # noqa: E402


def reference_values():
    flips = neuron.flip_bias()
    return {
        "irrigation_days_bias_0.5": float(neuron.irrigation_days(0.5)),
        "irrigation_days_bias_1.0": float(neuron.irrigation_days(1.0)),
        "irrigation_days_bias_1.5": float(neuron.irrigation_days(1.5)),
        "flip_bias_day_1": flips[0],
        "flip_bias_day_8": flips[7],
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
            good = got is not None and np.isclose(float(got), value, rtol=1e-3, atol=1e-9)
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
