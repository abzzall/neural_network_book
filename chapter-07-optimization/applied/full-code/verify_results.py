"""Check the values recorded in RESULTS.md for the chapter 7 applied exercise.

    python verify_results.py               print the reference values
    python verify_results.py RESULTS.md    check a filled-in results file
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare  # noqa: E402

KEYS = {"full-batch GD": "full_batch", "mini-batch SGD": "sgd",
        "SGD + momentum": "momentum", "Adam": "adam"}


def reference_values():
    X_train, X_test, y_train, y_test = compare.load_split()
    values = {}
    for name, net in compare.train_all(X_train, y_train).items():
        reached = compare.epochs_to(net.losses)
        values[f"{KEYS[name]}_epochs_to_0.05"] = "never" if reached is None else float(reached)
        values[f"{KEYS[name]}_test_accuracy"] = compare.accuracy(net, X_test, y_test)
    return values


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
