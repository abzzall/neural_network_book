"""Check the values recorded in RESULTS.md for the chapter 8 applied exercise.

    python verify_results.py               print the reference values
    python verify_results.py RESULTS.md    check a filled-in results file
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import regularise  # noqa: E402

KEYS = {"baseline": "baseline", "L2": "l2", "dropout": "dropout",
        "early stopping": "early_stopping", "batch norm": "batch_norm"}


def reference_values():
    train, val, test = regularise.load_splits()
    nets = regularise.train_all(train, val)
    values = {f"{KEYS[k]}_val_accuracy": regularise.accuracy(n, val) for k, n in nets.items()}
    values["early_stopping_epochs"] = float(len(nets["early stopping"].losses))
    best = regularise.choose(nets, val)
    values["chosen_setting"] = best
    values["chosen_test_accuracy"] = regularise.accuracy(nets[best], test)
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
            good = got is not None and got.lower() == value.lower()
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
