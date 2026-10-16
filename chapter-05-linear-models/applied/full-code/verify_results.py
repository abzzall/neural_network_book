"""Check the values recorded in RESULTS.md for the chapter 5 applied exercise.

    python verify_results.py               print the reference values
    python verify_results.py RESULTS.md    check a filled-in results file
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import classify  # noqa: E402


def reference_values():
    X, y, names = classify.load()
    X_train, X_test, y_train, y_test = classify.prepare(X, y)
    model = classify.train(X_train, y_train)
    result = classify.evaluate(model, X_test, y_test)
    top_name, top_weight = classify.largest_weights(model, names, k=1)[0]
    return {
        "final_training_loss": model.losses[-1],
        "test_accuracy": result["accuracy"],
        "missed_malignant": float(result["missed_malignant"]),
        "false_alarms": float(result["false_alarms"]),
        "sklearn_test_accuracy": classify.reference_accuracy(X_train, X_test, y_train, y_test),
        "largest_weight_feature": top_name,
        "largest_weight": top_weight,
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
            good = got is not None and got.strip().lower() == value
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
