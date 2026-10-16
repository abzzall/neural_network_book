"""Check the values recorded in RESULTS.md for the chapter 12 applied exercise.

    python verify_results.py               print the reference values
    python verify_results.py RESULTS.md    check a filled-in results file

PyTorch does not promise identical results on different machines, so the
reference values are computed on the machine that checks them.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cnn_vs_mlp as m  # noqa: E402


def reference_values():
    values = {"mlp_parameters": float(m.n_params(m.mlp())), "cnn_parameters": float(m.n_params(m.cnn()))}
    for name, builder, train_fn in (("mlp", m.mlp, m.fit), ("cnn", m.cnn, m.fit),
                                    ("cnn_augmented", m.cnn, m.fit_augmented)):
        plain, shifted = m.evaluate(builder, train_fn)
        values[f"{name}_test_accuracy"] = plain
        values[f"{name}_shifted_accuracy"] = shifted
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
        try:
            good = got is not None and np.isclose(float(got), value, rtol=1e-3, atol=1e-4)
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
