"""Print reference results, or check a student's completed RESULTS.md table."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autograd_xor


def reference_values():
    initial, final, error, probabilities = autograd_xor.run()
    return {'initial_mse': initial, 'final_mse': final,
            'first_gradient_error_below_1e-12': 'yes' if error < 1e-12 else 'no',
            **{f'p_{i}': p for i, p in enumerate(probabilities)},
            'all_four_correct': 'yes' if [p >= 0.5 for p in probabilities] == [False, True, True, False] else 'no'}


def check(path):
    recorded = {}
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 2:
            recorded[cells[0].strip('`')] = cells[1]
    ok = True
    for name, expected in reference_values().items():
        got = recorded.get(name)
        if isinstance(expected, str):
            good = got is not None and got.lower() == expected
        else:
            try:
                good = got is not None and np.isclose(float(got), expected, rtol=1e-3, atol=1e-6)
            except ValueError:
                good = False
        print(f"{'OK' if good else 'MISMATCH'} {name}: recorded {got}")
        ok = ok and good
    return ok


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit(0 if check(sys.argv[1]) else 1)
    for name, value in reference_values().items():
        print(f"{name}: {value if isinstance(value, str) else f'{value:.6g}'}")
