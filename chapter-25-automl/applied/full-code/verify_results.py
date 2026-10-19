"""Print reference results or check the completed search table.

Reference values were measured on a CPU (Python 3.14.4, scikit-learn 1.9.1,
18 October 2026). Allocated candidate-epoch budgets and the selected shape must match exactly.
"""

from pathlib import Path
import sys

import numpy as np

REFERENCE = {
    'candidates': 10,
    'exhaustive_validation': 0.9639, 'exhaustive_test': 0.975, 'exhaustive_epochs': 2000,
    'halving_validation': 0.9639, 'halving_test': 0.975, 'halving_epochs': 900,
    'same_selection': 1,
}
EXACT = {'candidates', 'exhaustive_epochs', 'halving_epochs', 'same_selection'}


def matches(name, actual, expected):
    if name in EXACT:
        return actual == expected
    return bool(abs(actual - expected) <= 0.02)


def check(path):
    recorded = {}
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 2:
            recorded[cells[0].strip('`')] = cells[1]
    ok = True
    for name, expected in REFERENCE.items():
        try:
            actual = float(recorded.get(name, 'nan'))
            good = np.isfinite(actual) and matches(name, actual, expected)
        except ValueError:
            good = False
        print(f"{'OK' if good else 'MISMATCH'} {name}: recorded {recorded.get(name)}")
        ok = ok and good
    return ok


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit(0 if check(sys.argv[1]) else 1)
    for name, value in REFERENCE.items():
        print(f'{name}: {value:g}')
