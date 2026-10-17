"""Print reference results or check the completed credit-decision table.

Reference values were measured with Python 3.14.4 and scikit-learn 1.9.1
(17 October 2026), as means over five random splits.
"""

from pathlib import Path
import sys

import numpy as np

REFERENCE = {
    'bayes_threshold': 0.833333,
    'approve_all_cost': 1.5,
    'logistic_at_0.5_cost': 0.856,
    'logistic_at_cost_threshold_cost': 0.48,
    'logistic_at_cost_threshold_approved': 0.412,
    'boosting_at_cost_threshold_cost': 0.641333,
}


def matches(name, actual, expected):
    tolerance = 1e-6 if name in ('bayes_threshold', 'approve_all_cost') else 0.02
    return bool(abs(actual - expected) <= tolerance)


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
        print(f'{name}: {value:.6g}')
