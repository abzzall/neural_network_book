"""Print reference results or check the completed fairness-audit table.

Reference values were measured with Python 3.14.4 and scikit-learn 1.9.1
(17 October 2026); they are means over five random splits.
"""

from pathlib import Path
import sys

import numpy as np

REFERENCE = {
    'with_age_accuracy': 0.750667,
    'with_age_older_tpr': 0.889456,
    'with_age_young_tpr': 0.775203,
    'with_age_opportunity_gap': -0.114253,
    'with_age_odds_gap': 0.143894,
    'without_age_opportunity_gap': -0.0941300,
    'group_threshold_accuracy': 0.75,
    'group_threshold_opportunity_gap': -0.00221325,
    'group_threshold_young_fpr': 0.716504,
    'group_threshold_odds_gap': 0.209032,
}


def matches(name, actual, expected):
    return bool(abs(actual - expected) <= 0.01)


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
