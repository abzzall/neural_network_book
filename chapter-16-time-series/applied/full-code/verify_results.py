"""Print reference results or check the completed probabilistic-forecast results table.

Reference values were measured on a CPU (Python 3.14.4, NumPy 2.5.4,
PyTorch 2.14.1, 16 October 2026). Counts must match exactly; scores may differ
slightly across platforms.
"""

from pathlib import Path
import sys

import numpy as np

REFERENCE = {
    'train_windows': 478,
    'origins': 25,
    'quantile_mase': 1.12876,
    'quantile_coverage': 0.822857,
    'quantile_pinball': 359.244,
    'naive_mase': 1.27554,
    'naive_coverage': 0.757143,
    'naive_pinball': 394.989,
}
COUNTS = ('train_windows', 'origins')


def matches(name, actual, expected):
    if name in COUNTS:
        return actual == expected
    if name.endswith('_coverage'):              # one day in 350 is 0.003
        return abs(actual - expected) <= 0.015
    return bool(np.isclose(actual, expected, rtol=0.01, atol=0))


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
