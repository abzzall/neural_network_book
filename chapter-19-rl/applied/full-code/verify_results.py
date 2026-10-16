"""Print reference results or check the completed DQN results table.

Reference values were measured on a CPU (Python 3.14.4, PyTorch 2.14.1,
17 October 2026). Reinforcement learning is sensitive to tiny numerical
differences, so tolerances are wide; the counts must be close, not equal.
"""

from pathlib import Path
import sys

import numpy as np

REFERENCE = {
    'final_mean': 500.0,
    'final_min': 500.0,
    'checkpoints_at_500': 3,
    'worst_after_40k': 14.3,
}
TOLERANCE = {'final_mean': 100.0, 'final_min': 150.0, 'checkpoints_at_500': 2,
             'worst_after_40k': 120.0}


def matches(name, actual, expected):
    return bool(abs(actual - expected) <= TOLERANCE[name])


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
