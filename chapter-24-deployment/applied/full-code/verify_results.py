"""Print reference results or check the completed drift-monitoring table.

Reference values were measured on a CPU (Python 3.14.4, PyTorch 2.14.1,
17 October 2026). Alarms must match exactly.
"""

from pathlib import Path
import sys

import numpy as np

REFERENCE = {
    'unchanged_intensity_psi': 0.0173, 'unchanged_confidence_alarm': 0,
    'shifted_accuracy': 0.6556, 'shifted_intensity_alarm': 0, 'shifted_confidence_alarm': 1,
    'fainter_accuracy': 0.9822, 'fainter_intensity_psi': 12.6275, 'fainter_intensity_alarm': 1,
    'swapped_accuracy': 0.78, 'swapped_confidence_alarm': 0,
}


def matches(name, actual, expected):
    if name.endswith('_alarm'):
        return actual == expected
    if name.endswith('_psi'):
        return bool(abs(actual - expected) <= max(0.02, 0.05 * expected))
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
