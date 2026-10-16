"""Print reference results or check the completed forecasting results table."""

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recurrent_forecast


@lru_cache(maxsize=1)
def reference_values():
    return recurrent_forecast.run()


def check(path):
    recorded = {}
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 2:
            recorded[cells[0].strip('`')] = cells[1]
    ok = True
    for name, expected in reference_values().items():
        try:
            actual = float(recorded.get(name, 'nan'))
            good = np.isfinite(actual) and (actual == expected if name == 'train_windows' else
                                           np.isclose(actual, expected, rtol=0.02, atol=1e-5))
        except ValueError:
            good = False
        print(f"{'OK' if good else 'MISMATCH'} {name}: recorded {recorded.get(name)}")
        ok = ok and good
    return ok


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit(0 if check(sys.argv[1]) else 1)
    for name, value in reference_values().items():
        print(f'{name}: {value:.6g}')
