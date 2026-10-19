"""Chapter 23: how much the fairness gaps of the audit vary between splits.

Runs the finished audit of the applied analysis on its five splits and prints,
for each variant, the mean parity gap and the smallest and largest opportunity
gap. Download the data first: python chapter-23-interpretability/credit_data.py --download
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'applied' / 'full-code'))
import fairness_audit as audit

X, good, young = audit.load()
rows = [audit.evaluate(seed, X, good.to_numpy(), young.to_numpy()) for seed in range(5)]
for variant in rows[0]:
    parity = [row[variant][7] for row in rows]
    opportunity = [row[variant][8] for row in rows]
    print(f'{variant}: parity gap {np.mean(parity):.3f}, '
          f'opportunity gap {min(opportunity):.3f} to {max(opportunity):.3f}')
