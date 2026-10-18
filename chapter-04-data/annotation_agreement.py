"""Chapter 4: agreement between two annotators, raw and corrected for chance."""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def kappa(table):
    """Cohen's kappa from a 2 x 2 table: rows annotator A, columns annotator B."""
    table = np.asarray(table, dtype=float)
    n = table.sum()
    observed = np.trace(table) / n                       # share of items labelled alike
    expected = (table.sum(axis=1) / n) @ (table.sum(axis=0) / n)   # alike by chance
    return observed, expected, (observed - expected) / (1 - expected)


def labels_from(table):
    """Expand a table of counts into two label vectors, to check against scikit-learn."""
    a, b = [], []
    for i in range(2):
        for j in range(2):
            a += [i] * table[i][j]
            b += [j] * table[i][j]
    return a, b


tables = {"balanced": [[45, 5], [10, 40]],       # 100 messages, about half spam
          "rare class": [[88, 5], [5, 2]]}       # 100 messages, few spam

if __name__ == "__main__":
    for name, table in tables.items():
        observed, expected, k = kappa(table)
        check = cohen_kappa_score(*labels_from(table))
        print(f"{name:10s}: raw agreement {observed:.3f}, by chance {expected:.4f}, "
              f"kappa {k:.3f} (scikit-learn {check:.3f})")
