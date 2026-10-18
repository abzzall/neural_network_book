"""Chapter 4: a correlation heat map of input features, and their relation to the target.

Data: Breast Cancer Wisconsin (Diagnostic), UCI, CC BY 4.0, as distributed with
scikit-learn; the ten "mean" measurements of the cell nuclei.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif

data = load_breast_cancer()
X, names = data.data[:, :10], [n.replace('mean ', '') for n in data.feature_names[:10]]
y = 1 - data.target                                      # 1 = malignant, as in chapter 5
R = np.corrcoef(X, rowvar=False)                         # 10 x 10 correlation matrix
with_target = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(10)])
mi = mutual_info_classif(X, y, random_state=0)

if __name__ == '__main__':
    pairs = sorted(((abs(R[i, j]), names[i], names[j]) for i in range(10)
                    for j in range(i + 1, 10)), reverse=True)
    for r, a, b in pairs[:3]:
        print(f'most correlated features: {a} and {b}, r = {r:.3f}')
    order = np.argsort(-np.abs(with_target))
    for j in order[:3]:
        print(f'with the diagnosis: {names[j]:18s} r = {with_target[j]:.3f}, '
              f'mutual information {mi[j]:.3f} nats')
    j = int(np.argmin(np.abs(with_target)))
    print(f'weakest: {names[j]} r = {with_target[j]:.3f}, mutual information {mi[j]:.3f}')
    if len(sys.argv) > 1:
        rows = [(i, j, R[i, j]) for i in range(10) for j in range(10)]
        np.savetxt(Path(sys.argv[1]) / 'correlation.dat', np.array(rows), fmt='%.4f',
                   header='x y r', comments='')
        print('features:', ', '.join(names))
