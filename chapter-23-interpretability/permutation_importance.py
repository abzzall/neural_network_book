"""Permutation importance from scratch, compared with scikit-learn's implementation."""

import numpy as np
from sklearn.inspection import permutation_importance as sklearn_importance
from sklearn.metrics import roc_auc_score

from credit_model import data, model


def permutation_importance(fitted, X, y, repeats=50, seed=0):
    """Mean drop in AUC when one column is shuffled, eq. (23.1)."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, fitted.predict_proba(X)[:, 1])
    drops = {}
    for column in X.columns:
        scores = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            scores.append(roc_auc_score(y, fitted.predict_proba(shuffled)[:, 1]))
        drops[column] = base - np.mean(scores)
    return base, drops


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, _, _ = data()
    fitted = model(X_train.columns).fit(X_train, y_train)
    base, mine = permutation_importance(fitted, X_test, y_test)
    theirs = sklearn_importance(fitted, X_test, y_test, scoring='roc_auc', n_repeats=50,
                                random_state=0).importances_mean
    print(f'test AUC: {base:.4f}')
    print('feature            ours     scikit-learn')
    order = sorted(mine, key=mine.get, reverse=True)
    for column in order[:6]:
        print(f'{column:16s} {mine[column]:7.4f}  {theirs[list(X_test.columns).index(column)]:7.4f}')
    diff = max(abs(mine[c] - theirs[i]) for i, c in enumerate(X_test.columns))
    print(f'largest difference over all {len(mine)} features: {diff:.4f}')
