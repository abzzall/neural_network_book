"""Native missing values, early stopping and monotonic constraints in a modern booster."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def data(seed=7, n=6000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    logits = 2.2 * X[:, 0] - 1.7 * X[:, 1] + 1.2 * np.sin(2 * X[:, 2])
    y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    missing = rng.random((n, 2)) < 0.10
    X[:, :2][missing] = np.nan
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


def monotonic_violations(model):
    grid = np.linspace(-3, 3, 301)
    base = np.zeros((grid.size, 4))
    base[:, 0] = grid
    increasing = model.predict_proba(base)[:, 1]
    base[:, 0] = 0
    base[:, 1] = grid
    decreasing = model.predict_proba(base)[:, 1]
    return int(np.sum(np.diff(increasing) < -1e-12) +
               np.sum(np.diff(decreasing) > 1e-12))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data()
    common = dict(max_iter=500, learning_rate=0.08, max_leaf_nodes=15,
                  l2_regularization=1.0, early_stopping=True, random_state=0)
    unconstrained = HistGradientBoostingClassifier(**common).fit(X_train, y_train)
    constrained = HistGradientBoostingClassifier(
        **common, monotonic_cst=[1, -1, 0, 0]).fit(X_train, y_train)

    for name, model in [('unconstrained', unconstrained), ('monotonic', constrained)]:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f'{name:13s} AUC {auc:.4f}, iterations {model.n_iter_:3d}, '
              f'grid violations {monotonic_violations(model)}')
