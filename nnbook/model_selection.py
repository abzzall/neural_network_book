"""Cross-validation splits (chapter 8)."""

import numpy as np


def kfold(n, k=5, seed=0, stratify=None):
    """Yield (train, validation) index arrays for k folds.

    With stratify, each fold receives the same share of every class.
    """
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    groups = [np.arange(n)] if stratify is None else \
        [np.flatnonzero(stratify == c) for c in np.unique(stratify)]
    offset = 0
    for members in groups:
        for i, idx in enumerate(rng.permutation(members)):
            folds[(i + offset) % k].append(idx)
        offset += len(members)
    for f in range(k):
        val = np.sort(folds[f])
        train = np.sort(np.concatenate([folds[g] for g in range(k) if g != f]))
        yield train, val
