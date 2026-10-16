"""Duplicated messages leak from training into test data and inflate scores."""

import numpy as np
from sklearn.metrics import f1_score

from spam_data import load, split
from tfidf_baseline import baseline

texts, labels = load()
scores = {'random split, all test': [], 'random split, seen in training': [],
          'random split, unseen': [], 'deduplicated split': []}
for seed in range(10):                       # a range of splits, not one lucky seed
    train_x, test_x, train_y, test_y = split(texts, labels, deduplicate=False, seed=seed)
    predicted = baseline().fit(train_x, train_y).predict(test_x)
    seen = np.isin(test_x, train_x)
    test_y = np.array(test_y)
    scores['random split, all test'].append(f1_score(test_y, predicted))
    scores['random split, seen in training'].append(f1_score(test_y[seen], predicted[seen]))
    scores['random split, unseen'].append(f1_score(test_y[~seen], predicted[~seen]))
    train_x, test_x, train_y, test_y = split(texts, labels, deduplicate=True, seed=seed)
    predicted = baseline().fit(train_x, train_y).predict(test_x)
    scores['deduplicated split'].append(f1_score(test_y, predicted))

print('spam F1 over 10 splits (mean, min, max):')
for name, values in scores.items():
    print(f'{name}: {np.mean(values):.4f}, {min(values):.4f}, {max(values):.4f}')
