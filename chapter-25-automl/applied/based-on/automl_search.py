"""INCOMPLETE exercise starter. Exhaustive search versus successive halving, with one untouched test set.

Ten candidates (five hidden-layer shapes, two weight penalties) are searched
on the digits. Weights are fitted on the training split, candidates are ranked
on the validation split, and only the two selected procedures are refitted on
all development rows and scored once on the test split. The cost of a search
is recorded as allocated candidate-epochs, actual fitting epochs and wall time.
Each halving round restarts its survivors from the same seed.
"""

import json
import warnings
import time

from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SHAPES = ((32,), (64,), (64, 32), (128, 64), (128, 64, 32))
ALPHAS = (1e-4, 1e-2)
FULL_EPOCHS = 200


def candidates():
    """Step 1: every combination of a shape and a penalty, in a fixed order."""
    raise NotImplementedError("Step 1: complete candidates")


def splits():
    """The splits of nas_search.py: 60 % train, 20 % validation, 20 % test."""
    X, y = load_digits(return_X_y=True)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=1)
    return X_train, y_train, X_val, y_val, X_dev, y_dev, X_test, y_test


def fit(candidate, epochs, X, y):
    shape, alpha = candidate
    model = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=shape, alpha=alpha, max_iter=epochs, random_state=0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)   # short budgets stop early
        return model.fit(X, y)


MEASUREMENTS = []


def validation_scores(cands, epochs, data):
    X_train, y_train, X_val, y_val = data[:4]
    scores = []
    for candidate in cands:
        started = time.perf_counter()
        model = fit(candidate, epochs, X_train, y_train)
        MEASUREMENTS.append((int(model[-1].n_iter_), time.perf_counter() - started))
        scores.append(accuracy_score(y_val, model.predict(X_val)))
    return scores


def exhaustive(cands, data):
    """Step 2: train every candidate for the full budget; keep the best on validation."""
    raise NotImplementedError("Step 2: complete exhaustive")


def successive_halving(cands, data, min_epochs=25, eta=2):
    """Step 3: train all briefly, keep the best 1/eta, multiply the budget by eta, repeat."""
    raise NotImplementedError("Step 3: complete successive_halving")


def final_test(candidate, data):
    """Step 4: refit the selected procedure on all development rows; score the test once."""
    raise NotImplementedError("Step 4: complete final_test")


def run():
    data, cands = splits(), candidates()
    out = {'candidates': len(cands)}
    for name, search in (('exhaustive', exhaustive), ('halving', successive_halving)):
        MEASUREMENTS.clear()
        (shape, alpha), validation, cost = search(cands, data)
        out[f'{name}_actual_epochs'] = sum(row[0] for row in MEASUREMENTS)
        out[f'{name}_fit_seconds'] = round(sum(row[1] for row in MEASUREMENTS), 3)
        out[f'{name}_shape'] = list(shape)
        out[f'{name}_alpha'] = alpha
        out[f'{name}_validation'] = round(validation, 4)
        out[f'{name}_test'] = round(final_test((shape, alpha), data), 4)
        out[f'{name}_epochs'] = cost
    return out


if __name__ == '__main__':
    print(json.dumps(run(), indent=1))
