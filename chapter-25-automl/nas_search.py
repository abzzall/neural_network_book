"""A small, reproducible architecture search with a separate final test set."""

import time

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def candidates():
    for widths in ((32,), (64,), (64, 32), (128, 64)):
        for alpha in (1e-4, 1e-2):
            yield widths, alpha


if __name__ == '__main__':
    X, y = load_digits(return_X_y=True)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=1)

    start = time.perf_counter()
    results = []
    for widths, alpha in candidates():
        model = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=widths, alpha=alpha, early_stopping=True,
            max_iter=400, random_state=0))
        model.fit(X_train, y_train)
        results.append((accuracy_score(y_val, model.predict(X_val)), widths, alpha))

    validation, widths, alpha = max(results, key=lambda row: row[0])
    final = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=widths, alpha=alpha, early_stopping=True,
        max_iter=400, random_state=0)).fit(X_dev, y_dev)
    test = accuracy_score(y_test, final.predict(X_test))
    print(f'candidates {len(results)}, selected {widths}, alpha {alpha:g}')
    print(f'validation accuracy {validation:.4f}, final test accuracy {test:.4f}')
    print(f'search seconds {time.perf_counter() - start:.2f}')
