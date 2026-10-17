"""Three model families on three data sets, by 5-fold cross-validation."""

from pathlib import Path
import sys
import time

import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'chapter-23-interpretability'))
from credit_data import load as load_credit


def families(prepare):
    return {'logistic regression': make_pipeline(prepare(), LogisticRegression(max_iter=5000)),
            'gradient boosting': make_pipeline(prepare(), HistGradientBoostingClassifier(
                random_state=0)),
            'neural network (MLP)': make_pipeline(prepare(), MLPClassifier(
                hidden_layer_sizes=(64,), max_iter=2000, random_state=0))}


def datasets():
    X, good, _ = load_credit()
    categorical = list(X.select_dtypes('object').columns)
    numeric = [c for c in X.columns if c not in categorical]
    credit = lambda: make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
        (StandardScaler(), numeric))
    cancer, digits = load_breast_cancer(), load_digits()
    return {'credit (table, 1000 rows)': (X, good, credit, 'roc_auc'),
            'breast cancer (table, 569 rows)': (cancer.data, cancer.target, StandardScaler,
                                                'roc_auc'),
            'digits (8x8 images, 1797)': (digits.data, digits.target, StandardScaler,
                                          'accuracy')}


if __name__ == '__main__':
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, (X, y, prepare, metric) in datasets().items():
        print(f'{name}, {metric}:')
        for family, model in families(prepare).items():
            start = time.perf_counter()
            scores = cross_val_score(model, X, y, cv=folds, scoring=metric)
            print(f'  {family:22s} {scores.mean():.4f} +- {scores.std():.4f} '
                  f'({time.perf_counter() - start:.1f} s)')
