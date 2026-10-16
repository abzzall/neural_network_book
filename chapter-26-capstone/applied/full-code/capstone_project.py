"""A complete, auditable classification project with a locked selection rule."""

import hashlib
import json
from pathlib import Path
import sys
from importlib.metadata import version
import platform

import numpy as np
from sklearn.preprocessing import StandardScaler

CHAPTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHAPTER))
from capstone_support import fit_candidate, synthetic_data

SEEDS = (42, 43, 44)
CANDIDATES = ('linear', 'quadratic', 'mlp8', 'mlp16')


def partition_ids(rows, seed=42):
    order = np.random.default_rng(seed).permutation(rows)
    train_end, validation_end = 3 * rows // 5, 4 * rows // 5
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def fit_scaler(x, train_ids):
    return StandardScaler().fit(x[train_ids])


def macro_f1(actual, predicted):
    scores = []
    for label in (0, 1):
        tp = np.sum((actual == label) & (predicted == label))
        fp = np.sum((actual != label) & (predicted == label))
        fn = np.sum((actual == label) & (predicted != label))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.)
    return float(np.mean(scores))


def choose_candidate(validation, tolerance=0.01):
    best = max(validation[name]['mean'] for name in CANDIDATES)
    return next(name for name in CANDIDATES
                if validation[name]['mean'] >= best - tolerance)


def error_rows(ids, actual, predicted):
    return [{'row_id': int(row), 'actual': int(a), 'predicted': int(p)}
            for row, a, p in zip(ids, actual, predicted) if a != p]


def audit_data(x, y, splits):
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError('expected a finite feature matrix and one target per row')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('expected binary targets 0 and 1')
    joined = np.concatenate(splits)
    if not np.array_equal(np.sort(joined), np.arange(len(x))):
        raise ValueError('partitions must cover each row exactly once')
    if any(len(part) == 0 or len(np.unique(y[part])) != 2 for part in splits):
        raise ValueError('each partition must contain both classes')
    duplicates = len(x) - len(np.unique(x, axis=0))
    if duplicates:
        raise ValueError('duplicate feature rows require a split policy')
    return {'rows': len(x), 'features': x.shape[1], 'duplicate_rows': duplicates,
            'nonfinite_values': 0,
            'partitions': {name: {'rows': len(part), 'positive': int(y[part].sum())}
                           for name, part in zip(('train', 'validation', 'test'), splits)}}


def run(x=None, y=None, epochs=160):
    if x is None and y is None:
        x, y = synthetic_data()
    elif x is None or y is None:
        raise ValueError('supply both x and y')
    train, validation_ids, test = partition_ids(len(x))
    audit = audit_data(x, y, (train, validation_ids, test))
    scaler = fit_scaler(x, train)
    train_x, val_x = scaler.transform(x[train]), scaler.transform(x[validation_ids])
    models, validation = {}, {}
    for name in CANDIDATES:
        models[name] = [fit_candidate(name, train_x, y[train], seed, epochs)
                        for seed in SEEDS]
        scores = [macro_f1(y[validation_ids], model.predict(val_x))
                  for model in models[name]]
        validation[name] = {'per_seed': scores, 'mean': float(np.mean(scores)),
                            'std': float(np.std(scores, ddof=0))}
    chosen = choose_candidate(validation)
    # Only after selection do we transform and score the final test partition.
    test_x = scaler.transform(x[test])
    tested = {}
    for name in dict.fromkeys(('linear', chosen)):
        predictions = [model.predict(test_x) for model in models[name]]
        scores = [macro_f1(y[test], p) for p in predictions]
        tested[name] = {'per_seed': scores, 'mean': float(np.mean(scores)),
                        'std': float(np.std(scores, ddof=0))}
    representative = models[chosen][0].predict(test_x)
    errors = error_rows(test, y[test], representative)
    low_signal = np.abs(x[test, 0] * x[test, 1]) < 0.25
    diagnostic = {'rows': int(low_signal.sum()),
                  'errors': int((representative[low_signal] != y[test][low_signal]).sum())}
    header = json.dumps({'x_shape': x.shape, 'y_shape': y.shape}, sort_keys=True).encode()
    payload = header + np.asarray(x, dtype='<f8').tobytes() + np.asarray(y, dtype='<i8').tobytes()
    return {'schema_version': 1, 'data_sha256': hashlib.sha256(payload).hexdigest(),
            'audit': audit, 'seeds': list(SEEDS), 'epochs': epochs,
            'data_seed': 2026, 'split_seed': 42,
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'scikit_learn': version('scikit-learn')},
            'split_ids': {name: part.tolist() for name, part in
                          zip(('train', 'validation', 'test'), (train, validation_ids, test))},
            'selection_tolerance': 0.01, 'selection_order': list(CANDIDATES),
            'validation': validation, 'chosen': chosen, 'test': tested,
            'scaler_mean': scaler.mean_.tolist(), 'representative_seed': SEEDS[0],
            'representative_test_errors': errors, 'low_signal_slice': diagnostic}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    results = run()
    text = json.dumps(results, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)
