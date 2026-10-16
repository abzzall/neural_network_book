"""Check the reference project's evidence contract, independently of its score."""

import math

import numpy as np


def validate(result):
    """Raise ValueError for an incomplete or internally inconsistent record."""
    try:
        assert result['schema_version'] == 1
        digest = result['data_sha256']
        assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        partitions = result['audit']['partitions']
        ids = result['split_ids']
        combined = []
        for name in ('train', 'validation', 'test'):
            assert len(ids[name]) == partitions[name]['rows'] > 0
            assert all(type(row) is int for row in ids[name])
            assert 0 < partitions[name]['positive'] < len(ids[name])
            combined.extend(ids[name])
        assert sorted(combined) == list(range(result['audit']['rows']))
        assert result['audit']['nonfinite_values'] == result['audit']['duplicate_rows'] == 0
        assert len(result['scaler_mean']) == result['audit']['features']
        assert np.isfinite(result['scaler_mean']).all()
        seeds = result['seeds']
        assert len(seeds) >= 2 and len(seeds) == len(set(seeds))
        assert result['representative_seed'] == seeds[0]
        order = result['selection_order']
        assert order == ['linear', 'quadratic', 'mlp8', 'mlp16']
        assert set(result['validation']) == set(order)
        assert math.isfinite(result['selection_tolerance']) and result['selection_tolerance'] >= 0
        for section in ('validation', 'test'):
            for summary in result[section].values():
                values = summary['per_seed']
                assert len(values) == len(seeds)
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                assert np.isclose(summary['mean'], np.mean(values), rtol=0, atol=1e-12)
                assert np.isclose(summary['std'], np.std(values), rtol=0, atol=1e-12)
        best = max(result['validation'][name]['mean'] for name in order)
        selected = next(name for name in order if result['validation'][name]['mean']
                        >= best - result['selection_tolerance'])
        assert result['chosen'] == selected
        assert set(result['test']) == {'linear', selected}
        errors = result['representative_test_errors']
        assert len({row['row_id'] for row in errors}) == len(errors)
        assert all(row['row_id'] in ids['test'] and row['actual'] in (0, 1)
                   and row['predicted'] in (0, 1) and row['actual'] != row['predicted']
                   for row in errors)
        slice_ = result['low_signal_slice']
        assert 0 <= slice_['errors'] <= min(slice_['rows'], len(errors))
        assert slice_['rows'] <= len(ids['test'])
    except (AssertionError, KeyError, TypeError, ValueError, StopIteration) as error:
        raise ValueError('incomplete or inconsistent capstone evidence') from error
    return True
