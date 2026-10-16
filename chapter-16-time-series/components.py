"""Autocorrelation of daily rentals before and after differencing."""

import numpy as np

from bike_data import load


def acf(y, max_lag):
    """Sample autocorrelation r_k for k = 0..max_lag, eq. (16.2)."""
    d = np.asarray(y, float) - np.mean(y)
    denominator = d @ d
    return np.array([d[k:] @ d[:len(d) - k] / denominator for k in range(max_lag + 1)])


if __name__ == '__main__':
    y = load()['cnt'].to_numpy(float)
    lags = [1, 2, 7, 14, 28]
    for name, series in (('rentals', y), ('first difference', np.diff(y)),
                         ('weekly difference', y[7:] - y[:-7])):
        r = acf(series, max(lags))
        print(f'{name:18s}', ' '.join(f'r{k}={r[k]:+.3f}' for k in lags))
    bound = 1.96 / np.sqrt(len(y))
    print(f'approximate 95% band for white noise: +-{bound:.3f}')
