"""Rolling-origin backtest of the baselines on daily bike rentals."""

import numpy as np

from baselines import holt_winters, moving_average, naive, seasonal_naive
from bike_data import load
from metrics import mae, mase

HORIZON, FIRST_ORIGIN, STEP = 14, 547, 7      # forecasts from 2012-07-01, weekly


def origins(n, horizon=HORIZON, first=FIRST_ORIGIN, step=STEP):
    """Forecast origins t: train on y[:t], score y[t:t+horizon]."""
    return list(range(first, n - horizon + 1, step))


def backtest(y, method):
    errors = []
    for t in origins(len(y)):
        forecast = method(y[:t], HORIZON)
        errors.append((mae(y[t:t + HORIZON], forecast),
                       mase(y[t:t + HORIZON], forecast, y[:t])))
    return np.mean(errors, axis=0)


if __name__ == '__main__':
    y = load()['cnt'].to_numpy(float)
    print(f'origins: {len(origins(len(y)))}, horizon: {HORIZON} days')
    methods = {'naive': naive, 'seasonal naive': seasonal_naive,
               '7-day mean': moving_average,
               'Holt-Winters': lambda history, h: holt_winters(history, h)[0]}
    for name, method in methods.items():
        error, scaled = backtest(y, method)
        print(f'{name:15s} MAE {error:7.1f}  MASE {scaled:.3f}')
