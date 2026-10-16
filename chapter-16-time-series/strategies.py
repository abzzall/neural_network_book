"""Recursive versus direct multi-step forecasting with the same linear inputs."""

import numpy as np

from backtest import FIRST_ORIGIN, HORIZON, origins
from bike_data import load
from metrics import mae
from windows import LOOKBACK

y = load()['cnt'].to_numpy(float)


def windows(end, steps):
    X = np.array([np.append(y[s - LOOKBACK:s] / y[s - LOOKBACK:s].mean(), 1)
                  for s in range(LOOKBACK, end - steps + 1)])
    Y = np.array([y[s:s + steps] / y[s - LOOKBACK:s].mean()
                  for s in range(LOOKBACK, end - steps + 1)])
    return X, Y


def ridge(X, Y, penalty=1.0):
    return np.linalg.solve(X.T @ X + penalty * np.eye(X.shape[1]), X.T @ Y)


one_step = ridge(*windows(FIRST_ORIGIN, 1))          # predicts only the next day
direct = ridge(*windows(FIRST_ORIGIN, HORIZON))      # one column per horizon step
errors = {'recursive': [], 'direct': []}
for t in origins(len(y)):
    history = list(y[:t])
    for _ in range(HORIZON):                         # feed predictions back as inputs
        past = np.array(history[-LOOKBACK:])
        history.append((np.append(past / past.mean(), 1) @ one_step)[0] * past.mean())
    errors['recursive'].append(np.abs(y[t:t + HORIZON] - history[t:]))
    past = y[t - LOOKBACK:t]
    errors['direct'].append(np.abs(y[t:t + HORIZON] - np.append(past / past.mean(), 1)
                                   @ direct * past.mean()))
for name, e in errors.items():
    e = np.array(e)
    print(f'{name:9s} MAE day 1: {e[:, 0].mean():6.1f}  day 14: {e[:, -1].mean():6.1f}'
          f'  all days: {e.mean():6.1f}')
