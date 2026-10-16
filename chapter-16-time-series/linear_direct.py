"""Direct multi-output linear forecaster (ridge) with and without the calendar."""

import numpy as np

from backtest import FIRST_ORIGIN, HORIZON, origins
from bike_data import load
from metrics import mae, mase
from windows import training_set, window


def design(past, calendar, use_calendar):
    parts = [past, np.ones((len(past), 1))] + ([calendar] if use_calendar else [])
    return np.hstack(parts)


def fit(y, workingday, end, use_calendar=True, penalty=1.0):
    """One weight column per horizon step, fitted on windows before `end`."""
    past, calendar, target = training_set(y, workingday, end, HORIZON)
    X = design(past, calendar, use_calendar)
    return np.linalg.solve(X.T @ X + penalty * np.eye(X.shape[1]), X.T @ target)


def forecast(W, y, workingday, t, use_calendar=True):
    past, calendar, level = window(y, workingday, t, HORIZON)
    return design(past[None], calendar[None], use_calendar)[0] @ W * level


if __name__ == '__main__':
    table = load()
    y, workingday = table['cnt'].to_numpy(float), table['workingday'].to_numpy(float)
    for use_calendar in (False, True):
        W = fit(y, workingday, FIRST_ORIGIN, use_calendar)   # trained once
        scores = [(mae(y[t:t + HORIZON], f), mase(y[t:t + HORIZON], f, y[:t]))
                  for t in origins(len(y))
                  for f in [forecast(W, y, workingday, t, use_calendar)]]
        error, scaled = np.mean(scores, axis=0)
        name = 'with calendar' if use_calendar else 'past values only'
        print(f'linear, {name:16s} MAE {error:7.1f}  MASE {scaled:.3f}')
