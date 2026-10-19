"""Training windows for learned forecasters: past values plus known-future calendar."""

import numpy as np

LOOKBACK = 56


def window(y, workingday, t, horizon, lookback=LOOKBACK):
    """At first-target index t: history ends at t-1; future flags begin at t."""
    past = np.asarray(y[t - lookback:t], float)
    level = past.mean()                              # per-window scale
    return past / level, np.asarray(workingday[t:t + horizon], float), level


def training_set(y, workingday, end, horizon, lookback=LOOKBACK):
    """Every window whose target lies entirely before `end`."""
    past, calendar, target = [], [], []
    for s in range(lookback, end - horizon + 1):
        p, c, level = window(y, workingday, s, horizon, lookback)
        past.append(p)
        calendar.append(c)
        target.append(np.asarray(y[s:s + horizon]) / level)
    return np.array(past), np.array(calendar), np.array(target)
