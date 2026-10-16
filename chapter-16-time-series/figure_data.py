"""Write the data behind the chapter's plots into a given folder.

Usage: figure_data.py OUTPUT_FOLDER. Writes series.dat (daily rentals and their
7-day mean), acf.dat (autocorrelations), and fan.dat (quantile forecast from
the first backtest origin, 2012-07-01, with the actual values).
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'applied' / 'full-code'))
from backtest import FIRST_ORIGIN, HORIZON
from bike_data import load
from components import acf
import quantile_bikes as qb
from windows import training_set


def write(folder):
    table = load()
    y, workingday = table['cnt'].to_numpy(float), table['workingday'].to_numpy(float)
    day = np.arange(len(y))
    mean7 = np.convolve(y, np.ones(7) / 7, mode='full')[:len(y)]
    mean7[:6] = np.nan
    np.savetxt(folder / 'series.dat', np.column_stack([day, y, mean7]), fmt='%.6g',
               header='day cnt mean7', comments='')
    lags = np.arange(29)
    np.savetxt(folder / 'acf.dat', np.column_stack([lags, acf(y, 28), acf(np.diff(y), 28)]),
               fmt='%.6g', header='lag level difference', comments='')
    past, calendar, target = training_set(y, workingday, FIRST_ORIGIN, HORIZON)
    model = qb.train(qb.design(past, calendar), target)
    q = qb.quantile_forecast(model, y, workingday, FIRST_ORIGIN)
    history = np.arange(FIRST_ORIGIN - 28, FIRST_ORIGIN)
    future = np.arange(FIRST_ORIGIN, FIRST_ORIGIN + HORIZON)
    rows = [(d, y[d], np.nan, np.nan, np.nan) for d in history]
    rows += [(d, y[d], *q[i]) for i, d in enumerate(future)]
    np.savetxt(folder / 'fan.dat', np.array(rows), fmt='%.6g',
               header='day actual q10 q50 q90', comments='')
    return q


if __name__ == '__main__':
    q = write(Path(sys.argv[1]))
    print(f'first origin, day 1 quantiles: {q[0, 0]:.0f} {q[0, 1]:.0f} {q[0, 2]:.0f}')
