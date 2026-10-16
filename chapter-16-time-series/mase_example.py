"""Seasonal-naive and naive forecasts scored by MAE and MASE on twelve values."""

import numpy as np

from baselines import naive, seasonal_naive
from metrics import mae, mase

history = np.array([10, 14, 18, 12, 11, 15, 20, 13])    # two cycles of four periods
actual = np.array([12, 16, 19, 16])                     # the next cycle
for name, forecast in (('seasonal naive', seasonal_naive(history, 4, m=4)),
                       ('naive', naive(history, 4))):
    print(f'{name}: forecast {forecast.astype(int).tolist()}, '
          f'MAE {mae(actual, forecast):.2f}, MASE {mase(actual, forecast, history, m=4):.2f}')
print(f'scale (in-sample seasonal-naive MAE): {np.mean(np.abs(history[4:] - history[:4])):.2f}')
