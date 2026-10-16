"""Backtest the TCN, trained once on data before the first origin, for three seeds."""

import numpy as np
import torch

from backtest import FIRST_ORIGIN, HORIZON, origins
from bike_data import load
from metrics import mae, mase
from tcn import predict, receptive_field, train
from windows import training_set, window

torch.set_num_threads(4)
table = load()
y, workingday = table['cnt'].to_numpy(float), table['workingday'].to_numpy(float)
past, calendar, target = training_set(y, workingday, FIRST_ORIGIN, HORIZON)
print(f'training windows: {len(past)}, receptive field: {receptive_field(3, [1, 2, 4, 8, 16])} days')
results = []
for seed in (42, 43, 44):
    model = train(past, calendar, target, seed)
    scores = []
    for t in origins(len(y)):
        p, c, level = window(y, workingday, t, HORIZON)
        f = predict(model, p, c)[0] * level
        scores.append((mae(y[t:t + HORIZON], f), mase(y[t:t + HORIZON], f, y[:t])))
    results.append(np.mean(scores, axis=0))
    print(f'seed {seed}: MAE {results[-1][0]:7.1f}  MASE {results[-1][1]:.3f}')
mean, spread = np.mean(results, axis=0), np.std(results, axis=0)
print(f'mean over seeds: MAE {mean[0]:.1f}  MASE {mean[1]:.3f} (std {spread[1]:.3f})')
