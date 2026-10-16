"""INCOMPLETE exercise starter. Probabilistic 14-day forecasts of bike rentals: linear quantile model vs naive interval.

The quantile model is trained once on windows before 2012-07-01 and evaluated
at 25 weekly origins; the interval should contain about 80% of actual values.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backtest import FIRST_ORIGIN, HORIZON, origins
from baselines import seasonal_naive
from bike_data import load
from metrics import coverage, mase
from windows import training_set, window

LEVELS = (0.1, 0.5, 0.9)


def pinball_loss(y, q, levels=LEVELS):
    """Step 1: mean pinball loss; y has shape (..., H), q has shape (..., H, 3)."""
    raise NotImplementedError("Step 1: complete pinball_loss")


def design(past, calendar):
    """Step 2: scaled past values, a constant, and the known-future calendar."""
    raise NotImplementedError("Step 2: complete design")


def train(X, target, seed=42, steps=4000, lr=2e-2):
    """Full-batch Adam with a cosine-decaying rate, long enough for seeds to agree."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(X.shape[1], HORIZON * len(LEVELS))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    X, target = torch.tensor(X, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    for _ in range(steps):
        # Step 3: pinball loss of the reshaped outputs, then one Adam step.
        raise NotImplementedError("Step 3: compute the loss and update the weights")
        schedule.step()
    return model


def quantile_forecast(model, y, workingday, t):
    """Step 4: rescale the three quantiles and sort them so they never cross."""
    raise NotImplementedError("Step 4: complete quantile_forecast")


def naive_interval(history):
    """Step 5: seasonal naive plus quantiles of its past 14-day errors (last year)."""
    raise NotImplementedError("Step 5: complete naive_interval")


def scores(actual, q, history):
    loss = pinball_loss(torch.tensor(actual), torch.tensor(q)).item()
    return mase(actual, q[:, 1], history), coverage(actual, q[:, 0], q[:, 2]), loss


def run():
    torch.set_num_threads(4)
    table = load()
    y, workingday = table['cnt'].to_numpy(float), table['workingday'].to_numpy(float)
    past, calendar, target = training_set(y, workingday, FIRST_ORIGIN, HORIZON)
    model = train(design(past, calendar), target)
    model_scores, naive_scores = [], []
    for t in origins(len(y)):          # Step 6: score both methods at every origin
        raise NotImplementedError("Step 6: score the quantile model and the naive interval")
    m, n = np.mean(model_scores, axis=0), np.mean(naive_scores, axis=0)
    return {'train_windows': len(past), 'origins': len(origins(len(y))),
            'quantile_mase': m[0], 'quantile_coverage': m[1], 'quantile_pinball': m[2],
            'naive_mase': n[0], 'naive_coverage': n[1], 'naive_pinball': n[2]}


if __name__ == '__main__':
    print(json.dumps({k: float(v) for k, v in run().items()}, indent=1))
