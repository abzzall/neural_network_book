"""Rolling one-step forecasts of a synthetic demand series, with a seasonal baseline.

Training targets: before 260; validation: 260..339; test: 340..419.
Test windows may contain earlier observed test values, never their own targets.
"""

import numpy as np
import torch
from torch import nn


def make_series():
    rng = np.random.default_rng(42)
    t = np.arange(420)
    return 20 + 0.03 * t + 4 * np.sin(2 * np.pi * t / 7) + 2 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 0.25, len(t))


def training_statistics(series):
    raise NotImplementedError("Step 1: complete training_statistics")


def history_windows(series, origins, lookback, mean, scale):
    raise NotImplementedError("Step 2: complete history_windows")


def seasonal_baseline(series, origins):
    raise NotImplementedError("Step 3: complete seasonal_baseline")


def make_windows(series, lookback=28):
    if not 7 <= lookback < 260:
        raise ValueError('lookback must preserve the seasonal lag and training windows')
    mean, scale = training_statistics(series)
    origins = np.arange(lookback, len(series))
    windows = history_windows(series, origins, lookback, mean, scale)
    return (torch.tensor(windows[..., None], dtype=torch.float32),
            torch.tensor(series[origins], dtype=torch.float32),
            torch.tensor(seasonal_baseline(series, origins), dtype=torch.float32), origins, float(scale))


class Forecaster(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.recurrent = nn.GRU(1, 16, batch_first=True)
        self.output = nn.Linear(16, 1)
        self.scale = scale

    def forward(self, x, baseline):
        raise NotImplementedError("Step 4: run the GRU over the window")
        raise NotImplementedError("Step 5: map the final output to a correction")
        raise NotImplementedError("Step 6: add the scaled correction to the baseline")


def fit(x, y, baseline, origins, scale, seed=42, epochs=250):
    torch.manual_seed(seed)
    model = Forecaster(scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train, validation = origins < 260, (origins >= 260) & (origins < 340)
    best, best_epoch, state = float('inf'), 0, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = ((model(x[train], baseline[train]) - y[train]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_mae = (model(x[validation], baseline[validation]) - y[validation]).abs().mean().item()
        if validation_mae < best:
            best, best_epoch = validation_mae, epoch + 1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval(), best_epoch


def run():
    torch.set_num_threads(1)
    x, y, baseline, origins, scale = make_windows(make_series())
    predictions, selected = [], []
    for seed in (42, 43, 44):
        model, epoch = fit(x, y, baseline, origins, scale, seed=seed)
        selected.append(epoch)
        with torch.inference_mode():
            predictions.append(model(x, baseline))
    result = {'train_windows': int((origins < 260).sum()), 'selected_epoch_seed42': selected[0]}
    for name, mask in (('validation', (origins >= 260) & (origins < 340)), ('test', origins >= 340)):
        result[f'{name}_seasonal_mae'] = (baseline[mask] - y[mask]).abs().mean().item()
        errors = [(prediction[mask] - y[mask]).abs().mean().item() for prediction in predictions]
        result[f'{name}_gru_mae'] = float(np.mean(errors))
        result[f'{name}_gru_std'] = float(np.std(errors))
    return result


if __name__ == '__main__':
    for name, value in run().items():
        print(f'{name}: {value:.6g}')
