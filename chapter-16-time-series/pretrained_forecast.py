"""Zero-shot forecasts from a pinned pretrained model (Chronos-Bolt tiny, Apache-2.0)."""

import argparse
from pathlib import Path

import numpy as np
import torch

from backtest import HORIZON, origins
from bike_data import load
from metrics import coverage, mae, mase, pinball

MODEL = 'amazon/chronos-bolt-tiny'
REVISION = 'a0e552de83495b5c28c14c71c374f3e33280b340'
CACHE = Path(__file__).resolve().parents[1] / 'data' / 'chronos-bolt-tiny-a0e552d'
LEVELS = [0.1, 0.5, 0.9]


def load_model(download=False, directory=CACHE):
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=directory,
                          allow_patterns=['config.json', 'model.safetensors', 'README.md'])
    if not (directory / 'model.safetensors').is_file():
        raise FileNotFoundError('Model missing: run pretrained_forecast.py --download first')
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(directory, device_map='cpu',
                                               dtype=torch.float32)


def quantile_forecasts(pipeline, y, starts):
    """10%, 50% and 90% forecasts from each origin; no training on this series."""
    contexts = [torch.tensor(y[:t], dtype=torch.float32) for t in starts]
    quantiles, _ = pipeline.predict_quantiles(contexts, prediction_length=HORIZON,
                                              quantile_levels=LEVELS)
    return np.stack([q.numpy() for q in quantiles])     # (origins, horizon, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    pipeline = load_model(parser.parse_args().download)
    y = load()['cnt'].to_numpy(float)
    starts = origins(len(y))
    q = quantile_forecasts(pipeline, y, starts)
    scores = []
    for i, t in enumerate(starts):
        actual = y[t:t + HORIZON]
        scores.append((mae(actual, q[i, :, 1]), mase(actual, q[i, :, 1], y[:t]),
                       coverage(actual, q[i, :, 0], q[i, :, 2]),
                       np.mean([pinball(actual, q[i, :, k], lv) for k, lv in enumerate(LEVELS)])))
    error, scaled, covered, loss = np.mean(scores, axis=0)
    print(f'median forecast: MAE {error:7.1f}  MASE {scaled:.3f}')
    print(f'80% interval coverage: {covered:.3f}')
    print(f'mean pinball loss over 3 quantiles: {loss:.1f}')
