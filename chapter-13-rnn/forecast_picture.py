"""The seasonal-correction GRU of the applied exercise against the data, for a figure."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'applied' / 'full-code'))
from recurrent_forecast import fit, make_series, make_windows  # noqa: E402

torch.set_num_threads(1)
x, y, baseline, origins, scale = make_windows(make_series())
model, _ = fit(x, y, baseline, origins, scale, seed=42)       # the first seed
with torch.inference_mode():
    gru = model(x, baseline)

if __name__ == '__main__':
    for name, mask in (('validation', (origins >= 260) & (origins < 340)),
                       ('test', origins >= 340)):
        seasonal = (baseline[mask] - y[mask]).abs().mean().item()
        learned = (gru[mask] - y[mask]).abs().mean().item()
        print(f'{name}: seasonal MAE {seasonal:.4f}, GRU MAE (seed 42) {learned:.4f}')
    if len(sys.argv) > 1:
        keep = origins >= 300
        rows = np.column_stack([origins[keep], y[keep].numpy(), baseline[keep].numpy(),
                                gru[keep].numpy()])
        np.savetxt(Path(sys.argv[1]) / 'forecast.dat', rows, fmt='%.5g',
                   header='t actual seasonal gru', comments='')
