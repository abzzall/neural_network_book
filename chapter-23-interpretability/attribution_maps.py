"""Saliency and integrated gradients drawn as images, for the trained and a random network."""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from integrated_gradients import integrated_gradients, saliency, train  # noqa: E402

images, labels = load_digits(return_X_y=True)
tr, te, y_tr, y_te = train_test_split(images / 16, labels, test_size=0.3, random_state=0)
x_tr, x_te = torch.tensor(tr, dtype=torch.float32), torch.tensor(te, dtype=torch.float32)
model = train(x_tr, torch.tensor(y_tr))                     # as in integrated_gradients.py
torch.manual_seed(1)
random_model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))


def save(folder, name, values):
    grid = values.reshape(8, 8)
    rows = [(c, r, grid[r, c]) for r in range(8) for c in range(8)]
    np.savetxt(Path(folder) / f'{name}.dat', np.array(rows), fmt='%.4g',
               header='x y v', comments='')


if __name__ == '__main__':
    for k in range(3):
        x, t = x_te[k], int(y_te[k])
        predicted = model(x[None]).argmax().item()
        sal = saliency(model, x, t).numpy()
        ig = integrated_gradients(model, x, t).detach().numpy()
        ig_random = integrated_gradients(random_model, x, t).detach().numpy()
        share = ig[ig > 0].sum() / np.abs(ig).sum()
        print(f'test image {k}: label {t}, predicted {predicted}, '
              f'positive share of |IG| {share:.3f}')
        if len(sys.argv) > 1:
            save(sys.argv[1], f'digit{k}', x.numpy())
            save(sys.argv[1], f'saliency{k}', sal / sal.max())
            save(sys.argv[1], f'ig{k}', ig / np.abs(ig).max())
            save(sys.argv[1], f'igrandom{k}', ig_random / np.abs(ig_random).max())
