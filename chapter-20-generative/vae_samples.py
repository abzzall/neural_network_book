"""Draw new digits from two trained VAEs and save them as one picture each."""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae import sample, train  # noqa: E402

torch.set_num_threads(4)
images, labels = load_digits(return_X_y=True)
images = images / 16.0
train_x, _, train_y, _ = train_test_split(images, labels, test_size=0.25,
                                          random_state=0, stratify=labels)
judge = LogisticRegression(max_iter=5000).fit(train_x, train_y)   # as in vae_demo.py


def mosaic(digits, per_row=10):
    """Arrange 8 x 8 digits in rows, one blank pixel between neighbours."""
    rows = len(digits) // per_row
    out = np.zeros((rows * 9 - 1, per_row * 9 - 1))
    for k, d in enumerate(digits):
        r, c = divmod(k, per_row)
        out[r * 9:r * 9 + 8, c * 9:c * 9 + 8] = d.reshape(8, 8)
    return out


if __name__ == '__main__':
    for latent in (2, 8):
        model = train(train_x, latent=latent, seed=0)
        drawn = sample(model, 30, seed=0)
        names = ''.join(str(d) for d in judge.predict(drawn))
        print(f'latent {latent}: the classifier reads the 30 samples as {names}')
        if len(sys.argv) > 1:
            grid = mosaic(drawn)
            rows = [(c, r, grid[r, c]) for r in range(grid.shape[0])
                    for c in range(grid.shape[1])]
            np.savetxt(Path(sys.argv[1]) / f'vae_samples_latent{latent}.dat', np.array(rows),
                       fmt='%.4g', header='x y v', comments='')
