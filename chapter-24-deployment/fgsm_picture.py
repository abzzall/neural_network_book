"""One digit before and after an FGSM attack, with the change itself, saved as images."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from digit_model import load_images, trained_model  # noqa: E402
from fgsm_demo import fgsm  # noqa: E402


def save(folder, name, image):
    grid = image.reshape(8, 8)
    rows = [(c, r, grid[r, c]) for r in range(8) for c in range(8)]
    np.savetxt(Path(folder) / f'{name}.dat', np.array(rows), fmt='%.4g',
               header='x y v', comments='')


if __name__ == '__main__':
    torch.set_num_threads(1)
    model = trained_model()
    _, _, X_test, y_test = load_images()
    epsilon = 0.1
    adversarial = fgsm(model, X_test, y_test, epsilon)
    with torch.inference_mode():
        before = model(X_test).argmax(1)
        after = model(adversarial).argmax(1)
    k = int(((before == y_test) & (after != y_test)).nonzero()[0])   # first digit it fools
    change = (adversarial[k] - X_test[k]).flatten()
    print(f'test digit {k}: label {y_test[k].item()}, predicted {before[k].item()} before, '
          f'{after[k].item()} after the attack')
    print(f'largest pixel change {change.abs().max().item():.2f} of a range of 1')
    if len(sys.argv) > 1:
        save(sys.argv[1], 'clean', X_test[k].numpy())
        save(sys.argv[1], 'change', (change / epsilon).numpy())    # -1, 0 or +1
        save(sys.argv[1], 'attacked', adversarial[k].numpy())
