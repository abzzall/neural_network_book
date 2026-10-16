"""Chapter 12: does reusing features help? Digits 0-4 as the source, 5-9 as the target.

A small CNN is trained on all training images of digits 0-4. For digits 5-9
only ten training images per class are available. Three networks learn them:
one from scratch, one that reuses the trained convolutional layers frozen and
trains only a new final layer, and one that fine-tunes all layers from the
transferred weights. Results are averaged over five seeds.
"""

import numpy as np
from torch import nn

from digits_data import accuracy, fit, load_images


def cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 2 * 2, 5))


def transferred(base, freeze):
    model = cnn()
    model.load_state_dict(base.state_dict())
    model[-1] = nn.Linear(32 * 2 * 2, 5)                  # new head for the new classes
    if freeze:
        for p in list(model.parameters())[:-2]:
            p.requires_grad_(False)                        # keep the transferred features
    return model


Xtr, ytr, Xte, yte = load_images()
low_tr, low_te = ytr < 5, yte < 5
X_high, y_high = Xte[~low_te], yte[~low_te] - 5           # test images of 5-9, labels 0-4
base = fit(cnn, Xtr[low_tr], ytr[low_tr], epochs=20, seed=0)

results = {"from scratch": [], "frozen features": [], "fine-tuned": []}
for seed in range(5):
    rng = np.random.default_rng(seed)
    few = np.concatenate([rng.choice(np.flatnonzero(ytr.numpy() == d), 10, replace=False)
                          for d in range(5, 10)])
    X_few, y_few = Xtr[few], ytr[few] - 5
    results["from scratch"].append(accuracy(fit(cnn, X_few, y_few, epochs=60, seed=seed), X_high, y_high))
    results["frozen features"].append(
        accuracy(fit(lambda: transferred(base, True), X_few, y_few, epochs=60, seed=seed), X_high, y_high))
    results["fine-tuned"].append(
        accuracy(fit(lambda: transferred(base, False), X_few, y_few, epochs=60, seed=seed, lr=0.01),
                 X_high, y_high))

if __name__ == "__main__":
    print(f"source task (digits 0-4) test accuracy: {accuracy(base, Xte[low_te], yte[low_te]):.3f}")
    for name, accs in results.items():
        print(f"target task (digits 5-9, 50 images), {name:15s} {np.mean(accs):.3f} +- {np.std(accs):.3f}")
