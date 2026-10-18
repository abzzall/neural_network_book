"""Chapter 24: flagging inputs unlike the training data, by confidence and by distance."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from digit_model import cnn, fit, load_images  # noqa: E402

torch.set_num_threads(1)
X_train, y_train, X_test, y_test = load_images()
known = y_train != 9                                   # the model never sees a 9
X_fit, y_fit = X_train[known][:900], y_train[known][:900]
X_val = X_train[known][900:]                           # unseen digits 0 to 8, for thresholds
model = fit(cnn, X_fit, y_fit, epochs=30, seed=0)
features = model[:-1]                                  # everything before the last layer

with torch.inference_mode():
    bank = features(X_fit)

    def confidence(x):                                 # high means "looks familiar"
        return model(x).softmax(dim=1).amax(dim=1)

    def closeness(x):                                  # minus the distance to the nearest
        return -torch.cdist(features(x), bank).min(dim=1).values   # training digit

    generator = torch.Generator().manual_seed(0)
    inputs = {'new digits 0-8': X_test[y_test != 9], 'the digit 9': X_test[y_test == 9],
              'random noise': torch.rand(178, 1, 8, 8, generator=generator),
              'inverted digits': 1 - X_test[y_test != 9]}

if __name__ == '__main__':
    with torch.inference_mode():
        for name, score in (('confidence', confidence), ('distance', closeness)):
            # Flag the lowest 5 % of validation digits: a 5 % false-alarm target.
            threshold = torch.quantile(score(X_val), 0.05)
            rates = ', '.join(f'{label} {(score(x) < threshold).float().mean().item():.3f}'
                              for label, x in inputs.items())
            print(f'{name:10s} flagged: {rates}')
