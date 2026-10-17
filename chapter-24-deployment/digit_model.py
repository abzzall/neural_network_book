"""The chapter's model: a small CNN for 8x8 digits, trained once and cached."""

from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'chapter-12-cnn'))
from digits_data import accuracy, fit, load_images, shift_right   # noqa: F401  (re-exported)

WEIGHTS = ROOT / 'outputs' / 'chapter24_cnn.pt'


def cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 2 * 2, 10))


def trained_model():
    """Load the cached weights, or train (seed 0) and save them first."""
    model = cnn()
    if WEIGHTS.is_file():
        model.load_state_dict(torch.load(WEIGHTS, weights_only=True))
        return model.eval()
    X_train, y_train, _, _ = load_images()
    model = fit(cnn, X_train, y_train, epochs=30, seed=0)
    WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)
    return model
