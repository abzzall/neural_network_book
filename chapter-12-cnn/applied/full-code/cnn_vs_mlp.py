"""Chapter 12 applied exercise: a CNN against an MLP, on original and shifted digits.

Data: UCI Optical Recognition of Handwritten Digits (Alpaydin and Kaynak, 1998,
CC BY 4.0), as distributed with scikit-learn.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the chapter folder
from digits_data import accuracy, fit, load_images, n_params, shift_right  # noqa: E402,F401


# Step 1: the multilayer perceptron of chapter 6, reading the 8x8 image as 64 numbers.
def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))


# Step 2: two convolution-ReLU-pooling stages and a linear classifier.
def cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 x 4 x 4
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 x 2 x 2
        nn.Flatten(), nn.Linear(32 * 2 * 2, 10))


# Step 3: shift a batch by -1, 0 or +1 pixels in each direction (data augmentation).
def random_shift(x, generator):
    dx, dy = torch.randint(-1, 2, (2,), generator=generator).tolist()
    out = torch.zeros_like(x)
    out[..., max(0, dy):8 - max(0, -dy), max(0, dx):8 - max(0, -dx)] = \
        x[..., max(0, -dy):8 - max(0, dy), max(0, -dx):8 - max(0, dx)]
    return out


def fit_augmented(build, X, y, epochs=30, seed=0):
    torch.manual_seed(seed)
    model = build()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(y), generator=generator).split(32):
            optimizer.zero_grad()
            loss_fn(model(random_shift(X[idx], generator)), y[idx]).backward()
            optimizer.step()
    return model.eval()


# Step 4: mean accuracy over three seeds on the original and on the shifted test images.
def evaluate(builder, train_fn, seeds=(0, 1, 2)):
    X_train, y_train, X_test, y_test = load_images()
    X_shifted = shift_right(X_test)
    plain, shifted = [], []
    for seed in seeds:
        model = train_fn(builder, X_train, y_train, epochs=30, seed=seed)
        plain.append(accuracy(model, X_test, y_test))
        shifted.append(accuracy(model, X_shifted, y_test))
    return float(np.mean(plain)), float(np.mean(shifted))
