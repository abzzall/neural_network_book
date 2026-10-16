"""Chapter 12: the digits as 1x8x8 images, split as in chapters 5 to 9, and training helpers.

Data: UCI Optical Recognition of Handwritten Digits (Alpaydin and Kaynak, 1998,
CC BY 4.0), as distributed with scikit-learn. Pixel values 0..16 are scaled to 0..1.
"""

import torch
from sklearn.datasets import load_digits

from nnbook.preprocessing import train_test_split


def load_images():
    X, y = load_digits(return_X_y=True)
    train, test = train_test_split(len(y), test_size=0.25, seed=0, stratify=y)
    images = torch.tensor(X / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)
    labels = torch.tensor(y)
    return images[train], labels[train], images[test], labels[test]


def shift_right(images, pixels=1):
    """Move every image right by some pixels, filling with background (zero)."""
    shifted = torch.zeros_like(images)
    shifted[..., pixels:] = images[..., :-pixels]
    return shifted


def fit(build, X, y, epochs=30, seed=0, lr=0.05):
    """Seed, build the model with build(), and train it with mini-batch SGD and momentum.

    The model is built after seeding, so its initial weights depend on the seed
    and on nothing else that ran before.
    """
    torch.manual_seed(seed)
    model = build()
    order_gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(y), generator=order_gen).split(32):
            optimizer.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            optimizer.step()
    return model.eval()


def accuracy(model, X, y):
    with torch.inference_mode():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


