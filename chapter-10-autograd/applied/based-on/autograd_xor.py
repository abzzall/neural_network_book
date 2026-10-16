"""Train a scalar 2-4-1 XOR network and check its first gradients with torch."""

import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from value import Value

DATA = [(-1., -1., 0.), (-1., 1., 1.), (1., -1., 1.), (1., 1., 0.)]


def hidden_values(p, x, y):
    raise NotImplementedError("Step 1: compute the four ReLU hidden activations")


def predict(p, x, y):
    raise NotImplementedError("Step 2: compute the sigmoid of the output affine function")


def objective(p):
    raise NotImplementedError("Step 3: average the four squared errors")


def clear_gradients(p):
    raise NotImplementedError("Step 4: clear every parameter gradient")


def update(p, learning_rate):
    raise NotImplementedError("Step 5: apply one gradient-descent update")


def train(p, steps=2000, learning_rate=0.5):
    for _ in range(steps):
        raise NotImplementedError("Step 6: clear parameter gradients")
        raise NotImplementedError("Step 7: build a fresh objective")
        raise NotImplementedError("Step 8: differentiate the loss")
        raise NotImplementedError("Step 9: apply the parameter update")


def run():
    rng = random.Random(42)
    initial = [rng.uniform(-1, 1) for _ in range(17)]
    p = [Value(v) for v in initial]
    loss = objective(p)
    loss.backward()
    t = torch.tensor(initial, dtype=torch.double, requires_grad=True)
    errors = []
    for x, y, target in DATA:
        h = torch.relu(t[:12].reshape(4, 3) @ torch.tensor([x, y, 1.], dtype=torch.double))
        errors.append((torch.sigmoid(t[12:16] @ h + t[16]) - target) ** 2)
    torch.stack(errors).mean().backward()
    difference = float(np.max(np.abs(np.array([v.grad for v in p]) - t.grad.numpy())))
    first_loss = loss.data
    train(p)
    probabilities = [predict(p, x, y).data for x, y, _ in DATA]
    assert all(math.isfinite(v) for v in probabilities)
    return first_loss, objective(p).data, difference, probabilities


if __name__ == '__main__':
    first, final, difference, probabilities = run()
    print(f'initial MSE: {first:.6f}')
    print(f'final MSE: {final:.6f}')
    print(f'first gradient difference: {difference:.2e}')
    print('probabilities: ' + ', '.join(f'{v:.4f}' for v in probabilities))
