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


def predict(p, x, y):
    hidden = [(p[3*j] * x + p[3*j+1] * y + p[3*j+2]).relu() for j in range(4)]
    z = sum(p[12+j] * h for j, h in enumerate(hidden)) + p[16]
    return (1 + (-z).exp()) ** -1


def objective(p):
    return sum((predict(p, x, y) - target) ** 2 for x, y, target in DATA) / len(DATA)


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
    for _ in range(2000):
        for v in p:
            v.grad = 0.
        loss = objective(p)
        loss.backward()
        for v in p:
            v.data -= 0.5 * v.grad
    probabilities = [predict(p, x, y).data for x, y, _ in DATA]
    assert all(math.isfinite(v) for v in probabilities)
    return first_loss, objective(p).data, difference, probabilities


if __name__ == '__main__':
    first, final, difference, probabilities = run()
    print(f'initial MSE: {first:.6f}')
    print(f'final MSE: {final:.6f}')
    print(f'first gradient difference: {difference:.2e}')
    print('probabilities: ' + ', '.join(f'{v:.4f}' for v in probabilities))
