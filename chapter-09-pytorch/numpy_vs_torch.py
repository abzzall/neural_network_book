"""Chapter 9: the same network in nnbook and in PyTorch gives the same gradients."""

import numpy as np
import torch
from torch import nn

from nnbook import Dense, Network, ReLU, SoftmaxCrossEntropy

rng = np.random.default_rng(0)
X, y = rng.normal(size=(16, 8)), rng.integers(0, 3, 16)

ours = Network([Dense(8, 5, rng), ReLU(), Dense(5, 3, rng)], SoftmaxCrossEntropy())
theirs = nn.Sequential(nn.Linear(8, 5), nn.ReLU(), nn.Linear(5, 3)).double()
with torch.no_grad():                                      # copy the weights across
    for linear, dense in ((theirs[0], ours.layers[0]), (theirs[2], ours.layers[2])):
        linear.weight.copy_(torch.from_numpy(dense.W))     # both store (outputs, inputs)
        linear.bias.copy_(torch.from_numpy(dense.b))

loss_ours = ours.loss_value(X, y)
ours.backward()
loss_theirs = nn.functional.cross_entropy(theirs(torch.from_numpy(X)), torch.from_numpy(y))
loss_theirs.backward()

if __name__ == "__main__":
    print(f"loss: nnbook {loss_ours:.10f}, PyTorch {loss_theirs.item():.10f}")
    diff = max(np.abs(dense.dW - linear.weight.grad.numpy()).max()
               for linear, dense in ((theirs[0], ours.layers[0]), (theirs[2], ours.layers[2])))
    print("weight gradients agree to 1e-12:", bool(diff < 1e-12))
