"""A temporal convolutional network: causal dilated convolutions over the past."""

import numpy as np
import torch
from torch import nn


def receptive_field(kernel_size, dilations):
    """Past steps one output can see, eq. (16.6)."""
    return 1 + (kernel_size - 1) * sum(dilations)


class CausalConv(nn.Module):
    """Pad on the left only, so output t never sees inputs after t."""

    def __init__(self, c_in, c_out, dilation, kernel_size=3):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCN(nn.Module):
    def __init__(self, horizon, channels=16, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.first = CausalConv(1, channels, dilations[0])
        self.blocks = nn.ModuleList(CausalConv(channels, channels, d) for d in dilations[1:])
        self.head = nn.Linear(channels + horizon, horizon)   # + known-future calendar

    def forward(self, past, calendar):
        h = torch.relu(self.first(past[:, None, :]))         # (batch, channels, time)
        for block in self.blocks:
            h = h + torch.relu(block(h))                      # residual connection
        return self.head(torch.cat([h[:, :, -1], calendar], dim=1))


def train(past, calendar, target, seed, epochs=300, lr=3e-3):
    torch.manual_seed(seed)                                   # seed before building
    model = TCN(target.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = [torch.tensor(a, dtype=torch.float32) for a in (past, calendar, target)]
    for _ in range(epochs):
        loss = nn.functional.l1_loss(model(*inputs[:2]), inputs[2])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model.eval()


def predict(model, past, calendar):
    with torch.inference_mode():
        x = [torch.tensor(np.atleast_2d(a), dtype=torch.float32) for a in (past, calendar)]
        return model(*x).numpy()
