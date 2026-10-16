"""Chapter 12: a residual block, y = relu(x + F(x))."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x):
        return torch.relu(x + self.body(x))           # the skip connection adds the input back


if __name__ == "__main__":
    torch.manual_seed(0)
    block = ResidualBlock(8).eval()
    x = torch.randn(2, 8, 16, 16)
    print("input shape:", tuple(x.shape), " output shape:", tuple(block(x).shape))
    nn.init.zeros_(block.body[4].weight)               # last batch-norm scale set to zero
    print("with F(x) = 0 the block passes relu(x) through:", torch.equal(block(x), torch.relu(x)))
