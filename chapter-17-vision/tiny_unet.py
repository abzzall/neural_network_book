"""A small same-padded U-Net with explicit alignment for odd spatial sizes."""

import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1, self.encoder2 = block(1, 4), block(4, 8)
        self.pool = nn.MaxPool2d(2)
        self.bottom = block(8, 16)
        self.up2, self.decoder2 = nn.ConvTranspose2d(16, 8, 2, stride=2), block(16, 8)
        self.up1, self.decoder1 = nn.ConvTranspose2d(8, 4, 2, stride=2), block(8, 4)
        self.output = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        first = self.encoder1(x)
        second = self.encoder2(self.pool(first))
        deep = self.bottom(self.pool(second))
        up = self.up2(deep)
        up = F.interpolate(up, size=second.shape[-2:], mode='nearest')
        decoded = self.decoder2(torch.cat([second, up], dim=1))
        up = self.up1(decoded)
        up = F.interpolate(up, size=first.shape[-2:], mode='nearest')
        return self.output(self.decoder1(torch.cat([first, up], dim=1)))
