"""Patch projection, a class token and one bidirectional transformer block."""

import torch
from torch import nn


class TinyViT(nn.Module):
    def __init__(self, image_size=8, patch_size=2, width=32, classes=10):
        super().__init__()
        if image_size % patch_size:
            raise ValueError('image size must be divisible by patch size')
        self.patches = nn.Conv2d(1, width, patch_size, stride=patch_size)
        tokens = (image_size // patch_size) ** 2 + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.positions = nn.Parameter(torch.randn(1, tokens, width) * .02)
        layer = nn.TransformerEncoderLayer(width, 4, dim_feedforward=64,
                    dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)

    def forward(self, images):
        patches = self.patches(images).flatten(2).transpose(1, 2)
        token = self.class_token.expand(len(images), -1, -1)
        tokens = torch.cat([token, patches], dim=1) + self.positions
        return self.head(self.norm(self.encoder(tokens)[:, 0]))
