"""A small pre-normalised, decoder-only character model with no dropout."""

import torch
from torch import nn


class DecoderBlock(nn.Module):
    def __init__(self, width=24, heads=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                         nn.Linear(4 * width, width))

    def forward(self, x):
        length = x.shape[1]
        # MultiheadAttention: True forbids a key. SDPA uses the opposite rule.
        forbidden = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        normalised = self.norm1(x)
        mixed, _ = self.attention(normalised, normalised, normalised,
                                  attn_mask=forbidden, need_weights=False)
        x = x + mixed
        return x + self.feedforward(self.norm2(x))


class CharacterTransformer(nn.Module):
    def __init__(self, vocabulary, context=16, width=24):
        super().__init__()
        self.context = context
        self.tokens = nn.Embedding(vocabulary, width)
        self.positions = nn.Embedding(context, width)
        self.block = DecoderBlock(width)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocabulary)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.tokens(ids) + self.positions(positions)
        return self.output(self.norm(self.block(x)))
