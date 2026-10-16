"""A single graph-attention head; dense masks are for small graphs only."""

import torch
from torch import nn
from torch.nn import functional as F


class GraphAttention(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.projection = nn.Linear(inputs, outputs, bias=False)
        self.receiver = nn.Parameter(torch.empty(outputs))
        self.sender = nn.Parameter(torch.empty(outputs))
        nn.init.normal_(self.receiver, std=.1)
        nn.init.normal_(self.sender, std=.1)

    def forward(self, features, adjacency):
        hidden = self.projection(features)
        mask = adjacency.bool() | torch.eye(len(hidden), dtype=torch.bool,
                                             device=hidden.device)
        scores = F.leaky_relu((hidden @ self.receiver)[:, None]
                              + (hidden @ self.sender)[None, :], .2)
        coefficients = scores.masked_fill(~mask, -torch.inf).softmax(dim=1)
        return coefficients @ hidden, coefficients


if __name__ == '__main__':
    torch.manual_seed(42)
    adjacency = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    output, coefficients = GraphAttention(2, 3)(torch.eye(3, 2), adjacency)
    print('output shape:', tuple(output.shape))
    print('attention row sums:', coefficients.sum(1).detach().tolist())
    print('non-neighbour weight:', float(coefficients[0, 2].detach()))
