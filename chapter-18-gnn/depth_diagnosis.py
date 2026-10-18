"""Why the eight-layer GCN fails: watch its signal, its gradient and its training loss."""

import torch
from torch import nn

from gcn_torch import GCNLayer, sparse_operator
from graph_data import community_graph


def diagnose(depth, seed=42, epochs=200):
    torch.set_num_threads(1)
    torch.manual_seed(seed)                          # as in depth_experiment.py
    adjacency, x, y, (train, _, _) = community_graph(42)
    operator = sparse_operator(adjacency.nonzero().T, len(x))
    widths = [8] + [16] * (depth - 1) + [2]
    layers = nn.ModuleList(GCNLayer(a, b) for a, b in zip(widths, widths[1:]))
    optimiser = torch.optim.Adam(layers.parameters(), lr=.01, weight_decay=.0005)

    def forward():
        hidden = x
        for index, layer in enumerate(layers):
            hidden = layer(operator, hidden)
            if index + 1 < len(layers):
                hidden = hidden.relu()
                last_hidden = hidden
        return hidden, last_hidden

    record = {}
    for epoch in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        logits, last_hidden = forward()
        loss = nn.functional.cross_entropy(logits[train], y[train])
        loss.backward()
        if epoch == 0:
            record['first-layer gradient norm'] = layers[0].linear.weight.grad.norm().item()
            record['signal size at start'] = last_hidden.abs().mean().item()
            record['relative spread across nodes at start'] = (
                last_hidden.std(dim=0).mean() / last_hidden.abs().mean()).item()
        optimiser.step()
    logits, last_hidden = forward()
    record['training loss at end'] = nn.functional.cross_entropy(logits[train], y[train]).item()
    record['signal size at end'] = last_hidden.abs().mean().item()
    record['spread across nodes at end'] = last_hidden.std(dim=0).mean().item()
    return record


if __name__ == '__main__':
    for depth in (4, 8):
        for name, value in diagnose(depth).items():
            print(f'depth {depth}: {name} = {value:.3g}')
