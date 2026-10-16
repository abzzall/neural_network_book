"""Fixed-graph transductive classification: depth chosen on validation only."""

import copy
import json
import torch
from torch import nn

from gcn_torch import GCNLayer, sparse_operator
from graph_data import community_graph


def run(seed=42, depth=2, epochs=200, graph_seed=42, loss_fn=None):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    adjacency, x, y, (train, validation, test) = community_graph(graph_seed)
    edges = adjacency.nonzero().T
    operator = sparse_operator(edges, len(x))
    widths = [8] + [16] * max(depth - 1, 0) + [2]
    layers = nn.ModuleList(GCNLayer(a, b) for a, b in zip(widths, widths[1:]))
    if depth == 0:
        layers = nn.ModuleList([nn.Linear(8, 16), nn.Linear(16, 2)])
    optimiser = torch.optim.Adam(layers.parameters(), lr=.01, weight_decay=.0005)

    def predict():
        hidden = x
        for index, layer in enumerate(layers):
            hidden = layer(operator, hidden) if depth else layer(hidden)
            if index + 1 < len(layers):
                hidden = hidden.relu()
        return hidden

    best, saved, best_epoch = float('inf'), None, None
    for epoch in range(epochs):
        layers.train()
        optimiser.zero_grad(set_to_none=True)
        logits = predict()
        loss = (loss_fn(logits, y, train) if loss_fn else
                nn.functional.cross_entropy(logits[train], y[train]))
        loss.backward()
        optimiser.step()
        layers.eval()
        with torch.inference_mode():
            validation_loss = nn.functional.cross_entropy(predict()[validation], y[validation]).item()
        if validation_loss < best:
            best, saved, best_epoch = validation_loss, copy.deepcopy(layers.state_dict()), epoch + 1
    layers.load_state_dict(saved)
    layers.eval()
    with torch.inference_mode():
        prediction = predict().argmax(1)
    return dict(seed=seed, depth=depth, epoch=best_epoch, validation_loss=best,
                validation_accuracy=float((prediction[validation] == y[validation]).float().mean()),
                test_accuracy=float((prediction[test] == y[test]).float().mean()))


if __name__ == '__main__':
    print(json.dumps([run(seed, depth) for seed in (42, 43, 44)
                      for depth in (0, 1, 2, 4, 8)], indent=2))
