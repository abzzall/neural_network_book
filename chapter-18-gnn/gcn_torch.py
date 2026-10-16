"""Sparse COO GCN propagation; row is receiver, column is sender."""

import numpy as np
import torch
from torch import nn


def sparse_operator(edges, nodes, dtype=torch.float32):
    edges = torch.as_tensor(edges, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2 or nodes < 1:
        raise ValueError('expected a (2, E) edge list and positive node count')
    if edges.numel() and (edges.min() < 0 or edges.max() >= nodes):
        raise ValueError('edge index outside node range')
    # Input is a simple undirected graph: symmetrise, deduplicate, add loops.
    loops = torch.arange(nodes).repeat(2, 1)
    indices = torch.cat([edges, edges.flip(0), loops], dim=1)
    raw = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]),
                                  (nodes, nodes), check_invariants=True).coalesce()
    indices = raw.indices()
    degrees = torch.bincount(indices[0], minlength=nodes).to(dtype)
    values = (degrees[indices[0]] * degrees[indices[1]]).rsqrt()
    return torch.sparse_coo_tensor(indices, values, (nodes, nodes),
                                   check_invariants=True).coalesce()


class GCNLayer(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs, bias=False)

    def forward(self, operator, features):
        return torch.sparse.mm(operator, self.linear(features))


if __name__ == '__main__':
    from gcn_numpy import example, normalise
    adjacency, features, weights = example()
    edges = torch.from_numpy(np.stack(adjacency.nonzero())).long()
    operator = sparse_operator(edges, 4, torch.float64)
    layer = GCNLayer(2, 1).double()
    with torch.no_grad():
        layer.linear.weight.copy_(torch.tensor(weights.T))
    output = layer(operator, torch.tensor(features)).relu()
    print('GCN output:', [round(x, 6) for x in output.flatten().tolist()])
    reference = torch.tensor(normalise(adjacency) @ features @ weights)
    print('dense and sparse agree:', torch.allclose(output, reference))
