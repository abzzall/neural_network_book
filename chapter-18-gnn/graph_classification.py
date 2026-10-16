"""Disjoint-union batching and a graph-size readout classification example."""

import torch

from gcn_torch import sparse_operator


def cycle(size):
    return torch.stack([torch.arange(size), torch.arange(size).roll(-1)])


def pooled(features, membership, graphs, mean=False):
    output = features.new_zeros(graphs, features.shape[1])
    output.index_add_(0, membership, features)
    if mean:
        counts = torch.bincount(membership, minlength=graphs).clamp_min(1)
        output = output / counts[:, None]
    return output


def demo():
    # Two degree-two graphs with identical initial features.
    edges = torch.cat([cycle(3), cycle(6) + 3], dim=1)
    operator = sparse_operator(edges, 9)
    membership = torch.tensor([0] * 3 + [1] * 6)
    embeddings = torch.sparse.mm(operator, torch.ones(9, 1))
    sums = pooled(embeddings, membership, 2)
    means = pooled(embeddings, membership, 2, mean=True)
    # A fixed size-based rule illustrates readout information, not learnt accuracy.
    return sums.flatten(), means.flatten(), (sums.flatten() > 4.5).long()


if __name__ == '__main__':
    sums, means, predictions = demo()
    print('sum readout:', sums.tolist())
    print('mean readout:', means.tolist())
    print('size-rule predictions:', predictions.tolist())
