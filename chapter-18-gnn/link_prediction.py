"""Leakage checks and negative sampling for undirected link prediction."""

import numpy as np
import torch

from gcn_torch import sparse_operator


def negative_edges(nodes, positives, count, seed=42):
    positives = {tuple(sorted(edge)) for edge in positives}
    candidates = [(u, v) for u in range(nodes) for v in range(u + 1, nodes)
                  if (u, v) not in positives]
    if count < 0 or count > len(candidates):
        raise ValueError('not enough distinct non-edges')
    chosen = np.random.default_rng(seed).choice(len(candidates), count, replace=False)
    return [candidates[index] for index in chosen]


def demo():
    observed = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]
    training, held_out = observed[:-2], observed[-2:]
    # Both directions of held-out positives are absent before normalisation.
    operator = sparse_operator(torch.tensor(training).T, 5)
    negatives = negative_edges(5, observed, 3)
    dense = operator.to_dense()
    absent = all(dense[u, v] == 0 and dense[v, u] == 0 for u, v in held_out)
    torch.manual_seed(42)
    embeddings = torch.sparse.mm(operator, torch.randn(5, 4))
    pairs = held_out + negatives
    scores = [(embeddings[u] * embeddings[v]).sum().sigmoid().item() for u, v in pairs]
    return negatives, bool(absent), scores


if __name__ == '__main__':
    negatives, absent, scores = demo()
    print('negative edges:', negatives)
    print('held-out edges absent from propagation:', absent)
    print('untrained decoder scores:', [round(score, 6) for score in scores])
