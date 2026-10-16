"""Original synthetic homophilous graph; no downloaded data or labels as inputs."""

import numpy as np
import torch


def community_graph(seed=42, nodes=160):
    rng = np.random.default_rng(seed)
    labels = np.arange(nodes) % 2
    same = labels[:, None] == labels[None, :]
    upper = np.triu(rng.random((nodes, nodes)) < np.where(same, .12, .01), 1)
    adjacency = upper | upper.T
    features = rng.normal(size=(nodes, 8))
    features[:, 0] += .6 * (2 * labels - 1)
    partitions = [[], [], []]
    for category in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == category))
        for target, chunk in zip(partitions, np.split(indices, [16, 32])):
            target.extend(chunk)
    return (torch.tensor(adjacency), torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels), [torch.tensor(p) for p in partitions])
