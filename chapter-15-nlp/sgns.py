"""Skip-gram with negative sampling: loss, gradients, and one SGD step."""

import numpy as np


def sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def loss_and_grads(v_center, u_context, u_negatives):
    """Loss for one (centre, context) pair and k negative words, eq. (15.5)."""
    pos_score, neg_scores = u_context @ v_center, u_negatives @ v_center
    pos = sigmoid(pos_score)                    # probability the true pair is real
    neg = sigmoid(neg_scores)                   # probabilities the noise pairs are real
    loss = np.logaddexp(0, -pos_score) + np.logaddexp(0, neg_scores).sum()
    grad_v = (pos - 1) * u_context + neg @ u_negatives
    grad_u = (pos - 1) * v_center
    grad_negatives = neg[:, None] * v_center
    return loss, grad_v, grad_u, grad_negatives


def sgd_step(V, U, center, context, negatives, lr):
    loss, gv, gu, gn = loss_and_grads(V[center], U[context], U[negatives])
    V[center] -= lr * gv
    U[context] -= lr * gu
    np.subtract.at(U, negatives, lr * gn)        # a word may be drawn twice
    return loss


def pairs(sentences, window):
    for sentence in sentences:
        for i, center in enumerate(sentence):
            lo, hi = max(0, i - window), min(len(sentence), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    yield center, sentence[j]


def noise_distribution(sentences, size, power=0.75):
    counts = np.bincount(np.concatenate(sentences), minlength=size)
    weights = counts ** power
    return weights / weights.sum()
