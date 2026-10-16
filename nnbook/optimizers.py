"""Optimisers and learning-rate schedules (chapter 7).

An optimiser receives the list of (value, gradient) pairs of a network and
updates every value in place. Its state (velocities, averages) is kept per
parameter, in the same order as the list.
"""

import math

import numpy as np


class SGD:
    """theta <- theta - lr * g. With momentum, a velocity accumulates past gradients."""

    def __init__(self, lr=0.1, momentum=0.0, nesterov=False):
        self.lr, self.momentum, self.nesterov = lr, momentum, nesterov
        self.velocity = None

    def step(self, params):
        if self.momentum == 0.0:
            for value, grad in params:
                value -= self.lr * grad
            return
        if self.velocity is None:
            self.velocity = [np.zeros_like(v) for v, _ in params]
        for (value, grad), vel in zip(params, self.velocity):
            vel *= self.momentum
            vel += grad
            value -= self.lr * (grad + self.momentum * vel if self.nesterov else vel)


class AdaGrad:
    """Divide each step by the root of the sum of all past squared gradients."""

    def __init__(self, lr=0.1, eps=1e-8):
        self.lr, self.eps, self.sum_sq = lr, eps, None

    def step(self, params):
        if self.sum_sq is None:
            self.sum_sq = [np.zeros_like(v) for v, _ in params]
        for (value, grad), s in zip(params, self.sum_sq):
            s += grad ** 2
            value -= self.lr * grad / (np.sqrt(s) + self.eps)


class RMSProp:
    """Divide each step by the root of a running average of squared gradients."""

    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.avg_sq = lr, rho, eps, None

    def step(self, params):
        if self.avg_sq is None:
            self.avg_sq = [np.zeros_like(v) for v, _ in params]
        for (value, grad), s in zip(params, self.avg_sq):
            s *= self.rho
            s += (1 - self.rho) * grad ** 2
            value -= self.lr * grad / (np.sqrt(s) + self.eps)


class Adam:
    """Running averages of the gradient (m) and its square (v), bias-corrected.

    weight_decay > 0 gives AdamW: the decay is applied to the weights directly,
    not added to the gradient.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.weight_decay, self.t, self.m, self.v = weight_decay, 0, None, None

    def step(self, params):
        if self.m is None:
            self.m = [np.zeros_like(v) for v, _ in params]
            self.v = [np.zeros_like(v) for v, _ in params]
        self.t += 1
        for (value, grad), m, v in zip(params, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * grad ** 2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            if self.weight_decay:
                value -= self.lr * self.weight_decay * value
            value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def step_decay(base_lr, epoch, drop=0.5, every=10):
    """Multiply the learning rate by `drop` every `every` epochs."""
    return base_lr * drop ** (epoch // every)


def cosine(base_lr, epoch, total, warmup=0):
    """Linear warm-up for `warmup` epochs, then cosine decay to zero at `total`."""
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))
