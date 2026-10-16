"""Scaled dot-product attention and its backward pass, without broadcasting."""

import numpy as np


def attention(q, k, v, allowed=None):
    """Last axes are (positions, features); True means an allowed key."""
    q, k, v = (np.asarray(x, dtype=float) for x in (q, k, v))
    if (min(q.ndim, k.ndim, v.ndim) < 2 or q.shape[:-2] != k.shape[:-2]
            or k.shape[:-2] != v.shape[:-2] or q.shape[-1] != k.shape[-1]
            or k.shape[-2] != v.shape[-2] or q.shape[-1] == 0):
        raise ValueError('incompatible attention shapes')
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1])
    if allowed is not None:
        allowed = np.broadcast_to(np.asarray(allowed, dtype=bool), scores.shape)
        if not allowed.any(axis=-1).all():
            raise ValueError('each query must have an allowed key')
        scores = np.where(allowed, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v, (q, k, v, weights)


def backward(cache, arriving):
    q, k, v, weights = cache
    dv = weights.swapaxes(-1, -2) @ arriving
    da = arriving @ v.swapaxes(-1, -2)
    ds = weights * (da - (da * weights).sum(axis=-1, keepdims=True))
    dq = ds @ k / np.sqrt(q.shape[-1])
    dk = ds.swapaxes(-1, -2) @ q / np.sqrt(q.shape[-1])
    return dq, dk, dv


def multihead(x, wq, wk, wv, wo, heads, allowed=None):
    """Bias-free self-attention for x shaped (batch, positions, width)."""
    batch, length, width = x.shape
    if heads < 1 or width % heads:
        raise ValueError('width must be divisible by a positive head count')
    def split(w):
        return (x @ w).reshape(batch, length, heads, width // heads).transpose(0, 2, 1, 3)
    out, _ = attention(split(wq), split(wk), split(wv), allowed)
    return out.transpose(0, 2, 1, 3).reshape(batch, length, width) @ wo
