"""A tanh RNN with sequence cross-entropy and backpropagation through time.

Shapes: x=(time,batch,input), targets=(time,batch), hidden=(time,batch,hidden).
Weights use input-by-output layout; a single bias is used for each affine map.
"""

import numpy as np


class RNN:
    def __init__(self, inputs, hidden, outputs, seed=42):
        rng = np.random.default_rng(seed)
        self.Wx = rng.normal(0, 1 / np.sqrt(inputs), (inputs, hidden))
        self.Wh = rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden))
        self.bh = np.zeros(hidden)
        self.Wy = rng.normal(0, 1 / np.sqrt(hidden), (hidden, outputs))
        self.by = np.zeros(outputs)

    def parameters(self):
        return (self.Wx, self.Wh, self.bh, self.Wy, self.by)

    def forward(self, x, h0=None):
        x = np.asarray(x, dtype=float)
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[2] != self.Wx.shape[0]:
            raise ValueError('x must have nonempty (time, batch, input) shape')
        h0 = np.zeros((x.shape[1], self.Wh.shape[0])) if h0 is None else np.asarray(h0)
        if h0.shape != (x.shape[1], self.Wh.shape[0]):
            raise ValueError('h0 must have (batch, hidden) shape')
        states = [h0.copy()]
        for xt in x:
            states.append(np.tanh(xt @ self.Wx + states[-1] @ self.Wh + self.bh))
        hidden = np.stack(states[1:])
        logits = hidden @ self.Wy + self.by
        self.cache = (x, np.stack(states))
        return logits, hidden

    def loss(self, x, targets, h0=None):
        logits, _ = self.forward(x, h0)
        targets = np.asarray(targets)
        if targets.shape != logits.shape[:2] or not np.issubdtype(targets.dtype, np.integer):
            raise ValueError('targets must be integer indices with (time, batch) shape')
        if np.any(targets < 0) or np.any(targets >= logits.shape[-1]):
            raise ValueError('target index outside output vocabulary')
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        self.probs, self.targets = np.exp(log_probs), targets.copy()
        return float(-np.take_along_axis(log_probs, targets[..., None], axis=-1).mean())

    def backward(self):
        x, states = self.cache
        time, batch, _ = x.shape
        dlogits = self.probs.copy().reshape(time * batch, -1)
        dlogits[np.arange(time * batch), self.targets.ravel()] -= 1
        dlogits = dlogits.reshape(self.probs.shape) / (time * batch)
        dWx, dWh, dbh, dWy, dby = [np.zeros_like(p) for p in self.parameters()]
        dh = np.zeros_like(states[0])
        dx = np.empty_like(x)
        for t in reversed(range(time)):
            dWy += states[t + 1].T @ dlogits[t]
            dby += dlogits[t].sum(axis=0)
            dh = dh + dlogits[t] @ self.Wy.T
            da = dh * (1 - states[t + 1] ** 2)
            dWx += x[t].T @ da
            dWh += states[t].T @ da
            dbh += da.sum(axis=0)
            dx[t] = da @ self.Wx.T
            dh = da @ self.Wh.T
        return (dWx, dWh, dbh, dWy, dby), dx, dh
