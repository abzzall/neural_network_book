"""Synthetic data and models reused from the earlier chapters."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from nnbook.activations import ReLU
from nnbook.layers import Dense
from nnbook.losses import SoftmaxCrossEntropy
from nnbook.network import Network
from nnbook.optimizers import Adam


def synthetic_data(rows=1200):
    """Original independent synthetic rows; no external data or licence needed."""
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(rows, 6))
    signal = x[:, 0] * x[:, 1] + 0.35 * x[:, 2] - 0.2 * x[:, 3]
    y = (signal + rng.normal(0., 0.2, rows) > 0).astype(int)
    return x, y


def fit_candidate(name, x, y, seed, epochs=160):
    if name == 'linear':
        return LogisticRegression(C=1., max_iter=1000).fit(x, y)
    if name == 'quadratic':
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             LogisticRegression(C=1., max_iter=1000)).fit(x, y)
    width = {'mlp8': 8, 'mlp16': 16}[name]
    rng = np.random.default_rng(seed)
    model = Network([Dense(x.shape[1], width, rng, init='he'), ReLU(),
                     Dense(width, 2, rng, init='glorot')], SoftmaxCrossEntropy())
    return model.fit(x, y, epochs=epochs, batch_size=64, seed=seed,
                     optimizer=Adam(lr=0.01, weight_decay=0.001))
