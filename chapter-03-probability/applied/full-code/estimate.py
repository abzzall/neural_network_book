"""Chapter 3 applied exercise: maximum likelihood by formula and by gradient descent."""

import numpy as np


def make_data(seed=42):
    """200 clicks (1) or non-clicks (0), and 200 delivery times in hours."""
    rng = np.random.default_rng(seed)
    clicks = (rng.random(200) < 0.3).astype(float)
    times = rng.normal(5.0, 2.0, 200)
    return clicks, times


# Step 1: the Bernoulli maximum-likelihood estimate is the fraction of ones.
def bernoulli_mle(clicks):
    return clicks.mean()


# Step 2: average binary cross-entropy of the data under probability p, in nats.
def bernoulli_nll(clicks, p):
    return -np.mean(clicks * np.log(p) + (1 - clicks) * np.log(1 - p))


# Step 3: minimise it by gradient descent on p.
def bernoulli_gd(clicks, eta=0.05, steps=500):
    p = 0.5
    for _ in range(steps):
        grad = -np.mean(clicks / p - (1 - clicks) / (1 - p))
        p -= eta * grad
    return p


# Step 4: the Gaussian maximum-likelihood estimates: the mean and the variance divided by n.
def gaussian_mle(times):
    mu = times.mean()
    return mu, np.mean((times - mu) ** 2)


# Step 5: average Gaussian negative log-likelihood, with s = log(variance).
def gaussian_nll(times, mu, s):
    return 0.5 * np.log(2 * np.pi) + 0.5 * s + 0.5 * np.mean((times - mu) ** 2) * np.exp(-s)


# Step 6: minimise it by gradient descent on (mu, s).
def gaussian_gd(times, eta=0.5, steps=2000):
    mu, s = 0.0, 0.0
    for _ in range(steps):
        r = times - mu
        grad_mu = -np.mean(r) * np.exp(-s)
        grad_s = 0.5 - 0.5 * np.mean(r ** 2) * np.exp(-s)
        mu, s = mu - eta * grad_mu, s - eta * grad_s
    return mu, np.exp(s)
