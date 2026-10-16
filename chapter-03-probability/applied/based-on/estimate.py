"""Chapter 3 applied exercise (starting point): maximum likelihood by formula and by gradient descent."""

import numpy as np


def make_data(seed=42):
    """200 clicks (1) or non-clicks (0), and 200 delivery times in hours."""
    rng = np.random.default_rng(seed)
    clicks = (rng.random(200) < 0.3).astype(float)
    times = rng.normal(5.0, 2.0, 200)
    return clicks, times


# Step 1: the Bernoulli maximum-likelihood estimate is the fraction of ones.
def bernoulli_mle(clicks):
    raise NotImplementedError("Step 1: return the fraction of ones")


# Step 2: average binary cross-entropy of the data under probability p, in nats.
def bernoulli_nll(clicks, p):
    raise NotImplementedError("Step 2: average of -[c log p + (1 - c) log(1 - p)]")


# Step 3: minimise it by gradient descent on p.
def bernoulli_gd(clicks, eta=0.05, steps=500):
    raise NotImplementedError("Step 3: gradient descent on p from p = 0.5")


# Step 4: the Gaussian maximum-likelihood estimates: the mean and the variance divided by n.
def gaussian_mle(times):
    raise NotImplementedError("Step 4: the mean, and the mean squared deviation divided by n")


# Step 5: average Gaussian negative log-likelihood, with s = log(variance).
def gaussian_nll(times, mu, s):
    raise NotImplementedError("Step 5: the average Gaussian negative log-likelihood with s = log(variance)")


# Step 6: minimise it by gradient descent on (mu, s).
def gaussian_gd(times, eta=0.5, steps=2000):
    raise NotImplementedError("Step 6: gradient descent on (mu, s) from (0, 0); return (mu, exp(s))")
