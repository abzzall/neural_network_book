"""Chapter 6: the kind of target decides the output layer and the loss.

Two synthetic problems, each fitted by the same linear model with two different
output layers and losses:
  counts      -- identity output with squared error, or exp output with Poisson NLL;
  uncertainty -- one output (the mean) with squared error, or two outputs (mean and
                 log variance) with the Gaussian negative log-likelihood.
"""

import numpy as np

rng = np.random.default_rng(0)


def descend(gradient, w, steps=3000, rate=0.05):
    for _ in range(steps):
        w = w - rate * gradient(w)
    return w


# Counts: y ~ Poisson(exp(0.5 + x1 - 0.7 x2)), such as calls to a service per hour.
X = np.c_[np.ones(4000), rng.normal(size=(4000, 2))]
y = rng.poisson(np.exp(X @ [0.5, 1.0, -0.7]))
train, test = slice(0, 2000), slice(2000, None)
w_mse = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
w_poisson = descend(lambda w: X[train].T @ (np.exp(X[train] @ w) - y[train]) / 2000,
                    np.zeros(3))


def deviance(mu, y):
    """Mean Poisson deviance; it needs mu > 0, so predictions are clipped at 0.01."""
    mu = np.maximum(mu, 0.01)
    return 2 * np.mean(np.where(y > 0, y * np.log(np.maximum(y, 1) / mu), 0) - (y - mu))


# Uncertainty: y = x + noise whose spread grows with |x|.
x = rng.uniform(-3, 3, 4000)
t = x + rng.normal(size=4000) * (0.2 + 0.5 * np.abs(x))
F = np.c_[np.ones(4000), x, np.abs(x)]
mean_only = np.linalg.lstsq(F[train], t[train], rcond=None)[0]
sigma = np.std(t[train] - F[train] @ mean_only)                 # one spread for all inputs


def gaussian_gradient(w):
    """Gradient of the mean of 0.5 (log s2 + (t - m)^2 / s2) over m = F a, log s2 = F b."""
    a, b = w[:3], w[3:]
    m, s2 = F[train] @ a, np.exp(F[train] @ b)
    r = t[train] - m
    return np.r_[-F[train].T @ (r / s2), F[train].T @ (0.5 - 0.5 * r ** 2 / s2)] / 2000


w = descend(gaussian_gradient, np.zeros(6), steps=5000)


def coverage(m, s, region):
    inside = np.abs(t[test] - m) <= 1.96 * s
    return inside[region].mean()


if __name__ == '__main__':
    mu_mse, mu_poisson = X[test] @ w_mse, np.exp(X[test] @ w_poisson)
    print(f'counts, identity + squared error:  {np.mean(mu_mse < 0):.1%} of predictions '
          f'negative, Poisson deviance {deviance(mu_mse, y[test]):.3f}')
    print(f'counts, exp + Poisson likelihood:  {np.mean(mu_poisson < 0):.1%} of predictions '
          f'negative, Poisson deviance {deviance(mu_poisson, y[test]):.3f}')
    print(f'  learned weights {np.round(w_poisson, 3)}, true [0.5, 1.0, -0.7]')
    calm, wild = np.abs(x[test]) < 1, np.abs(x[test]) > 2
    m1, m2, s2 = F[test] @ mean_only, F[test] @ w[:3], np.exp(F[test] @ w[3:])
    print(f'95% interval, one spread for all:  covers {coverage(m1, sigma, calm):.1%} '
          f'where |x| < 1, {coverage(m1, sigma, wild):.1%} where |x| > 2')
    print(f'95% interval, predicted spread:    covers {coverage(m2, np.sqrt(s2), calm):.1%} '
          f'where |x| < 1, {coverage(m2, np.sqrt(s2), wild):.1%} where |x| > 2')
