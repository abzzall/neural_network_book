"""The ELBO for a one-dimensional Gaussian model, in closed form and by sampling.

Model: z ~ N(0, 1), x | z ~ N(z, 1). Observed x = 2. Approximate posterior
q(z) = N(mu, s^2). Then p(x) = N(0, 2) and the true posterior is N(x/2, 1/2).
"""

import math

import numpy as np

x = 2.0


def elbo_terms(mu, s):
    """Expected log-likelihood and KL(q || prior), eq. (20.3) and (20.4)."""
    expected_loglik = -0.5 * math.log(2 * math.pi) - 0.5 * ((x - mu) ** 2 + s ** 2)
    kl = 0.5 * (s ** 2 + mu ** 2 - 1 - math.log(s ** 2))
    return expected_loglik, kl


def elbo_by_sampling(mu, s, n=200_000, seed=0):
    """Reparameterisation: z = mu + s * eps with eps ~ N(0, 1), eq. (20.5)."""
    eps = np.random.default_rng(seed).standard_normal(n)
    z = mu + s * eps
    log_lik = -0.5 * math.log(2 * math.pi) - 0.5 * (x - z) ** 2
    log_prior = -0.5 * math.log(2 * math.pi) - 0.5 * z ** 2
    log_q = -0.5 * math.log(2 * math.pi * s ** 2) - 0.5 * eps ** 2
    return float(np.mean(log_lik + log_prior - log_q))


log_evidence = -0.5 * math.log(2 * math.pi * 2) - x ** 2 / 4
for mu, s in ((1.0, 0.8), (1.0, math.sqrt(0.5))):
    loglik, kl = elbo_terms(mu, s)
    print(f'q = N({mu}, {s ** 2:.2f}): expected log-likelihood {loglik:.4f}, KL {kl:.4f}, '
          f'ELBO {loglik - kl:.4f}, sampled {elbo_by_sampling(mu, s):.4f}')
print(f'log p(x) = {log_evidence:.4f}')
