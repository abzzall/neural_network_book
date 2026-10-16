"""Eight Gaussian clusters on a circle: a 2-D dataset with clearly separated modes."""

import numpy as np

CENTRES = 2.0 * np.array([[np.cos(a), np.sin(a)] for a in np.arange(8) * np.pi / 4])
SPREAD = 0.05


def sample_ring(n, rng):
    which = rng.integers(8, size=n)
    return CENTRES[which] + SPREAD * rng.standard_normal((n, 2))


def mode_report(points, radius=0.2, min_share=0.01):
    """Modes covered (at least 1% of samples within `radius` of a centre) and the
    share of samples near any centre."""
    d = np.linalg.norm(points[:, None, :] - CENTRES[None], axis=2)
    near = d.min(axis=1) < radius
    shares = np.bincount(d.argmin(axis=1)[near], minlength=8) / len(points)
    return int((shares >= min_share).sum()), float(near.mean())
