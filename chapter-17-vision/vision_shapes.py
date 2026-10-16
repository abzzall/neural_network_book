"""Original binary shape masks and noisy images with varying illumination."""

import numpy as np
import torch


def make_shapes(count, seed, size=32):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    images, masks = [], []
    for _ in range(count):
        cx, cy = rng.integers(9, size - 9, size=2)
        radius = rng.integers(4, 8)
        if rng.random() < .5:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        else:
            mask = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius - 1)
        background = rng.uniform(.05, .3)
        contrast = rng.uniform(.25, .5)
        gradient = rng.uniform(-.2, .2) * (xx / (size - 1) - .5)
        image = background + contrast * mask + gradient + rng.normal(0, .12, mask.shape)
        images.append(np.clip(image, 0, 1))
        masks.append(mask)
    return (torch.tensor(np.stack(images), dtype=torch.float32)[:, None],
            torch.tensor(np.stack(masks), dtype=torch.float32)[:, None])


def flip_pair(images, masks, horizontal=False, vertical=False):
    dimensions = ([-1] if horizontal else []) + ([-2] if vertical else [])
    return ((images.flip(dimensions), masks.flip(dimensions)) if dimensions
            else (images, masks))
