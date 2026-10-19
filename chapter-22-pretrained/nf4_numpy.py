"""Chapter 22: the 4-bit NormalFloat (NF4) code of QLoRA, against a uniform 4-bit code.

Weights are stored in blocks of 64, each scaled by its largest absolute value, and
every scaled weight is replaced by the nearest of 16 levels. NF4 places the levels
at quantiles of the normal distribution, where trained weights are dense; the
uniform code spaces them evenly.
"""

import numpy as np
from scipy.stats import norm

BLOCK = 64


def nf4_levels(offset=0.9677083):
    """8 positive and 7 negative normal quantiles, plus an exact zero, scaled to [-1, 1]."""
    positive = norm.ppf(np.linspace(offset, 0.5, 9)[:-1])
    negative = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])
    levels = np.sort(np.r_[positive, 0.0, negative])
    return levels / np.abs(levels).max()


def quantise(w, levels, block_size=BLOCK):
    """Return blockwise codes, scales and the original size.

    The last block is padded with zeros. Codes are returned unpacked as uint8 for
    clarity; a real 4-bit implementation packs two codes into each byte.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.size == 0:
        raise ValueError('cannot quantise an empty array')
    if block_size <= 0:
        raise ValueError('block_size must be positive')
    levels = np.asarray(levels, dtype=np.float32)
    if levels.ndim != 1 or len(levels) != 16 or not np.all(np.diff(levels) > 0):
        raise ValueError('levels must contain 16 increasing values')
    size = w.size
    padded_size = ((size + block_size - 1) // block_size) * block_size
    flat = np.pad(w.reshape(-1), (0, padded_size - size))
    blocks = flat.reshape(-1, block_size)
    scale = np.abs(blocks).max(axis=1, keepdims=True)
    safe_scale = np.where(scale == 0, 1, scale)
    codes = np.abs(blocks[..., None] / safe_scale[..., None] - levels).argmin(axis=-1)
    return codes.astype(np.uint8), scale.astype(np.float32), size


def dequantise(codes, scale, levels, shape, size=None):
    """Dequantise unpacked codes and remove any padding."""
    size = int(np.prod(shape)) if size is None else size
    flat = (np.asarray(levels)[codes] * scale).reshape(-1)[:size]
    return flat.reshape(shape).astype(np.float32)


NF4 = nf4_levels()
UNIFORM = np.linspace(-1, 1, 16)

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.02, (1024, 1024)).astype(np.float32)
    X = rng.normal(0, 1, (256, 1024)).astype(np.float32)
    print('NF4 levels:', ' '.join(f'{v:+.4f}' for v in NF4))
    exact = X @ W.T
    for name, levels in (('uniform 4-bit', UNIFORM), ('NF4', NF4)):
        codes, scale, size = quantise(W, levels)
        Wq = dequantise(codes, scale, levels, W.shape, size)
        weight_error = np.sqrt(np.mean((Wq - W) ** 2)) / W.std()
        output_error = np.sqrt(np.mean((X @ Wq.T - exact) ** 2)) / exact.std()
        print(f'{name:14s} relative error: weights {weight_error:.4f}, outputs {output_error:.4f}')
    bits = 4 + 32 / BLOCK
    print(f'theoretical packed storage: {bits:.2f} bits per weight against 16 for bfloat16, '
          f'{1024 * 1024 * bits / 8 / 2**20:.3f} MiB against {1024 * 1024 * 2 / 2**20:.3f} MiB')
