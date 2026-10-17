"""Affine int8 quantisation: scale, zero point, rounding error, and a whole layer."""

import numpy as np


def quantise(x, qmin=-128, qmax=127):
    """q = clamp(round(x / s) + z), with s and z mapping [min, max] onto [qmin, qmax], eq. (24.1)."""
    lo, hi = min(float(np.min(x)), 0.0), max(float(np.max(x)), 0.0)  # keep 0 exact
    scale = (hi - lo) / (qmax - qmin)
    zero_point = int(round(qmin - lo / scale))
    q = np.clip(np.round(np.asarray(x) / scale) + zero_point, qmin, qmax).astype(np.int8)
    return q, scale, zero_point


def dequantise(q, scale, zero_point):
    return (q.astype(np.float32) - zero_point) * scale


if __name__ == '__main__':
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    q, s, z = quantise(x)
    print(f'scale {s:.6f}, zero point {z}')
    print('int8 codes:', q.tolist())
    print('dequantised:', ' '.join(f'{v:.4f}' for v in dequantise(q, s, z)))
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.1, (256, 256)).astype(np.float32)        # a layer's weights
    qw, s, z = quantise(w)
    err = np.abs(dequantise(qw, s, z) - w)
    print(f'layer: {w.nbytes} bytes -> {qw.nbytes} bytes, '
          f'max error {err.max():.5f} (half a step: {s / 2:.5f})')
