"""Low-rank adaptation of one linear layer, in NumPy: counts, start, and merging."""

import numpy as np


def lora_forward(x, W, A, B, alpha):
    """y = x W^T + (alpha / r) x A^T B^T: frozen W plus a rank-r update, eq. (22.2)."""
    r = A.shape[0]
    return x @ W.T + (alpha / r) * (x @ A.T) @ B.T


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    d_out, d_in, r, alpha = 384, 384, 8, 16
    W = rng.normal(0, 0.05, (d_out, d_in))               # pretrained, frozen
    A = rng.normal(0, 0.01, (r, d_in))                   # trainable, small random
    B = np.zeros((d_out, r))                             # trainable, starts at zero
    print(f'full fine-tuning: {W.size} trainable weights in this layer')
    print(f'LoRA rank {r}: {A.size + B.size} trainable weights '
          f'({100 * (A.size + B.size) / W.size:.2f}% of full)')
    x = rng.normal(size=(5, d_in))
    print('output unchanged at the start:', np.allclose(lora_forward(x, W, A, B, alpha), x @ W.T))
    B = rng.normal(0, 0.01, (d_out, r))                  # pretend training changed B
    merged = W + (alpha / r) * B @ A                     # fold the update into W
    print('merged layer gives the same output:',
          np.allclose(lora_forward(x, W, A, B, alpha), x @ merged.T))
    print('rank of the update:', np.linalg.matrix_rank(merged - W))
