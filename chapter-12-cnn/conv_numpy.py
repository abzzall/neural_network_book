"""Chapter 12: two-dimensional convolution from scratch, checked against PyTorch."""

import numpy as np
import torch


def conv2d(image, kernel, padding=0, stride=1):
    """Slide the kernel over the image and take a dot product at every position.

    (Deep-learning libraries call this convolution; mathematically it is
    cross-correlation, because the kernel is not flipped.)
    """
    image = np.pad(image, padding)
    k = kernel.shape[0]
    out = (image.shape[0] - k) // stride + 1
    result = np.zeros((out, out))
    for i in range(out):
        for j in range(out):
            patch = image[i * stride:i * stride + k, j * stride:j * stride + k]
            result[i, j] = np.sum(patch * kernel)
    return result


def kernel_gradient(image, upstream):
    """dL/dK: the image convolved with the gradient arriving at the output (stride 1, no padding)."""
    return conv2d(image, upstream)


image = np.array([[0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1]], dtype=float)          # a vertical edge
kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=float)               # responds to left-to-right increases

if __name__ == "__main__":
    out = conv2d(image, kernel)
    print("output (3x3):", out.tolist())
    print("with padding 1:", conv2d(image, kernel, padding=1).shape,
          " with stride 2:", conv2d(image, kernel, stride=2).shape)
    theirs = torch.nn.functional.conv2d(torch.tensor(image)[None, None], torch.tensor(kernel)[None, None])
    print("matches torch.nn.functional.conv2d:", bool(np.allclose(out, theirs[0, 0].numpy())))
