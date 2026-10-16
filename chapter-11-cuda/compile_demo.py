"""Check a compiled pointwise function and its gradient against eager mode."""

import torch


def transform(x):
    return torch.relu(x * 2 + 1)


def run():
    x = torch.linspace(-2, 2, 257, device='cuda', requires_grad=True)
    compiled = torch.compile(transform, fullgraph=True)
    actual = compiled(x)  # The first call includes compilation.
    expected = transform(x)
    torch.testing.assert_close(actual, expected)
    grad, = torch.autograd.grad(actual.sum(), x)
    reference, = torch.autograd.grad(expected.sum(), x)
    torch.testing.assert_close(grad, reference)
    return actual.shape[0]


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('SKIP: CUDA unavailable')
    else:
        print(f'compiled values and gradients checked: {run()}')
