"""A forward-only, contiguous float32 vector kernel with a masked tail."""

import torch


def add(x, y):
    import triton
    import triton.language as tl

    if (x.device.type != 'cuda' or x.device != y.device
            or x.dtype != torch.float32 or y.dtype != torch.float32
            or x.ndim != 1 or x.shape != y.shape
            or not x.is_contiguous() or not y.is_contiguous()):
        raise ValueError('expected same-device contiguous CUDA float32 vectors')
    if x.requires_grad or y.requires_grad:
        raise ValueError('forward-only kernel; inputs must not require gradients')

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        a = tl.load(X + offsets, mask=mask, other=0)
        b = tl.load(Y + offsets, mask=mask, other=0)
        tl.store(Z + offsets, a + b, mask=mask)

    out = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), 256)
    return out


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('SKIP: CUDA unavailable')
    else:
        torch.manual_seed(42)
        x, y = (torch.randn(1003, device='cuda') for _ in range(2))
        result = add(x, y)
        torch.testing.assert_close(result, x + y, rtol=0, atol=0)
        print(f'Triton vector addition checked: {result.numel()} elements')
