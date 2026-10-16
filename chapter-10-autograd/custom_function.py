"""A custom smooth operation, checked numerically in double precision."""

import torch


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        s = torch.sigmoid(x)
        return grad_output * (s + x * s * (1 - s))


if __name__ == '__main__':
    x = torch.tensor([-2., -0.5, 0., 1., 3.], dtype=torch.double, requires_grad=True)
    print(f'gradcheck: {torch.autograd.gradcheck(Swish.apply, (x,))}')
    Swish.apply(x).sum().backward()
    reference = x.detach().clone().requires_grad_()
    (reference * torch.sigmoid(reference)).sum().backward()
    print(f'max gradient difference: {(x.grad - reference.grad).abs().max().item():.2e}')
