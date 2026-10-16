"""An exact scalar memory-path experiment at the zero tanh trajectory."""

import torch


def derivative(weight, steps):
    h0 = torch.tensor(0., dtype=torch.double, requires_grad=True)
    h = h0
    for _ in range(steps):
        h = torch.tanh(weight * h)
    return torch.autograd.grad(h, h0)[0].item()


if __name__ == '__main__':
    for weight in (0.5, 1., 1.5):
        print(f'weight {weight:.1f}, 20-step state derivative: {derivative(weight, 20):.6g}')
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3., 4.])
    before = torch.nn.utils.clip_grad_norm_([p], max_norm=1.)
    print(f'gradient norm before clipping: {before.item():.1f}')
    print(f'gradient norm after clipping: {p.grad.norm().item():.6f}')
