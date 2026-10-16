"""Compare full-precision and mixed-precision fitting on synthetic data."""

from contextlib import nullcontext
import json

import torch


def data():
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(128, 4, generator=generator)
    y = x @ torch.tensor([[1.], [-2.], [0.5], [3.]]) + 0.25
    return x, y


def move_batch(x, y, device):
    raise NotImplementedError('Step 1: move both inputs and targets')


def forward_loss(model, x, y, amp):
    raise NotImplementedError('Step 2: select the autocast context')
    with context:
        raise NotImplementedError('Step 3: compute prediction and mean squared error')


def clear_gradients(optimiser):
    raise NotImplementedError('Step 4: clear gradients')


def differentiate(loss, scaler):
    raise NotImplementedError('Step 5: differentiate the scaled loss')


def update(optimiser, scaler):
    raise NotImplementedError('Step 6: step the optimiser and update the scale')


def fit(device='cpu', amp=False, steps=80):
    if amp and torch.device(device).type != 'cuda':
        raise ValueError('float16 exercise requires CUDA')
    x, y = move_batch(*data(), device)
    model = torch.nn.Linear(4, 1).to(device)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=128.)
    initial = forward_loss(model, x, y, False).item()
    first_gradients = None
    for step in range(steps):
        clear_gradients(optimiser)
        loss = forward_loss(model, x, y, amp)
        differentiate(loss, scaler)
        if step == 0:
            # Inspect gradients only after undoing the scale; step will not
            # unscale a second time when unscale_ was called explicitly.
            scaler.unscale_(optimiser)
            first_gradients = torch.cat([p.grad.flatten() for p in model.parameters()])
            first_gradients = first_gradients.detach().cpu().clone()
        update(optimiser, scaler)
    with torch.inference_mode():
        # Evaluate every fitted model in float32 for a common comparison.
        final = forward_loss(model, x, y, False).item()
    return initial, final, first_gradients


def run():
    initial, final, cpu_grad = fit()
    results = {'cpu_initial_mse': initial, 'cpu_final_mse': final}
    if torch.cuda.is_available():
        for mode, amp in [('cuda_fp32', False), ('cuda_amp', True)]:
            _, final, grad = fit('cuda', amp)
            results[f'{mode}_final_mse'] = final
            results[f'{mode}_gradient_error'] = (grad - cpu_grad).abs().max().item()
    return results


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(json.dumps(run(), indent=2))
