"""One float16-autocast training step with float32 parameters."""

import torch


def run():
    torch.manual_seed(42)
    model = torch.nn.Linear(8, 4).cuda()
    x = torch.randn(32, 8, device='cuda')
    target = torch.randn(32, 4, device='cuda')
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler('cuda')
    before = model.weight.detach().clone()
    optimiser.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, target)
    scaler.scale(loss).backward()
    scaler.step(optimiser)
    scaler.update()
    changed = not torch.equal(before, model.weight)
    finite = all(torch.isfinite(p.grad).all().item()
                 for p in model.parameters())
    return str(prediction.dtype), str(loss.dtype), changed, finite


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('SKIP: CUDA unavailable')
    else:
        prediction, loss, changed, finite = run()
        print(f'prediction dtype: {prediction}')
        print(f'loss dtype: {loss}')
        print(f'parameters updated: {changed}')
        print(f'finite unscaled gradients: {finite}')
