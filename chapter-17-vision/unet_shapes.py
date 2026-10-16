"""Inspect feature-map shapes without training or external data."""

import torch

from tiny_unet import TinyUNet
from vision_shapes import make_shapes


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = TinyUNet().eval()
    for name in ('encoder1', 'encoder2', 'bottom', 'decoder2', 'decoder1', 'output'):
        getattr(model, name).register_forward_hook(
            lambda module, inputs, output, label=name: print(f'{label}: {list(output.shape)}'))
    images, masks = make_shapes(2, seed=10)
    with torch.inference_mode():
        result = model(images)
    print(f'output matches target shape: {result.shape == masks.shape}')
    print(f'parameters: {sum(p.numel() for p in model.parameters())}')
