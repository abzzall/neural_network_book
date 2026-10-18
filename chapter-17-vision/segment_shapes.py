"""Train a tiny U-Net; freeze thresholds on validation data before testing."""

import json
import numpy as np
import torch

from segmentation_metrics import class_dice, choose_threshold, segmentation_loss
from tiny_unet import TinyUNet
from vision_shapes import flip_pair, make_shapes


def train_model(seed=42, epochs=40, loss_fn=segmentation_loss):
    """Train on the fixed synthetic training images; return the model in evaluation mode."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    train_x, train_y = make_shapes(128, 10)
    model = TinyUNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=.003)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for ids in torch.randperm(len(train_x), generator=generator).split(16):
            flips = torch.rand(2, generator=generator) < .5
            images, masks = flip_pair(train_x[ids], train_y[ids], *flips.tolist())
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimiser.step()
    return model.eval()


def run(seed=42, epochs=40, loss_fn=segmentation_loss,
        threshold_fn=choose_threshold, score_fn=class_dice):
    model = train_model(seed, epochs, loss_fn)
    val_x, val_y = make_shapes(32, 11)
    test_x, test_y = make_shapes(64, 12)
    with torch.inference_mode():
        threshold = threshold_fn(model(val_x).sigmoid(), val_y)
        baseline_threshold = threshold_fn(val_x, val_y)
        scores = score_fn(test_y, model(test_x).sigmoid() >= threshold)
        baseline = score_fn(test_y, test_x >= baseline_threshold)
    return dict(seed=seed, threshold=threshold, dice=scores,
                baseline_threshold=baseline_threshold, baseline_dice=baseline)


if __name__ == '__main__':
    runs = [run(seed) for seed in (42, 43, 44)]
    scores = np.array([item['dice'] for item in runs])
    print(json.dumps(dict(runs=runs, mean=scores.mean(0).tolist(),
                          std=scores.std(0).tolist()), indent=2))
