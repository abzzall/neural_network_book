"""Two test images, their true masks, and the masks of the U-Net and of the threshold."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_shapes import train_model  # noqa: E402
from segmentation_metrics import choose_threshold, class_dice  # noqa: E402
from vision_shapes import make_shapes  # noqa: E402

model = train_model(seed=42)                        # the first seed of the exercise
val_x, val_y = make_shapes(32, 11)
test_x, test_y = make_shapes(64, 12)
with torch.inference_mode():
    threshold = choose_threshold(model(val_x).sigmoid(), val_y)
    baseline_threshold = choose_threshold(val_x, val_y)
    network = model(test_x).sigmoid() >= threshold
baseline = test_x >= baseline_threshold


def save(folder, name, image):
    grid = image.float().numpy()
    rows = [(c, r, grid[r, c]) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
    np.savetxt(Path(folder) / f'{name}.dat', np.array(rows), fmt='%.4g',
               header='x y v', comments='')


if __name__ == '__main__':
    print(f'thresholds: network {threshold}, intensity {baseline_threshold}')
    for k in (0, 1):
        net = class_dice(test_y[k], network[k])[1]
        base = class_dice(test_y[k], baseline[k])[1]
        print(f'test image {k}: foreground Dice network {net:.3f}, threshold {base:.3f}')
        if len(sys.argv) > 1:
            for name, image in (('image', test_x[k, 0]), ('truth', test_y[k, 0]),
                                ('network', network[k, 0]), ('threshold', baseline[k, 0])):
                save(sys.argv[1], f'{name}{k}', image)
