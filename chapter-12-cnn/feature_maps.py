"""Chapter 12: what the first convolutional layer of the trained digit CNN computes."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "applied" / "full-code"))
from cnn_vs_mlp import cnn  # noqa: E402
from digits_data import accuracy, fit, load_images  # noqa: E402

X_train, y_train, X_test, y_test = load_images()
model = fit(cnn, X_train, y_train, epochs=30, seed=0)       # seed 0 of the exercise
image = X_test[:1]                                          # the first test digit
with torch.inference_mode():
    maps = model[1](model[0](image))[0]                     # 16 maps of 8 x 8, after ReLU
    predicted = model(image).argmax(dim=1).item()
kernels = model[0].weight.detach()[:, 0]                    # 16 kernels of 3 x 3


def save(folder, name, grid):
    rows = [(col, row, grid[row, col]) for row in range(grid.shape[0])
            for col in range(grid.shape[1])]
    np.savetxt(Path(folder) / f"{name}.dat", np.array(rows), fmt="%.6g",
               header="x y v", comments="")


if __name__ == "__main__":
    print(f"test accuracy = {accuracy(model, X_test, y_test):.4f}")
    print(f"first test digit: label {y_test[0].item()}, predicted {predicted}")
    strongest = maps.sum(dim=(1, 2)).argsort(descending=True)[:4].tolist()
    print("feature maps with the largest total response:", strongest)
    print(f"maps that are zero everywhere: {int((maps.amax(dim=(1, 2)) == 0).sum())} of 16")
    if len(sys.argv) > 1:
        save(sys.argv[1], "digit", image[0, 0].numpy())
        for k in strongest:
            save(sys.argv[1], f"map{k}", maps[k].numpy())
            save(sys.argv[1], f"kernel{k}", kernels[k].numpy())
