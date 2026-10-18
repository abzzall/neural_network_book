# Chapter 12 — Convolutional Neural Networks

Executable material for chapter 12 of the book. Run every command from the
repository root.

| File | Book | What it shows |
|---|---|---|
| `conv_numpy.py` | Listing 12.1 | Convolution from scratch, its kernel gradient, and a check against PyTorch |
| `residual.py` | Listing 12.2 | A residual block, and that it passes its input through when F(x) = 0 |
| `digits_data.py` | — | The digits as 1 x 8 x 8 images, with training and evaluation helpers |
| `transfer_demo.py` | — | Reusing features learned on digits 0-4 for digits 5-9: an honest negative result |
| `feature_maps.py` | Figure data | The learned first-layer kernels and their feature maps for one digit |
| `kazakh_keywords.py` | Section 12.6, figure 12.5 | Log-mel spectrograms of spoken Kazakh commands and a CNN; robustness to noise (download first) |
| `test_scripts.py` | — | Runs every script and checks what it prints |
| `applied/` | Applied analysis | A CNN against an MLP on original and shifted digits, with augmentation |

Data: UCI Optical Recognition of Handwritten Digits (Alpaydin and Kaynak, 1998,
https://doi.org/10.24432/C50P49, CC BY 4.0), distributed with scikit-learn.

The spoken commands are the Kazakh Speech Commands Dataset (ISSAI, Kuzdeuov
et al. 2023, MIT licence), pinned to revision `8efc3b8` and downloaded only on
request (75 MB) with `kazakh_keywords.py --download`; its test is skipped until
then. Training takes a few minutes on a CPU.

```bash
.venv/bin/python chapter-12-cnn/conv_numpy.py
.venv/bin/python -m pytest chapter-12-cnn
```

PyTorch does not promise bit-identical training results on different machines,
so the last digits of trained accuracies may differ from those in the book.

## Applied exercise

Start from `applied/INSTRUCTIONS.md`. The starting point is `applied/based-on/`
(its tests fail until you complete it); the finished code is
`applied/full-code/`.
