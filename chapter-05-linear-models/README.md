# Chapter 5 — Linear Models and the Perceptron

Executable material for chapter 5 of the book. Run every command from the
repository root.

| File | Book | What it shows |
|---|---|---|
| `perceptron_demo.py` | Listing 5.2 | Trains the perceptron on logical AND and on two separable point clouds |
| `perceptron_trace.py` | Worked example | Every update of the perceptron rule on AND |
| `xor_demo.py` | — | No linear model separates XOR; the feature x1*x2 does |
| `softmax_digits.py` | Listing 5.4 | Softmax regression from scratch on handwritten digits |
| `test_perceptron_demo.py`, `test_scripts.py` | — | Run every script and check every printed line |
| `applied/` | Applied analysis | Logistic regression from scratch on tumour measurements |

The `Perceptron` class (listing 5.1) is in [`nnbook/perceptron.py`](../nnbook/perceptron.py);
logistic and softmax regression (listing 5.3) are in [`nnbook/linear.py`](../nnbook/linear.py),
tested against scikit-learn in [`tests/test_linear.py`](../tests/test_linear.py).

Data: the digits are the UCI Optical Recognition of Handwritten Digits test set
(Alpaydin and Kaynak, 1998, https://doi.org/10.24432/C50P49, CC BY 4.0); the
tumour measurements are the UCI Breast Cancer Wisconsin (Diagnostic) data
(Wolberg, Mangasarian, Street and Street, 1993, https://doi.org/10.24432/C5DW2B,
CC BY 4.0). Both are distributed with scikit-learn; nothing is downloaded.

```bash
.venv/bin/python chapter-05-linear-models/perceptron_demo.py
.venv/bin/python -m pytest chapter-05-linear-models
```

Expected output of `perceptron_demo.py`:

```text
AND: w = [3. 2.] b = -4.0
AND: predictions = [-1 -1 -1  1]
AND: errors per epoch = [2, 3, 3, 2, 2, 3, 2, 1, 0]
Clouds: accuracy = 1.00, epochs = 2
```

## Applied exercise

Start from `applied/INSTRUCTIONS.md`. The starting point is `applied/based-on/`
(its tests fail until you complete it); the finished code is
`applied/full-code/`.
