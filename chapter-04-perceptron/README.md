# Chapter 4 — The Perceptron

Executable material for chapter 4 of the book.

| File | Book | What it shows |
|---|---|---|
| `perceptron_demo.py` | Listing 4.2 | Trains the perceptron on logical AND and on two separable point clouds |
| `test_perceptron_demo.py` | — | Runs the demo and checks every printed result |

The `Perceptron` class itself (listing 4.1) is in [`nnbook/perceptron.py`](../nnbook/perceptron.py),
tested by [`tests/test_perceptron.py`](../tests/test_perceptron.py).

Run from the repository root:

```bash
python chapter-04-perceptron/perceptron_demo.py
python -m pytest chapter-04-perceptron
```

Expected output of the demo:

```text
AND: w = [3. 2.] b = -4.0
AND: predictions = [-1 -1 -1  1]
AND: errors per epoch = [2, 3, 3, 2, 2, 3, 2, 1, 0]
Clouds: accuracy = 1.00, epochs = 2
```
