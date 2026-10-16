# Chapter 2 — Linear Algebra and Calculus

Executable material for chapter 2 of the book. Run every command from the
repository root.

| File | Book | What it shows |
|---|---|---|
| `layer.py` | Listing 2.1 | A layer of two neurons as a matrix–vector product, for one sample and for a batch |
| `chain_rule.py` | Listing 2.2 | Forward and backward values on the graph of one squared error |
| `numerical_gradient.py` | Listing 2.3 | A central-difference gradient checked against the formula for a quadratic form |
| `two_steps.py` | Worked example | Two gradient-descent steps for a line through three points, and the exact solution |
| `gd_paths.py` | Figure data | Gradient descent on an elongated bowl at three learning rates |
| `test_scripts.py` | — | Runs every script and checks every printed line |
| `applied/` | Applied analysis | Linear regression by gradient descent and by the normal equations |

```bash
.venv/bin/python chapter-02-linear-algebra/two_steps.py
.venv/bin/python -m pytest chapter-02-linear-algebra
```

Expected output of `two_steps.py`:

```text
step 0: theta = [0. 0.], loss = 15.0000
step 1: gradient = [ -7.3333 -16.6667], theta = [0.7333 1.6667], loss = 0.2341
step 2: gradient = [0.8    1.8222], theta = [0.6533 1.4844], loss = 0.0577
normal equations: theta = [0.6667 1.5   ], loss = 0.0556
learning-rate bound = 0.1803
```

`gd_paths.py OUTPUT_DIR` writes the paths drawn in the book's figure.

## Applied exercise

Start from `applied/INSTRUCTIONS.md`. The starting point is `applied/based-on/`
(its tests fail until you complete it); the finished code is
`applied/full-code/`.
