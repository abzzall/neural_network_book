# Chapter 9 — From NumPy to PyTorch

Executable material for chapter 9 of the book. Run every command from the
repository root. PyTorch is installed in this chapter; see the repository
README for the CPU-only installation used to verify the code.

| File | Book | What it shows |
|---|---|---|
| `tensors_demo.py` | — | Tensors: shapes, dtypes, broadcasting, and sharing memory with NumPy |
| `autograd_demo.py` | Worked example | Autograd reproduces the gradients computed by hand in chapters 2 and 6 |
| `numpy_vs_torch.py` | — | The same network in nnbook and PyTorch gives the same loss and gradients |
| `train_loop.py` | Listing 9.1 | A complete training loop with DataLoader, evaluation, saving and reloading |
| `test_scripts.py` | — | Runs every script and checks what it prints |
| `applied/` | Applied analysis | The dropout network of chapter 8 in PyTorch, across five seeds |

```bash
.venv/bin/python chapter-09-pytorch/autograd_demo.py
.venv/bin/python -m pytest chapter-09-pytorch
```

Expected output of `autograd_demo.py`:

```text
chapter 2: dL/dw = -4.0, dL/db = -4.0
chapter 6: loss = 0.8068, dW1 = [[-0.1343, -0.0671], [0.1841, 0.0921]]
```

PyTorch does not promise bit-identical training results on different machines,
so the last digits of trained accuracies may differ from those in the book.

## Applied exercise

Start from `applied/INSTRUCTIONS.md`. The starting point is `applied/based-on/`
(its tests fail until you complete it); the finished code is
`applied/full-code/`.
