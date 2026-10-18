# Chapter 7 — Optimising the Training

Executable material for chapter 7 of the book. Run every command from the
repository root.

| File | Book | What it shows |
|---|---|---|
| `optimizer_paths.py` | Figure data | Gradient descent, momentum, RMSProp and Adam on a narrow valley, each at its best learning rate |
| `adam_by_hand.py` | Worked example | Three steps of Adam with every quantity shown |
| `init_demo.py` | Figure data | How the scale of random weights changes the signal through ten layers |
| `debug_checks.py` | Debugging checklist | The initial loss, memorising a tiny batch, and a learning rate far too large |
| `test_scripts.py` | — | Runs every script and checks every printed line |
| `applied/` | Applied analysis | The digits network trained with four optimisers |

The optimisers and learning-rate schedules are in
[`nnbook/optimizers.py`](../nnbook/optimizers.py) (listing 7.1), tested in
[`tests/test_optimizers.py`](../tests/test_optimizers.py); the initialisations are
options of `Dense` in [`nnbook/layers.py`](../nnbook/layers.py).

```bash
.venv/bin/python chapter-07-optimization/adam_by_hand.py
.venv/bin/python -m pytest chapter-07-optimization
```

Expected output of `adam_by_hand.py`:

```text
step 1: g = 2.0000, m = 0.2000, v = 0.004000, m_hat = 2.0000, v_hat = 4.0000, t = 0.9000
step 2: g = 1.8000, m = 0.3600, v = 0.007236, m_hat = 1.8947, v_hat = 3.6198, t = 0.8004
step 3: g = 1.6008, m = 0.4841, v = 0.009791, m_hat = 1.7863, v_hat = 3.2671, t = 0.7016
```

## Applied exercise

Start from `applied/INSTRUCTIONS.md`. The starting point is `applied/based-on/`
(its tests fail until you complete it); the finished code is
`applied/full-code/`.
