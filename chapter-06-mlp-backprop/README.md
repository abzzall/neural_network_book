# Chapter 6 — Multilayer Networks and Backpropagation

Executable material for chapter 6 of the book. Run every command from the
repository root.

| File | Book | What it shows |
|---|---|---|
| `backprop_by_hand.py` | Worked example | One forward and one backward pass through a 2-2-1 network |
| `xor_mlp.py` | Figure data | Two hidden units learn XOR; the hidden features make it separable |
| `approximation_demo.py` | — | Wider hidden layers can approximate sin(2x) more closely |
| `test_scripts.py` | — | Runs every script and checks what it prints |
| `applied/` | Applied analysis | A multilayer network from scratch on handwritten digits |

The network itself is in `nnbook`: [`layers.py`](../nnbook/layers.py) (listing 6.1),
[`activations.py`](../nnbook/activations.py), [`losses.py`](../nnbook/losses.py),
[`network.py`](../nnbook/network.py) (listing 6.2) and
[`gradcheck.py`](../nnbook/gradcheck.py), tested in
[`tests/test_network.py`](../tests/test_network.py).

```bash
.venv/bin/python chapter-06-mlp-backprop/backprop_by_hand.py
.venv/bin/python -m pytest chapter-06-mlp-backprop
```

Expected output of `backprop_by_hand.py`:

```text
z1 = [0.35 0.7 ]  a1 = [0.5866 0.6682]
z2 = -0.2157, p = 0.4463, loss = 0.8068
delta2 = -0.5537, dw2 = [-0.3248 -0.37  ] , db2 = -0.5537
delta1 = [-0.1343  0.1841]
dW1 = [[-0.1343, -0.0671], [0.1841, 0.0921]]
```

## Applied exercise

Start from `applied/INSTRUCTIONS.md`. The starting point is `applied/based-on/`
(its tests fail until you complete it); the finished code is
`applied/full-code/`.
