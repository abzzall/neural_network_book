# Chapter 10 — Automatic Differentiation from Scratch

The executable examples below have been run and checked on Python 3.14.4,
NumPy 2.5.4 and PyTorch 2.14.1, on a CPU (16 October 2026).
The guided XOR exercise includes a starting point, acceptance tests, a finished
solution and a results verifier in `applied/`. See `applied/INSTRUCTIONS.md`.

Run commands from the repository root, using the environment described in
chapter 9 and the repository README.

| File | What is verified |
|---|---|
| `value.py` | Scalar operation derivatives, shared inputs and accumulation, compared with PyTorch |
| `value_demo.py` | Gradients from chapter 2, shared inputs, and repeated backward calls |
| `tiny_tensor.py` | Matrix products, broadcasting, hooks, graph retention and error cases |
| `rules_demo.py` | Leaves, retained gradients, removable hooks, accumulation, no-grad and detached mutation |
| `custom_function.py` | Custom Swish forward and backward against native PyTorch and finite differences |
| `applied/full-code/autograd_xor.py` | XOR training, all first-step parameter gradients against PyTorch, and predictions |

```bash
.venv/bin/python chapter-10-autograd/value_demo.py
.venv/bin/python chapter-10-autograd/rules_demo.py
.venv/bin/python chapter-10-autograd/custom_function.py
.venv/bin/python chapter-10-autograd/applied/full-code/autograd_xor.py
.venv/bin/python -m pytest chapter-10-autograd
```

Expected output of `value_demo.py`:

```text
chapter 2: dL/dw = -4.0, dL/db = -4.0
shared input: dz/dx = -0.007436 (PyTorch -0.007436), dz/dw1 = 0.004958 (PyTorch 0.004958)
after two backward calls: dL/dw = -8.0
```

Expected output of the XOR example on the verified environment:

```text
initial MSE: 0.329236
final MSE: 0.000420
first gradient difference: 4.16e-17
probabilities: 0.0068, 0.9892, 0.9896, 0.0375
```

Last digits can differ between platforms. Tests check numerical tolerances,
classification of all four XOR inputs, consistency of the printed loss with
the predictions, and the gradient comparison rather than requiring an exact
floating-point residual.

## Teaching-engine boundaries

`Value` is a scalar, first-order teaching engine. It retains its graph for
repeated backward calls and does not protect against changing values between
forward and backward. Update weights only after backward and build a fresh
graph for each training step.

`tiny_tensor` supports real float64 arrays, elementwise addition and
multiplication, two-dimensional matrix products, ReLU, and full summation.
It demonstrates selected PyTorch rules, with shorter error messages. It does
not implement higher derivatives, general view or in-place history, concurrent
grad modes, inference mode, or the complete PyTorch API. Use `add_` for its
supported mutations; writing directly to `.data` bypasses version checks.

`test_scripts.py` runs every demonstration in a subprocess from a temporary
working directory and checks its output. `test_engines.py` checks derivatives
and error behaviour. Both are included in `./scripts/check-all.sh`.

## Guided exercise

Start with `applied/based-on/autograd_xor.py`. Its incomplete functions raise
`NotImplementedError` until you complete the numbered steps. Tests are supplied
in both starting and finished folders; only the finished solution is included
in the repository-wide checks. Record your values in `applied/RESULTS.md`.

```bash
.venv/bin/python chapter-10-autograd/applied/full-code/verify_results.py
.venv/bin/python chapter-10-autograd/applied/full-code/verify_results.py chapter-10-autograd/applied/RESULTS.md
```

The first command prints the reference results. The second checks your filled
table; an empty template is expected to fail.
