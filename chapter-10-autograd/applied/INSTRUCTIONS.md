# Chapter 10 applied exercise — train with scalar autograd

## 1. Outcome

A 2-4-1 ReLU network trained on all four XOR inputs using scalar automatic
differentiation. Before training, compare all 17 parameter gradients with
PyTorch. The four points are generated in the file; no data download is needed.
This is a check of implementation and optimisation, not a generalisation study.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, PyTorch 2.14.1 and pytest 9.1.1, on a CPU,
16 October 2026. Use the existing environment from chapter 9.

## 3. Files supplied

- `based-on/autograd_xor.py`: incomplete functions and training-loop lines, numbered below.
- `based-on/test_autograd_xor.py`: complete acceptance tests.
- `full-code/`: finished code, tests and a results verifier.
- `RESULTS.md`: the results table and explanation questions.

The parameter initialisation, PyTorch reference, imports and printing are
supplied. The scalar `Value` engine lives two levels above the exercise.

## 4. Steps

Edit `based-on/autograd_xor.py`. Run the tests from the repository root after
each function is completed:

```bash
.venv/bin/python -m pytest chapter-10-autograd/applied/based-on
```

1. In `hidden_values`, compute each of four affine functions and apply ReLU.
   Hidden unit j uses parameters `p[3*j]`, `p[3*j+1]`, `p[3*j+2]`.
2. In `predict`, weight the four hidden values with `p[12:16]`, add `p[16]`,
   and apply the sigmoid using `Value.exp()` and reciprocal power.
3. In `objective`, average the squared errors on the four rows in `DATA`.
4. In `clear_gradients`, set each parameter's `.grad` to zero.
5. In `update`, subtract `learning_rate * grad` from each parameter's `.data`.
   This runs only after backward has finished using the forward values.
6. In the supplied `train` loop, replace the first error with a call to
   `clear_gradients(p)` so this step uses fresh parameter gradients.
7. Replace the next error with `loss = objective(p)` to record the forward
   computation at the current parameters.
8. Replace the next error with `loss.backward()` to compute the derivatives.
9. Replace the final error with `update(p, learning_rate)`. Keep the supplied
   defaults: 2000 iterations and learning rate 0.5.
10. Run the completed script. Its supplied `run` function compares every
    first-step gradient with PyTorch before training starts.
11. Record its results and explain them in `RESULTS.md`.

## 5. What to record

Record initial and final mean squared error, whether the maximum first-step
gradient discrepancy is below 1e-12, all four final probabilities in DATA
order, and whether all four classifications are correct at threshold 0.5.
Keep at least four significant figures.

## 6. What to explain

Answer the three questions in `RESULTS.md` in your own words.

## 7. Verification

```bash
.venv/bin/python chapter-10-autograd/applied/based-on/autograd_xor.py
.venv/bin/python chapter-10-autograd/applied/full-code/verify_results.py chapter-10-autograd/applied/RESULTS.md
```

The verifier prints `OK` for each matching field and exits with status zero.
With no results-file argument it prints the reference values. The explanations
require human assessment; numerical agreement does not establish understanding.

## 8. Troubleshooting

- `NotImplementedError`: complete the numbered function named in the message.
- Wrong first gradients: check parameter order, ReLU derivatives and division
  by the number of rows. Do not update parameters before the comparison.
- Loss stops improving: check that gradients are cleared and the graph is
  rebuilt on every step; keep the supplied seed and initialisation.
- The verifier rejects rounded values: keep four or more significant figures.
