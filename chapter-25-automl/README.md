# Automated model design

`nas_search.py` searches eight MLP depth/width and regularisation choices on a
validation split, refits the selected procedure on the development data, and uses
the test set once. It is deliberately small enough to show the protocol on a CPU;
production NAS uses more efficient search and substantially larger budgets.

```bash
.venv/bin/python chapter-25-automl/nas_search.py
.venv/bin/python -m pytest chapter-25-automl
```

The applied runner compares exhaustive search and successive halving. The
`*_epochs` fields describe allocated maximum candidate-epoch budgets. Actual
iterations and fitting time are separate `*_actual_epochs` and `*_fit_seconds`
fields; restarts and early convergence are included in those measurements.

Measured search-only example on the verification CPU with bounded BLAS threads:
exhaustive search used 1,275 actual fitting epochs (2,000 allocated), and halving
710 (900 allocated). Fit times were 19.7 and 18.6 seconds. Both selected
`(128, 64, 32), alpha=0.0001`, validation accuracy 0.9639 and final test accuracy
0.9750. Timings depend on hardware and concurrent work; allocated budgets do not
promise a proportional wall-time reduction.
