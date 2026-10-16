# Chapter 7 applied exercise — results

Fill in the second column with the values your code prints. For a setting that
never reaches the target loss, write `never`. Then run, from the repository root:

```bash
.venv/bin/python chapter-07-optimization/applied/full-code/verify_results.py chapter-07-optimization/applied/RESULTS.md
```

| Quantity | Value |
|---|---|
| `full_batch_epochs_to_0.05` | |
| `full_batch_test_accuracy` | |
| `sgd_epochs_to_0.05` | |
| `sgd_test_accuracy` | |
| `momentum_epochs_to_0.05` | |
| `momentum_test_accuracy` | |
| `adam_epochs_to_0.05` | |
| `adam_test_accuracy` | |

## What to explain

1. Full-batch gradient descent and mini-batch SGD see every sample once per
   epoch. Why does mini-batch SGD get so much further in 30 epochs?
2. Why does momentum reach the target loss so much sooner than plain SGD with
   the same learning rate?
3. Adam with its default learning rate is not the fastest here. What would you
   try before concluding that it is worse for this problem?
