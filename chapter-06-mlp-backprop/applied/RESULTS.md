# Chapter 6 applied exercise — results

Fill in the second column with the values your code prints, to four
significant figures. Then run, from the repository root:

```bash
.venv/bin/python chapter-06-mlp-backprop/applied/full-code/verify_results.py chapter-06-mlp-backprop/applied/RESULTS.md
```

| Quantity | Value |
|---|---|
| `gradient_check_below_1e-5` | |
| `first_loss` | |
| `final_loss` | |
| `train_accuracy` | |
| `test_accuracy` | |
| `test_mistakes` | |

## What to explain

1. Why is the first loss close to log(10) = 2.303?
2. The training accuracy is 1.0 but the test accuracy is lower. What does the
   gap suggest, and which chapter addresses it?
3. Pick one misclassified test digit, display it with
   `plt.imshow(X_test[i].reshape(8, 8))`, and explain in words why the
   confusion is understandable.
