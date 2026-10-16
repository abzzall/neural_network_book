# Chapter 10 applied exercise — results

Fill in the Value column. Probabilities follow the order of DATA:
(-1, -1), (-1, 1), (1, -1), (1, 1). Keep four or more significant figures.

| Quantity | Value |
|---|---|
| `initial_mse` | |
| `final_mse` | |
| `first_gradient_error_below_1e-12` | |
| `p_0` | |
| `p_1` | |
| `p_2` | |
| `p_3` | |
| `all_four_correct` | |

## What to explain

1. Why must parameter gradients be cleared between training steps even though
   intermediate gradients are reset inside Value.backward?
2. Why compare every gradient before the first update, rather than just
   comparing the final loss after training?
3. Does classifying all four training points establish generalisation to
   unseen data? What does this experiment actually verify?
