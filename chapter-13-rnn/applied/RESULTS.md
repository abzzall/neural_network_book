# Chapter 13 applied exercise — results

Fill the Value column with four or more significant figures.

| Quantity | Value |
|---|---|
| `train_windows` | |
| `selected_epoch_seed42` | |
| `validation_seasonal_mae` | |
| `validation_gru_mae` | |
| `validation_gru_std` | |
| `test_seasonal_mae` | |
| `test_gru_mae` | |
| `test_gru_std` | |

## What to explain

1. Why is it legitimate for a later test window to contain an earlier observed
   test value? Why would using the current target be leakage?
2. Why is the GRU evaluated against the seven-day baseline instead of just
   reporting its own error? Why can test error exceed validation error?
3. Which changes would be needed for an entire future interval forecast from
   one origin without observing its intermediate values?
