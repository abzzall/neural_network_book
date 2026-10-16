# Chapter 16 applied exercise — results

Fill the Value column with four or more significant figures.

| Quantity | Value |
|---|---|
| `train_windows` | |
| `origins` | |
| `quantile_mase` | |
| `quantile_coverage` | |
| `quantile_pinball` | |
| `naive_mase` | |
| `naive_coverage` | |
| `naive_pinball` | |

## What to explain

1. A nominal 80% interval covers 82% of the actual values for one method and
   76% for the other. Which deviation would you worry about more, and why is
   coverage alone not enough to compare intervals?
2. Why does the pinball loss reward a narrow interval only when it is also
   correctly placed?
3. All 25 origins fall on a Sunday. What could change if the origins were
   spread over all weekdays, and how would you test it?
4. Which inputs of the quantile model are known at the forecast origin, and
   which would be unavailable in real use if you added the temperature?
