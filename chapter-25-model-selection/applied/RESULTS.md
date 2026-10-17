# Chapter 25 applied exercise — results

| Quantity | Value |
|---|---|
| `bayes_threshold` | |
| `approve_all_cost` | |
| `logistic_at_0.5_cost` | |
| `logistic_at_cost_threshold_cost` | |
| `logistic_at_cost_threshold_approved` | |
| `boosting_at_cost_threshold_cost` | |

## What to explain

1. Why is the cost of approving everyone exactly 1.5?
2. The cost-optimal threshold approves far fewer applicants. Why is that cheaper,
   and who bears the cost of the rejected good applicants?
3. Gradient boosting has nearly the same AUC as logistic regression, yet costs more
   at the same threshold. What property of its probabilities matters here?
4. Which fairness checks of chapter 23 would you repeat after changing the threshold?
