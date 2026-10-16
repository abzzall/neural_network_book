# Chapter 4 applied exercise — cleaning and preparing a table

## 1. Outcome

A cleaned table of 500 simulated apartment listings, prepared from scratch
(imputation, scaling, one-hot encoding fitted on training rows only), a line
fitted by least squares, and a measured example of target leakage.

## 2. Verified versions

Python 3.14.4, NumPy 2.5.4, pandas 3.0.6, pytest 9.1.1 (2026-10-16). Runs on a CPU in about a second.

## 3. Files supplied

- `../messy_data.py` — generates the listings table and documents its defects.
- `based-on/prepare.py` — the starting point; each function raises
  `NotImplementedError` naming its step.
- `based-on/test_prepare.py` — the tests your code must pass.
- `full-code/` — the finished solution, its tests, and `verify_results.py`.
- `RESULTS.md` — the values to record and the questions to answer.

## 4. Steps

Work in `based-on/prepare.py` and run the tests after each step:

```bash
.venv/bin/python -m pytest chapter-04-data/applied/based-on
```

1. `normalise_districts`: strip spaces and lower-case the district names.
2. `remove_duplicates`: drop exact duplicate rows.
3. `mark_impossible`: replace areas above 400 m2 by NaN.
4. `split`: an 80/20 split with `nnbook.preprocessing.train_test_split` (seed 0).
5. `fit_preparation` and `transform`: medians, scaling and districts learned from the training rows.
6. `fit_linear` and `rmse`: least squares as in chapter 2, and the validation error.
7. Run `run(leak=True)`, which adds `valuation_after_sale_k` to the features.

## 5. What to record

Fill in the table in `RESULTS.md`. `rows_before` and `district_spellings_before`
come from `make_listings()` before step 1.

## 6. What to explain

Answer the three questions in `RESULTS.md` in your own words.

## 7. Verification

```bash
.venv/bin/python chapter-04-data/applied/full-code/verify_results.py chapter-04-data/applied/RESULTS.md
```

Every line must start with `OK`.

## 8. Troubleshooting

- `ChainedAssignmentError` — pandas 3 does not allow `df["a"][mask] = value`;
  use `df.loc[mask, "a"] = value` or `df.assign(...)`.
- `ValueError: could not convert string to float` — a text column reached NumPy;
  select only the numeric columns before `to_numpy`.
