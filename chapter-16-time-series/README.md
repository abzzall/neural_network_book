# Chapter 16 — Time-Series Data

Examples verified on a CPU with Python 3.14.4, NumPy 2.5.4, pandas 3.0.6 and
PyTorch 2.14.1 (16 October 2026). Run commands from the repository root.

| File | What is checked |
|---|---|
| `mase_example.py` | Seasonal-naive and naive forecasts, MAE and MASE on twelve values |
| `bike_data.py` | Download (hash-checked) and summary of daily bike rentals |
| `components.py` | Autocorrelation from scratch, before and after differencing |
| `baselines.py`, `metrics.py` | Naive, seasonal naive, moving average, Holt-Winters; MAE, RMSE, MAPE, MASE, pinball loss, coverage |
| `backtest.py` | Rolling-origin backtest of the baselines: 25 weekly origins, 14-day horizon |
| `windows.py`, `linear_direct.py` | Direct multi-output linear forecasts, with and without the known-future calendar |
| `strategies.py` | Recursive versus direct multi-step forecasting |
| `tcn.py`, `tcn_backtest.py` | Temporal convolutional network: causality, receptive field, backtest over three seeds |
| `pretrained_forecast.py` | Optional: zero-shot forecasts from a pinned pretrained model |
| `applied/` | Probabilistic forecasts: linear quantile model versus a naive interval |

`mase_example.py` needs no download. The others need the data:

```bash
.venv/bin/python chapter-16-time-series/bike_data.py --download
.venv/bin/python chapter-16-time-series/backtest.py
.venv/bin/python chapter-16-time-series/linear_direct.py
.venv/bin/python chapter-16-time-series/tcn_backtest.py
.venv/bin/python -m pytest chapter-16-time-series --ignore-glob='*/based-on/*'
```

Expected output of `backtest.py`:

```text
origins: 25, horizon: 14 days
naive           MAE  1385.8  MASE 1.585
seasonal naive  MAE  1114.6  MASE 1.272
7-day mean      MAE  1055.4  MASE 1.205
Holt-Winters    MAE  1055.9  MASE 1.204
```

The TCN reaches mean MASE 1.082 over seeds 42, 43 and 44 (standard deviation
0.025); last digits can vary across platforms.

## Optional pretrained forecaster

`pretrained_forecast.py` needs the extra packages in `requirements-chronos.txt`
and a one-time download of the model:

```bash
uv pip install -p .venv -r requirements-chronos.txt
.venv/bin/python chapter-16-time-series/pretrained_forecast.py --download
```

Its test is skipped when the package or the model is missing. Reference output:

```text
median forecast: MAE   993.9  MASE 1.135
80% interval coverage: 0.757
mean pinball loss over 3 quantiles: 343.1
```

## Data and model

- **Bike Sharing Dataset** (731 days, 2011-01-01 to 2012-12-31), H. Fanaee-T,
  UCI Machine Learning Repository, doi 10.24432/C5W894, licensed CC BY 4.0.
  The dataset's readme asks users to cite Fanaee-T and Gama (2013). Downloaded
  by `bike_data.py` and checked against a SHA-256 hash; not stored here.
- **Chronos-Bolt tiny** (`amazon/chronos-bolt-tiny`, revision
  `a0e552de83495b5c28c14c71c374f3e33280b340`), Apache-2.0 licence, used through
  `chronos-forecasting` 2.3.2 (Apache-2.0). Cached in `data/`; not stored here.

## Applied exercise

Start with `applied/INSTRUCTIONS.md`. The incomplete `based-on/` functions
raise errors naming the step to complete; the finished `full-code/` solution
passes the same acceptance tests.

```bash
.venv/bin/python chapter-16-time-series/applied/full-code/quantile_bikes.py
.venv/bin/python chapter-16-time-series/applied/full-code/verify_results.py
```

Reference output (last digits can vary across platforms):

```text
train_windows: 478
origins: 25
quantile_mase: 1.12876
quantile_coverage: 0.822857
quantile_pinball: 359.244
naive_mase: 1.27554
naive_coverage: 0.757143
naive_pinball: 394.989
```
