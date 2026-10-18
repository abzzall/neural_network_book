# Chapter 13 — Recurrent Neural Networks

Examples verified on a CPU with Python 3.14.4, NumPy 2.5.4 and PyTorch 2.14.1
(16 October 2026). Use the existing environment described in chapter 9.
Run commands from the repository root.

| File | What is checked |
|---|---|
| `rnn_numpy.py` | Tanh recurrence and manual BPTT; every parameter, input and initial-state derivative against PyTorch; all parameter gradients against central differences |
| `bptt_demo.py` | A two-step worked example with hidden values, loss and derivatives |
| `gradient_paths.py` | Vanishing/exploding scalar state derivatives and gradient norm clipping |
| `gates_demo.py` | LSTM and PyTorch-style GRU gate equations against library cells |
| `char_rnn.py` | A small next-character model, trained on an author-written toy corpus |
| `applied/` | Rolling one-step GRU forecasts compared with a seven-day seasonal baseline |
| `forecast_picture.py` | Figure data | The seasonal-correction GRU against the data over the validation and test steps |

```bash
.venv/bin/python chapter-13-rnn/bptt_demo.py
.venv/bin/python chapter-13-rnn/gradient_paths.py
.venv/bin/python chapter-13-rnn/gates_demo.py
.venv/bin/python chapter-13-rnn/char_rnn.py
.venv/bin/python -m pytest chapter-13-rnn --ignore-glob='*/based-on/*'
```

Expected output of `bptt_demo.py`:

```text
hidden: 0.537050, -0.015736
class 0 probabilities: 0.745376, 0.492132
mean cross-entropy: 0.485701
dWx: -0.339664, dWh: 0.264234, dbh: 0.398352
input gradients: -0.046830, 0.246005
initial-state gradient: -0.023415
```

The character example reduces training cross-entropy from 2.812362 to 0.005467
on the verified environment. It demonstrates fitting an intentionally repeated
corpus; this is not a held-out language-model score. Its training text is
original material supplied under this repository's MIT licence. Both examples
use CPU execution; no GPU, download or pretrained weights are required.

## Applied exercise

Start with `applied/INSTRUCTIONS.md`. The incomplete `based-on/` functions
raise errors naming the step to complete; the finished `full-code/` solution
passes the same acceptance tests. Record values in `applied/RESULTS.md` and
check the completed table with `applied/full-code/verify_results.py`.

```bash
.venv/bin/python chapter-13-rnn/applied/full-code/recurrent_forecast.py
.venv/bin/python chapter-13-rnn/applied/full-code/verify_results.py
```

Reference output (last digits can vary across platforms):

```text
train_windows: 232
selected_epoch_seed42: 112
validation_seasonal_mae: 1.76248
validation_gru_mae: 0.407699
validation_gru_std: 0.0570718
test_seasonal_mae: 1.69621
test_gru_mae: 1.08933
test_gru_std: 0.0525891
```

GRU errors are means over seeds 42, 43 and 44; standard deviations are population
standard deviations of those three errors, not uncertainty estimates over data.
Scaling is fitted only before index 260; validation targets are 260..339; test
targets are 340..419. Windows contain 28 earlier observations. This protocol
observes each test value before constructing later test windows; it is not a
multi-step forecast from one fixed origin. The generated demand series and the
character corpus are teaching examples, not representative real-world datasets.
