# Chapter 3 — Probability, Statistics, and Information Theory

Executable material for chapter 3 of the book. Run every command from the
repository root.

| File | Book | What it shows |
|---|---|---|
| `bayes_demo.py` | Worked example | Bayes' rule for a screening test for a rare condition |
| `stats_demo.py` | Listing 3.1 | Mean, covariance and correlation by formula, checked against NumPy |
| `mle_demo.py` | Listing 3.2 | Maximum likelihood for a coin, by formula and by gradient descent |
| `entropy_demo.py` | Listing 3.3 | Entropy, cross-entropy, and the asymmetry of the KL divergence |
| `test_scripts.py` | — | Runs every script and checks every printed line |
| `applied/` | Applied analysis | Maximum likelihood for a Bernoulli and a Gaussian, two ways |

```bash
.venv/bin/python chapter-03-probability/mle_demo.py
.venv/bin/python -m pytest chapter-03-probability
```

Expected output of `mle_demo.py`:

```text
heads = 7 of 10
maximum-likelihood estimate k/n = 0.7000
gradient descent on the cross-entropy: p = 0.7000
cross-entropy at p = 0.5: 0.6931, at p = 0.7: 0.6109
```

## Applied exercise

Start from `applied/INSTRUCTIONS.md`. The starting point is `applied/based-on/`
(its tests fail until you complete it); the finished code is
`applied/full-code/`.
