# Chapter 20 — Generative Models

Examples verified on a CPU with Python 3.14.4, NumPy 2.5.4, scikit-learn 1.9.1
and PyTorch 2.14.1 (17 October 2026). Run commands from the repository root.
The digits dataset is bundled with scikit-learn (UCI, CC BY 4.0); the ring data
are generated. Nothing is downloaded.

| File | What is checked |
|---|---|
| `autoencoder.py`, `anomaly_scan.py` | Autoencoder vs PCA anomaly scores, each digit class held out in turn, three seeds |
| `elbo_example.py` | ELBO of a one-dimensional Gaussian model in closed form and by reparameterised sampling |
| `vae.py`, `vae_demo.py` | Variational autoencoder: test ELBO terms and which digits its samples resemble, latent size 2 vs 8 |
| `gan_optimal_d.py` | The optimal discriminator and the Jensen-Shannon divergence for two Gaussians |
| `ring_data.py`, `gan_2d.py` | A GAN on eight clusters: modes covered, three seeds |
| `diffusion_2d.py` | A denoising diffusion model on the same clusters; forward process checked step by step |
| `applied/` | Anomaly alarm: threshold, recall, false alarms, precision at low prevalence |
| `vae_samples.py` | Figure data | Thirty new digits from each of two VAEs |

```bash
.venv/bin/python chapter-20-generative/elbo_example.py
.venv/bin/python chapter-20-generative/anomaly_scan.py
.venv/bin/python chapter-20-generative/vae_demo.py
.venv/bin/python chapter-20-generative/gan_2d.py
.venv/bin/python chapter-20-generative/diffusion_2d.py
.venv/bin/python -m pytest chapter-20-generative --ignore-glob='*/based-on/*'
```

Expected output of `elbo_example.py` (the sampled values can vary in the last digit):

```text
q = N(1.0, 0.64): expected log-likelihood -1.7389, KL 0.5431, ELBO -2.2821, sampled -2.2824
q = N(1.0, 0.50): expected log-likelihood -1.6689, KL 0.5966, ELBO -2.2655, sampled -2.2655
log p(x) = -2.2655
```

`anomaly_scan.py` reports a mean AUC over the ten held-out classes of 0.880 for
PCA and 0.912 for the autoencoder; `gan_2d.py` covers 8, 7 and 8 of the 8
modes for seeds 0, 1 and 2; `diffusion_2d.py` covers all 8 for every seed.
Trained results can differ in the last digits across platforms.

## Applied exercise

Start with `applied/INSTRUCTIONS.md`.

```bash
.venv/bin/python chapter-20-generative/applied/full-code/anomaly_threshold.py
.venv/bin/python chapter-20-generative/applied/full-code/verify_results.py
```

Reference output:

```text
test_items: 506
test_anomalies: 183
ae_recall: 0.306011
ae_false_alarm: 0.0660475
ae_precision: 0.716316
ae_precision_at_1pct: 0.0437215
pca_recall: 0.123862
pca_false_alarm: 0.0515996
pca_precision: 0.556588
pca_precision_at_1pct: 0.0232948
```
