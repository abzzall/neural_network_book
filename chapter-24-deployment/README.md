# Chapter 24 — Deployment and MLOps

Examples verified on a CPU with Python 3.14.4, NumPy 2.5.4 and PyTorch 2.14.1
(17 October 2026). Run commands from the repository root. The digit model reuses
the data helpers of chapter 12 and is cached in `outputs/` on first use.

| File | What is checked |
|---|---|
| `quantize_numpy.py` | Affine int8 quantisation: worked example and a whole layer |
| `digit_model.py` | The chapter's small CNN for 8x8 digits, trained once and cached |
| `fgsm_demo.py` | Adversarial examples by the fast gradient sign method versus random noise |
| `export_onnx.py` | Optional: ONNX export, ONNX Runtime check, int8 quantisation, batching |
| `serve_api.py` | Optional: a FastAPI prediction service with input validation |
| `applied/` | Drift monitoring with PSI and the Kolmogorov-Smirnov statistic |

```bash
.venv/bin/python chapter-24-deployment/quantize_numpy.py
.venv/bin/python chapter-24-deployment/fgsm_demo.py
.venv/bin/python -m pytest chapter-24-deployment --ignore-glob='*/based-on/*'
```

Expected output of `quantize_numpy.py`:

```text
scale 0.011765, zero point -43
int8 codes: [-128, -43, -1, 127]
dequantised: -1.0000 0.0000 0.4941 2.0000
layer: 262144 bytes -> 65536 bytes, max error 0.00181 (half a step: 0.00181)
```

## Optional export and serving

`export_onnx.py` and `serve_api.py` need the packages in
`requirements-deployment.txt` (onnx and onnx-ir: Apache-2.0; onnxruntime,
onnxscript and FastAPI: MIT; httpx: BSD-3-Clause):

```bash
uv pip install -p .venv -r requirements-deployment.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv/bin/python chapter-24-deployment/export_onnx.py
```

Their tests are skipped when the packages are missing. On the verified CPU the
float32 ONNX model is 35,606 bytes and the int8 model 15,882 bytes, both with
the PyTorch model's test accuracy of 0.9822; timings depend on the machine.

## Applied exercise

```bash
.venv/bin/python chapter-24-deployment/applied/full-code/drift_monitor.py
.venv/bin/python chapter-24-deployment/applied/full-code/verify_results.py
```

Reference output:

```text
unchanged_intensity_psi: 0.0173
unchanged_confidence_alarm: 0
shifted_accuracy: 0.6556
shifted_intensity_alarm: 0
shifted_confidence_alarm: 1
fainter_accuracy: 0.9822
fainter_intensity_psi: 12.6275
fainter_intensity_alarm: 1
swapped_accuracy: 0.78
swapped_confidence_alarm: 0
```
