# Compare CPU and CUDA training

## Outcome

Complete a device-aware linear-regression training step. Compare first-step
gradients and final mean squared error (MSE) on CPU, CUDA float32 and CUDA
float16 autocast. This is a numerical correctness exercise, not a speed contest.

## Verified versions

2026-10-16: Python 3.14.4, PyTorch 2.14.1+cu132, RTX 4050 Laptop, driver 595.99.02.
The CPU branch needs no GPU. The exercise functions use only PyTorch;
Triton and a compiler are needed only for the chapter's separate kernel demos.

## Files supplied

`based-on/cuda_training.py` supplies seeded synthetic data, zero initialisation,
the training loop, first-gradient capture and JSON printing. Its tests are
complete. `full-code/` contains the solution and the results verifier.
The synthetic data and supplied code are covered by the repository MIT licence.

## Steps

1. In `move_batch`, return both `x.to(device)` and `y.to(device)`.
   A model and its inputs and targets must use compatible devices.
2. In `forward_loss`, choose `torch.autocast(device_type='cuda', dtype=torch.float16)`
   when `amp` is true, otherwise `nullcontext()`. Keep the supplied `with context:`.
3. Inside that context, return `torch.nn.functional.mse_loss(model(x), y)`.
   The default mean reduction matches the reference objective.
4. In `clear_gradients`, call `optimiser.zero_grad(set_to_none=True)`.
5. In `differentiate`, call `scaler.scale(loss).backward()`.
6. In `update`, call `scaler.step(optimiser)`.
7. Immediately after it, call `scaler.update()` to adapt the scale for the next step.
8. Run the tests and script. Record the JSON and answer the questions below.

The supplied scaler is enabled only for CUDA AMP; when disabled these calls
act as ordinary backward and optimiser step. Its initial scale is 128 to keep
the first captured gradients finite for this dataset. A default larger scale
can overflow and legitimately skip an early update. The supplied first-step
inspection calls `unscale_` before reading gradients; do not divide them again.

## What to record

Copy the printed JSON into `results.json`. Record the exact environment and
whether CUDA ran in `RESULTS.md`. Without CUDA, record only the two CPU fields;
do not invent GPU measurements. Keep the fixed seed, 128 rows, four features,
zero parameters, learning rate 0.1 and 80 iterations.

## What to explain

- Why compare gradients before training rather than only final MSE?
- Why are mixed-precision gradients compared after unscaling?
- Why is final evaluation float32 for every fitted model?
- Why do the small fitting errors provide no evidence about unseen data?
- Why can this small training loop run slower on a GPU?

## Verification

From this folder, using the repository's Python environment:

```bash
../../../.venv/bin/python -m pytest based-on/test_cuda_training.py
../../../.venv/bin/python based-on/cuda_training.py > results.json
../../../.venv/bin/python full-code/verify_results.py results.json
```

CPU final MSE must be below `1e-6`; both GPU final errors below `1e-5`.
The first-gradient maximum absolute difference must be below `1e-5` for GPU
float32 and `0.01` for AMP. The verifier also compares the record with a fresh
local run. These are acceptance bounds, not an assertion of bitwise equality.

## Troubleshooting

A numbered `NotImplementedError` identifies an unfinished step. Device errors
often mean only the inputs or only the model were moved. Infinite first-step
gradients can indicate an excessively large scale. Do not remove scaling to
silence that issue: understand its finite-gradient check and skipped update.
CPU-only runs omit GPU fields; use the same hardware availability when verifying
a record. The exercise fits a synthetic training set and does not evaluate
generalisation.
