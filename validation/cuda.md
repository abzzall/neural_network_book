# Chapter 11 validation

Verified on 2026-10-16 on Linux x86-64 with Python 3.14.4,
PyTorch 2.14.1+cu132, Triton 3.8.0, driver 595.99.02 and an
NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9).

## Compiler prerequisites

Eager CUDA ran before compiler validation. Compilation first failed because
`Python.h` was missing. The matching Ubuntu package `libpython3.14-dev`
(`3.14.4-1ubuntu0.2`) was downloaded and extracted into a user-owned directory.
The compiler was given both `usr/include/python3.14` and `usr/include` from
that extraction through `CPATH`. The second directory is needed for the
multiarch `pyconfig.h` include. No virtual-environment files were patched.
Installing matching development headers system-wide is an alternative.

To reproduce with an extracted package, replace the path below with its root:

```bash
export CPATH="/path/to/extraction/usr/include/python3.14:/path/to/extraction/usr/include"
./scripts/check-all.sh
```

Use the chapter README for the optional package versions and normal setup.
Cached compiled artifacts can mask missing compiler prerequisites; a fresh
machine still needs those prerequisites even if a warmed machine runs without
an include-path override.

## Numerical and portability checks

The repository-wide pytest run passed all 181 tests, including the CUDA tests.

- Eager matrix multiplication agrees with the CPU reference.
- Event and wall timers are positive; no speedup threshold is imposed.
- Pinned transfer preserves values; live storage is released and reservation
  covers allocation.
- AMP produces float16 predictions, float32 loss and finite unscaled gradients;
  its example updates the model weights.
- Compiled forward values and input gradients agree with eager execution.
- Triton addition matches PyTorch exactly, including empty vectors and lengths
  around a tile boundary. Unsupported layout, dtype and gradient inputs fail.
- Profiling records matrix multiplication and produces finite results.
- Hiding CUDA with `CUDA_VISIBLE_DEVICES=''` gives 12 passed, 16 skipped in
  the chapter tests. Each CUDA-only script prints an explicit skip.
- The unfinished exercise gives four failures and one passing guard test.
  Filling its numbered steps reproduces the full solution exactly and gives
  five passing acceptance tests from a temporary directory.
- The results verifier accepts a local completed record and rejects invalid
  or corrupted measurements.

The exercise's measured CPU initial/final MSE is `12.82404899597168` /
`1.2942019678163774e-9`. CUDA float32 final MSE is
`1.2938585758348609e-9`, with first-gradient error `9.5367431640625e-7`.
AMP final float32-evaluation MSE is `7.958044534461806e-7`, with first-gradient
error `0.0013284683227539062`. The smaller initial scale of 128 keeps that
first inspection finite; the default larger scale overflowed for this dataset.
The acceptance bounds in the exercise allow normal numerical variation.

No multi-GPU execution was tested. No measured timing is a portable expected
value or evidence of speedup for these small examples.
