# Chapter 11 — GPU Computing with CUDA

Small examples of device selection, event timing, memory accounting, pinned
transfers, float16 mixed precision, compilation, a Triton kernel and profiling.
Inputs are synthetic; there are no external datasets or weights.

## Environment

Verified on 2026-10-16 with Python 3.14.4, PyTorch 2.14.1+cu132, Triton 3.8.0,
Linux x86-64, NVIDIA RTX 4050 Laptop (6 GB, compute capability 8.9), driver
595.99.02. Use the environment from chapter 9. To choose this optional CUDA
build instead of its CPU build, run from the repository root:

```bash
uv pip install -p .venv -r requirements-cuda.txt
uv pip install -p .venv -r requirements.txt
```

The first command takes PyTorch and Triton from PyTorch's CUDA index; the second
installs everything else from PyPI and keeps `torch 2.14.1+cu132`.

A compatible NVIDIA driver is required. The wheel supplies its CUDA runtime;
`nvidia-smi`'s CUDA version describes driver support, not an installed compiler.
These examples do not call `nvcc`. Compilation and Triton additionally need
a C compiler and matching Python development headers. On Ubuntu with system
Python 3.14, the administrator can install `build-essential` and `python3.14-dev`.
A managed Python distribution may already include its headers.

On the verified machine, missing development headers were unpacked from Ubuntu
package `libpython3.14-dev` version `3.14.4-1ubuntu0.2` into a user-owned toolchain
directory. Validation supplied those two include directories through `CPATH`;
no virtual-environment packages were altered to hide the missing dependency.
For a normal system installation, `Python.h` should be available in the include
directory reported by `python -c 'import sysconfig; print(sysconfig.get_path("include"))'`.

## Run

From the repository root, for example:

```bash
.venv/bin/python chapter-11-cuda/device_demo.py
.venv/bin/python chapter-11-cuda/work_model.py
.venv/bin/python chapter-11-cuda/timing_demo.py
.venv/bin/python chapter-11-cuda/memory_transfer.py
.venv/bin/python chapter-11-cuda/amp_demo.py
.venv/bin/python chapter-11-cuda/compile_demo.py
.venv/bin/python chapter-11-cuda/triton_add.py
.venv/bin/python chapter-11-cuda/profile_demo.py
```

The device demo falls back to CPU and prints the matrix product
`[[5.0, 14.0], [14.0, 50.0]]`. The other demos print `SKIP: CUDA unavailable`
when there is no accessible CUDA GPU. GPU tests skip in that case; tests also
hide the GPU in subprocesses to check these paths on a GPU machine.
`work_model.py` is a CPU arithmetic model, not a performance measurement: it
checks launch coverage (four programs, 21 masked offsets) and idealised
arithmetic intensities (0.083333 and 85.333333 FLOP per byte).

Expected checks on CUDA:

| Script | Expected result |
| --- | --- |
| `timing_demo.py` | Positive event and wall times, CPU reference passed |
| `memory_transfer.py` | 4096 payload bytes, live allocation released, reserved covers live |
| `amp_demo.py` | float16 prediction, float32 loss, updated weights and finite gradients |
| `compile_demo.py` | 257 compiled values and gradients checked |
| `triton_add.py` | 1003 outputs exactly matching PyTorch |
| `profile_demo.py` | Operator table, matrix multiplication recorded, finite output |

Times depend on clocks, contention, dispatch and hardware; the tests impose
no speedup requirement. Compilation has a first-call cost and requires its
toolchain even though eager CUDA operations already work. The Triton example
accepts only same-device contiguous float32 vectors without autograd; it is
not a differentiable replacement for `torch.add`.

The profiler requires CUDA profiling support. Restricted/container installations
may allow CUDA execution while denying profiling; that is an environment issue
to resolve before running the profiling example.

## Applied exercise

See [instructions](applied/INSTRUCTIONS.md) and the
[results template](applied/RESULTS.md). The full solution compares first-step
gradients and final fitting errors on CPU, CUDA float32 and CUDA AMP. Only CPU
metrics appear without CUDA. Run the complete checks with `./scripts/check-all.sh`.
