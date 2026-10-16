# CUDA training results

Environment (Python, PyTorch build, driver, GPU or CPU-only):

Save the script's complete JSON output as `results.json` next to this file.

| Quantity | Recorded value |
| --- | --- |
| cpu_initial_mse | |
| cpu_final_mse | |
| cuda_fp32_final_mse (if available) | |
| cuda_fp32_gradient_error (if available) | |
| cuda_amp_final_mse (if available) | |
| cuda_amp_gradient_error (if available) | |

Explain in prose:

1. What does the first-gradient comparison isolate?
2. Why must gradients be unscaled before comparison?
3. What does a common float32 evaluation make comparable?
4. What can and cannot be concluded from fitting these synthetic training rows?
5. Why might a larger workload be needed to benefit from a GPU?
