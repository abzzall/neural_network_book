"""Attribute a small workload to CPU operators and CUDA activity."""

import torch


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('SKIP: CUDA unavailable')
    else:
        torch.manual_seed(42)
        x = torch.randn(256, 256, device='cuda')
        for _ in range(3):
            torch.relu(x @ x)
        torch.cuda.synchronize()
        activities = [torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as profile:
            with torch.profiler.record_function('matrix_then_relu'):
                result = torch.relu(x @ x)
            torch.cuda.synchronize()
        events = profile.key_averages()
        print(events.table(sort_by='self_cuda_time_total', row_limit=8))
        print(f'matrix multiplication recorded: {any(e.key == "aten::mm" for e in events)}')
        print(f'finite output: {torch.isfinite(result).all().item()}')
