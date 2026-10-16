"""Measure warmed, resident matrix multiplication with CUDA events."""

from time import perf_counter

import torch


def measure(size=512, repeats=20):
    torch.manual_seed(42)
    a = torch.randn(size, size, device='cuda')
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    for _ in range(5):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = perf_counter()
    start.record()
    for _ in range(repeats):
        torch.mm(a, b, out=out)
    end.record()
    end.synchronize()
    wall_ms = (perf_counter() - wall_start) * 1000 / repeats
    event_ms = start.elapsed_time(end) / repeats
    reference = a.cpu().double() @ b.cpu().double()
    torch.testing.assert_close(out.cpu().double(), reference,
                               rtol=1e-4, atol=1e-4)
    return event_ms, wall_ms


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('SKIP: CUDA unavailable')
    else:
        event_ms, wall_ms = measure()
        print(f'event milliseconds per product: {event_ms:.6f}')
        print(f'wall milliseconds per product: {wall_ms:.6f}')
        print('CPU reference check: passed')
