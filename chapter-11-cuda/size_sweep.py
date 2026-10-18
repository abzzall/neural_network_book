"""CPU against GPU for square float32 matrix products of growing size.

Both sides are warmed up and timed over repeated products with the inputs
already in place; the GPU side uses CUDA events. A second GPU column adds the
copies of the inputs to the device and of the result back, which a program
pays when its data do not stay on the GPU. Each time is the median of five
measurements, because single measurements of short products vary from run to run.
"""

from statistics import median
from time import perf_counter

import torch


def cpu_ms(a, b, repeats):
    for _ in range(3):
        a @ b
    start = perf_counter()
    for _ in range(repeats):
        a @ b
    return (perf_counter() - start) * 1000 / repeats


def gpu_ms(a, b, repeats, with_copies):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    a_dev, b_dev = a.cuda(), b.cuda()
    for _ in range(3):
        a_dev @ b_dev
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        if with_copies:
            (a.cuda() @ b.cuda()).cpu()
        else:
            a_dev @ b_dev
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('SKIP: CUDA unavailable')
    else:
        torch.manual_seed(0)
        print(f'{"n":>5} {"CPU ms":>10} {"GPU ms":>10} {"GPU+copy ms":>12} {"speed-up":>9}')
        for n in (32, 128, 512, 2048):
            a, b = torch.randn(n, n), torch.randn(n, n)
            repeats = 200 if n <= 512 else 20
            cpu = median(cpu_ms(a, b, repeats) for _ in range(5))
            gpu = median(gpu_ms(a, b, repeats, with_copies=False) for _ in range(5))
            copy = median(gpu_ms(a, b, repeats, with_copies=True) for _ in range(5))
            print(f'{n:5d} {cpu:10.4f} {gpu:10.4f} {copy:12.4f} {cpu / gpu:9.1f}')
