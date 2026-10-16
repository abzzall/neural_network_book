"""Copy pinned input safely and inspect live versus reserved GPU memory."""

import torch


def run():
    host = torch.arange(1024, dtype=torch.float32).pin_memory()
    before = torch.cuda.memory_allocated()
    device = host.to('cuda', non_blocking=True)
    # Keep host alive and unchanged until the copy has completed.
    torch.cuda.synchronize()
    increase = torch.cuda.memory_allocated() - before
    reserved = torch.cuda.memory_reserved()
    live = torch.cuda.memory_allocated()
    torch.testing.assert_close(device.cpu(), host, rtol=0, atol=0)
    payload = device.numel() * device.element_size()
    del device
    after = torch.cuda.memory_allocated()
    return payload, increase, reserved >= live, after == before


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('SKIP: CUDA unavailable')
    else:
        payload, increase, covers, released = run()
        print(f'payload bytes: {payload}')
        print(f'allocated increase bytes: {increase}')
        print(f'reserved covers allocated: {covers}')
        print(f'live allocation released: {released}')
