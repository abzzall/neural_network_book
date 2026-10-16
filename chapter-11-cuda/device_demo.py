"""Select a device and verify a small matrix product against the CPU."""

import torch


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    result = x.to(device) @ x.to(device).T
    torch.testing.assert_close(result.cpu(), x @ x.T)
    return device.type, result.cpu().tolist()


if __name__ == '__main__':
    device, result = run()
    print(f'device: {device}')
    print(f'matrix product: {result}')
