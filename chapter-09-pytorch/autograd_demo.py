"""Chapter 9: autograd reproduces the gradients computed by hand in chapters 2 and 6."""

import torch

# Chapter 2: L = (w x + b - y)^2 at x = 1, y = 2, w = b = 0.
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
loss = (w * 1.0 + b - 2.0) ** 2
loss.backward()                                            # fills w.grad and b.grad

# Chapter 6: the 2-2-1 sigmoid network of the worked example, in float64.
x = torch.tensor([1.0, 0.5], dtype=torch.float64)
W1 = torch.tensor([[0.5, -0.3], [0.2, 0.8]], dtype=torch.float64, requires_grad=True)
b1 = torch.tensor([0.0, 0.1], dtype=torch.float64, requires_grad=True)
w2 = torch.tensor([1.0, -1.5], dtype=torch.float64, requires_grad=True)
b2 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
p = torch.sigmoid(w2 @ torch.sigmoid(W1 @ x + b1) + b2)
nll = -torch.log(p)                                        # target y = 1
nll.backward()

if __name__ == "__main__":
    print(f"chapter 2: dL/dw = {w.grad.item()}, dL/db = {b.grad.item()}")
    print(f"chapter 6: loss = {nll.item():.4f}, dW1 =", W1.grad.round(decimals=4).tolist())
