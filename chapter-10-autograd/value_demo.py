"""Chapter 10: the scalar engine on the examples of chapters 2 and 6, checked with PyTorch."""

import torch

from value import Value

# Chapter 2: L = (w x + b - y)^2 at x = 1, y = 2, w = b = 0.
w, b = Value(0.0), Value(0.0)
loss = (w * 1.0 + b - 2.0) ** 2
loss.backward()
print(f"chapter 2: dL/dw = {w.grad}, dL/db = {b.grad}")

# A neuron with a shared input: z = x*w1 + x*w2, so x feeds two operations.
x, w1, w2 = Value(2.0), Value(-3.0), Value(0.5)
z = (x * w1 + x * w2).relu() + (x * w1).exp()
z.backward()
xt, w1t, w2t = (torch.tensor(v, requires_grad=True) for v in (2.0, -3.0, 0.5))
zt = torch.relu(xt * w1t + xt * w2t) + torch.exp(xt * w1t)
zt.backward()
print(f"shared input: dz/dx = {x.grad:.6f} (PyTorch {xt.grad.item():.6f}), "
      f"dz/dw1 = {w1.grad:.6f} (PyTorch {w1t.grad.item():.6f})")

# Calling backward twice adds the gradients, as PyTorch does.
w.grad = b.grad = 0.0
loss = (w * 1.0 + b - 2.0) ** 2
loss.backward()
loss.backward()
print(f"after two backward calls: dL/dw = {w.grad}")
