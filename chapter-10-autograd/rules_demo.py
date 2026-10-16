"""Small executable examples of the supported autograd rules."""

import numpy as np

from tiny_tensor import Tensor, no_grad

x = Tensor([2., 3.], requires_grad=True)
y = x * x
y.retain_grad()
handle = y.register_hook(lambda g: g * 2)
y.backward(np.ones(2), retain_graph=True)
print(f'leaf: {x.is_leaf}, result leaf: {y.is_leaf}')
print(f'hooked leaf gradient: {x.grad.tolist()}')
print(f'retained gradient: {y.grad.tolist()}')
handle.remove()
y.backward(np.ones(2))
print(f'accumulated gradient: {x.grad.tolist()}')
with no_grad():
    print(f'no_grad result requires grad: {(x * x).requires_grad}')
z = x * x
x.detach().add_(1)
try:
    z.sum().backward()
except RuntimeError:
    print('detached mutation: version mismatch detected')
