"""Two recurrent steps, with every parameter gradient computed by BPTT."""

import numpy as np

from rnn_numpy import RNN

model = RNN(1, 1, 2)
model.Wx[:] = 0.5
model.Wh[:] = 0.25
model.bh[:] = 0.1
model.Wy[:] = [[1., -1.]]
x = np.array([[[1.]], [[-0.5]]])
targets = np.array([[0], [1]])
loss = model.loss(x, targets)
grads, dx, dh0 = model.backward()
print('hidden: ' + ', '.join(f'{h:.6f}' for h in model.cache[1][1:, 0, 0]))
print('class 0 probabilities: ' + ', '.join(f'{p:.6f}' for p in model.probs[:, 0, 0]))
print(f'mean cross-entropy: {loss:.6f}')
print(f'dWx: {grads[0].item():.6f}, dWh: {grads[1].item():.6f}, dbh: {grads[2].item():.6f}')
print('input gradients: ' + ', '.join(f'{d:.6f}' for d in dx.ravel()))
print(f'initial-state gradient: {dh0.item():.6f}')
