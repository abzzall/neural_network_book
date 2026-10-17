"""NumPy arrays: shape, dtype, indexing, views and copies, boolean masks."""

import numpy as np

a = np.arange(12).reshape(3, 4)                 # 3 rows, 4 columns
print('a =\n', a)
print('shape', a.shape, 'dtype', a.dtype, 'size', a.size)
print('a[1, 2] =', a[1, 2], ' row 0 =', a[0], ' column 1 =', a[:, 1])
print('every second column:\n', a[:, ::2])

view = a[0]                                     # slicing gives a view ...
view[0] = 100
print('after changing the view, a[0, 0] =', a[0, 0])
copy = a[1].copy()                              # ... copy() gives independent data
copy[0] = -1
print('after changing the copy, a[1, 0] =', a[1, 0])

mask = a % 2 == 0                               # boolean mask, same shape as a
print('even entries:', a[mask])
print('column means:', a.mean(axis=0), ' row sums:', a.sum(axis=1))

x = np.array([1, 2, 3], dtype=np.float32)
print('float32 vector:', x, x.dtype, ' x @ x =', x @ x)
