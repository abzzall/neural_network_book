# Appendix A — Python and NumPy

Examples verified with Python 3.14.4, NumPy 2.5.4 and matplotlib 3.11.2
(17 October 2026). Run commands from the repository root.

| File | What is shown |
|---|---|
| `python_basics.py` | Lists, dictionaries, comprehensions, loops, functions, a class |
| `numpy_arrays.py` | Shape, dtype, indexing and slicing, views and copies, masks, reductions |
| `broadcasting.py` | Broadcasting rules; a distance matrix with loops and vectorised |
| `plotting.py` | A matplotlib figure saved to `outputs/appendix_a_sigmoid.png` |

```bash
.venv/bin/python appendix-a-python-numpy/numpy_arrays.py
.venv/bin/python -m pytest appendix-a-python-numpy
```

Timings in `broadcasting.py` depend on the machine; the tests check only that
both methods agree and that the vectorised one is faster. The integer `dtype`
printed by `numpy_arrays.py` is `int64` on the verified Linux system.
