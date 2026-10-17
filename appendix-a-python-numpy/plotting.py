"""A matplotlib figure saved to a file: a function, its derivative, labels, legend."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')                           # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 4, 200)
sigmoid = 1 / (1 + np.exp(-x))

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(x, sigmoid, color='black', label='sigmoid')
ax.plot(x, sigmoid * (1 - sigmoid), color='black', linestyle='--', label='derivative')
ax.set_xlabel('x')
ax.set_ylabel('value')
ax.legend()
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / 'outputs' / 'appendix_a_sigmoid.png'
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print('saved', out.name, 'with', len(ax.lines), 'curves')
