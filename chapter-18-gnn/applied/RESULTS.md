# Measured results

The fixed original synthetic graph has 160 nodes, eight features, two balanced
classes, and a 32/32/96 node split. All node features and edges are visible;
only training labels enter the loss. This is transductive evaluation.

| Depth (0 = MLP) | Mean validation loss | Mean test accuracy | Seed std |
|---|---:|---:|---:|
| 0 | 0.670685 | 0.625000 | 0.069618 |
| 1 | 0.354093 | 0.854167 | 0 |
| 2 | 0.041475 | 0.989583 | 0 |
| 4 | 0.001080 | 1.000000 | 0 |
| 8 | 0.693126 | 0.500000 | 0 |

Mean validation cross-entropy selects depth 4. Each run restores the checkpoint
with lowest validation loss within 200 epochs. Seeds 42/43/44 change model
initialisation, not the graph or its split. The eight-layer result alone
does not prove over-smoothing: optimisation and activation effects also matter.
The separate `smoothing_demo.py` isolates repeated linear mixing.
