# Chapter 18 — Graph Neural Networks

Plain NumPy and PyTorch examples: no graph-specific dependency or external data.
Run commands from the companion repository root using the existing environment.

| File | Purpose |
|---|---|
| `gcn_numpy.py` | Four-node path: normalised adjacency and one GCN layer |
| `gcn_torch.py` | Simple undirected graph, sparse COO propagation, checked invariants |
| `gat_torch.py` | Dense masked single-head graph attention for small graphs |
| `graph_classification.py` | Disjoint-union batching; size-sensitive sum versus mean |
| `link_prediction.py` | Held-out edge removal and negative sampling without duplicates |
| `smoothing_demo.py` | Linear mixing on a regular cycle, without learned weights |
| `graph_data.py`, `depth_experiment.py` | Original synthetic graph and validation-selected depth experiment |
| `depth_diagnosis.py` | Applied solution | Why the eight-layer GCN fails: vanishing signal and gradient |

```bash
.venv/bin/python chapter-18-gnn/gcn_numpy.py
.venv/bin/python chapter-18-gnn/gcn_torch.py
.venv/bin/python chapter-18-gnn/gat_torch.py
.venv/bin/python chapter-18-gnn/graph_classification.py
.venv/bin/python chapter-18-gnn/link_prediction.py
.venv/bin/python chapter-18-gnn/smoothing_demo.py
.venv/bin/python chapter-18-gnn/depth_experiment.py
.venv/bin/python -m pytest chapter-18-gnn --ignore-glob='*/based-on/*'
```

The four-node GCN output is `[1.316497, 2.074915, 2.483163, 2.224745]`.
Dense and sparse calculations agree. Sparse row indices identify receivers,
columns identify senders; the sparse helper symmetrises and deduplicates
unweighted input edges and includes one self-loop per node. NumPy additionally
supports symmetric nonnegative weighted adjacency and adds the identity matrix.

The batching example has constant degree-two node features: sum preserves graph
size whereas mean does not. Its classifier is a fixed size rule, not a learned
benchmark. The link decoder is also untrained: its scores demonstrate the
interface and do not establish predictive accuracy. Missing edges can be
unobserved positives. Always remove held-out positive edges before propagation.

The depth experiment fixes one homophilous synthetic graph and a 32/32/96 split.
Only training labels enter the objective. All permitted features and edges are
visible, so this is transductive evaluation. Seeds 42/43/44 vary initialisation,
not the graph. Mean validation loss selects depth four; test accuracy on this
fixture is 1.0. Eight layers give 0.5. Optimisation effects also matter:
these scores alone do not prove over-smoothing. The independent cycle example
isolates mixing, and a new-graph claim requires multiple graph draws or suitable
held-out real graphs. See `applied/RESULTS.md` and its JSON record.

All graphs and features here are original synthetic examples, covered by this
repository's MIT licence. No dataset or pretrained weights are downloaded.

The guided exercise is in `applied/`: masked loss, validation-based depth
selection and node variation. The three starter functions intentionally fail
until completed; the solution and acceptance tests are in `full-code/`.
