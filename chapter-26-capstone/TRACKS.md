# Choose one capstone track

These are project proposals, not completed experiments or licence approvals for
an arbitrary dataset. Confirm source permissions and task-specific evaluation
before acquiring data or using a checkpoint.

| Track | Bounded example | Baseline and neural comparison | Essential split/evidence contract |
| --- | --- | --- | --- |
| Tabular | Classify an equipment event from available measurements | Linear and suitable nonlinear classical model; MLP | Keep repeated equipment/entities together when generalising to new ones; fit preprocessing on training only; record feature availability and per-class metrics |
| Image | Classify objects or defects | Simple image features with a classifier; CNN or adapted encoder | Split by physical object/capture session, then augment training only; record image transforms, checkpoint revision and licence |
| Text | Classify message intent | TF-IDF with a linear classifier; frozen or fine-tuned encoder | Group duplicates and related threads; record label policy, tokeniser, truncation, checkpoint and macro-F1 |
| Time series | Forecast a fixed demand horizon | Naive/seasonal and suitable classical method; neural forecast model | Chronological backtesting; inputs known at each origin; record horizon, multi-step policy and scale-appropriate metrics, plus interval coverage when produced |
| Graph | Classify whole graphs | Graph-summary features with a classifier; message-passing model | Whole-graph splits; train-only fitted transformations; graph/label provenance and declared metric |

A node- or link-prediction extension needs an explicit information-access
protocol: distinguish visible graph structure from held-out labels or edges.
Do not assume that a random node split measures performance on new graphs.

For every track, inspect task-relevant errors, report sample counts, record
versions/seeds/splits and compare methods on the same evaluation units.
Synthetic examples are acceptable for implementation verification when clearly
labelled. Claims about real use need suitable independent evidence.

The supplied binary reference tests are examples of contracts. Preserve their
leakage and provenance principles while adapting target shapes, grouping,
metrics and expected outputs to the chosen project.
