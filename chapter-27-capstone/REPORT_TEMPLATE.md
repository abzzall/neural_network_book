# Capstone report

## Task and declared plan

State the unit, target, prediction time, intended use, primary metric, baseline,
candidate budget, selection rule and any changes to the original brief.

## Data and permissions

Give source/version/licence, data fingerprint, collection or generation method,
label policy, measured audit, exclusions and grouping. Explain known limitations
and consequences of errors, including relevant group coverage for human targets.

## Evaluation design

Give split IDs/policy, availability of each feature, fitted preprocessing,
validation procedure, seed handling, final test policy and any planned refit.
Explain why this estimates the intended form of generalisation.

## Baselines and model development

Describe fair comparators, neural architecture, training budget and source
attribution. Record configurations that affected the choice, including failed
experiments. Distinguish supplied infrastructure from your implementation.

## Results

Provide per-run validation metrics and the decision they imply. Then give the
final test result for the locked procedure and relevant baseline. Define spread
and uncertainty measures precisely. Do not substitute a favourable seed or omit
an unsuccessful neural comparison.

For an external-data project, preserve paired predictions and example or forecast
identifiers for every comparator. Account for groups or time dependence in any
uncertainty calculation. Measure end-to-end latency and peak memory on the target
machine if they affect the recommendation; state the batch size, warm-up,
thread count and included preprocessing. Mark unmeasured requirements explicitly.

## Error analysis

Provide traceable examples and slice counts. Distinguish predeclared slices
from exploratory ones. Separate observed facts from possible explanations and
state how an explanation could be checked. If no errors are observed, explain
the evaluation's coverage limits.

## Recommendation and limits

What would you choose under the declared constraints? What evidence supports
that choice? What is unresolved, and which new data or experiment could change
it? Avoid treating teaching-fixture performance as a deployment guarantee.

## Reproduction and attribution

Record source revision and working-tree state, data identity and split record,
Python/package versions, seeds, exact commands and saved artifact locations.
List data/checkpoint/code licences and references. State any run that did not
finish rather than supplying an invented result.
