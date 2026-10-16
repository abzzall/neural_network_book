# Probe a pretrained sentiment classifier

## Outcome

Evaluate a frozen sentiment model, select a decision threshold using a small
selection set and compare test accuracy before/after. No pretrained weights
are updated. The internal `calibration` split name means threshold selection;
this exercise does not calibrate probabilities.

## Verified versions

2026-10-16: Python 3.14.4, PyTorch 2.14.1+cu132 executing on CPU,
Transformers 5.20.1, huggingface-hub 2.2.0, tokenizers 0.23.3,
safetensors 0.8.0. See the chapter README for installation and the explicit
checkpoint download. Its exact revision and Apache 2.0 licence are recorded
there. Original review sentences are supplied under the repository MIT licence.

## Files supplied

- `reviews.json`: eight selection sentences and twelve distinct test sentences,
  with author-intended labels (1 positive, 0 negative).
- `based-on/sentiment_probe.py`: model loading, score extraction and experiment
  order supplied; threshold selection, accuracy and error reporting incomplete.
- `based-on/test_sentiment_probe.py`: complete acceptance tests.
- `full-code/`: completed solution and a results verifier.

## Steps

1. In `choose_threshold`, declare candidates `[0.2, 0.35, 0.5, 0.65, 0.8]`.
2. Compute accuracy for each candidate on the supplied probabilities and labels.
   A prediction is positive when its positive-class probability is at least the
   candidate. Use NumPy comparison followed by a mean.
3. Return the candidate with highest accuracy. Break ties by distance to 0.5,
   then by smaller threshold. One implementation minimises the tuple
   `(-accuracy, abs(threshold - 0.5), threshold)` over candidate indices.
4. In `accuracy`, return mean correctness as a Python float.
5. In `errors`, return misclassified texts in their original order.
6. Run the tests and completed script. Save its JSON and fill in `RESULTS.md`.

The runner scores the selection sentences and chooses the threshold before
using test outcomes. Do not retune the candidates or intended labels after
seeing the test results. The test set is tiny and hand-written; it is not an
independent sample from a deployment population.

## What to record

Record the exact software and checkpoint revision, selected threshold,
selection accuracy, test accuracy before/after and the returned error list.
Save the script's complete JSON as `results.json` next to this file.

## What to explain

- Why does threshold selection use different labels from final reporting?
- Why does this procedure not fine-tune the model or calibrate its confidence?
- If there are errors, inspect one. If there are none, explain why perfect
  accuracy on twelve author-written examples is weak evidence about real reviews.
- What new data and evaluation would be needed before adapting the model?

## Verification

From this folder, using the repository Python environment:

```bash
../../../.venv/bin/python -m pytest based-on/test_sentiment_probe.py
../../../.venv/bin/python based-on/sentiment_probe.py > results.json
../../../.venv/bin/python full-code/verify_results.py results.json
```

The unfinished starter fails with numbered errors. The completed solution
passes; model integration tests skip explicitly without the downloaded weights.
Download the pinned checkpoint to verify recorded pretrained results.

## Troubleshooting

The missing-model error gives the download command. Routine execution is
local-only, so do not expect an absent checkpoint to download automatically.
Use the checkpoint's tokeniser and label mapping. Keep `top_k=None` so score
extraction can select the positive label rather than assuming a list order.
The model may print weight-loading progress on stderr; JSON results go to
stdout. A verifier skip means weights are absent, not that recorded results
were accepted.
