# Chapter 14 — Attention and Transformers

NumPy attention with its backward rules, multi-head layout comparisons, position
encodings, a small PyTorch character transformer and a pinned pretrained
sentiment classifier. Everything executes on CPU.

## Environment

Verified on 2026-10-16 with Python 3.14.4, NumPy 2.5.4, PyTorch 2.14.1+cu132
(CPU execution), Transformers 5.20.1, huggingface-hub 2.2.0, tokenizers 0.23.3
and safetensors 0.8.0. A CUDA installation is not required for this chapter.
The repository's requirements pin the text packages; update the chapter 9
environment from the repository root:

```bash
uv pip install -p .venv -r requirements.txt
```

## Attention and character examples

```bash
.venv/bin/python chapter-14-transformers/attention_demo.py
.venv/bin/python chapter-14-transformers/position_demo.py
.venv/bin/python chapter-14-transformers/character_demo.py
```

The attention demo prints Q, K, V, scaled scores, weights and outputs, then
their causal versions. Its unmasked output is approximately
`[[1.604448, 0.796664], [1.401112, 1.203336], [1.758725, 1.0]]`.
The NumPy backward rules agree with PyTorch and selected central differences;
multi-head output also agrees after copying the projection matrices.
NumPy boolean masks use **true = allowed**; PyTorch MultiheadAttention masks
use **true = forbidden**. The tests check this difference.

The position example confirms permutation equivariance without positions and
the change caused by fixed positional vectors. The character model instead
uses learned positions, a pre-normalised causal block and no dropout.
Its initial/final training cross-entropy is 2.958524 / 0.065127 on the verified
CPU environment; last digits may differ. The 48-character greedy sample begins
`rain falls. sun shines. wind blows.`. This memorises an original repeated
toy corpus under the repository MIT licence; it is not a held-out language test.

## Optional pretrained checkpoint

The model is
[`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english),
pinned at revision `714eb0fa89d2f80546fda750413ed43d93601a13`.
Its [model card at that revision](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blob/714eb0fa89d2f80546fda750413ed43d93601a13/README.md)
declares Apache 2.0 licensing. The classifier was fine-tuned on SST-2; we do
not download that dataset. Our review sentences are original MIT-licensed
examples. Respect the checkpoint's model-card limitations and source licence.

Download explicitly once, then run locally:

```bash
.venv/bin/python chapter-14-transformers/pretrained_probe.py --download
.venv/bin/python chapter-14-transformers/pretrained_probe.py
```

This stores approximately 256 MiB in the ignored `data/` directory, including
the local model card. No weights are committed. Ordinary runs pass
`local_files_only=True`; the tests and repository verifier never download.
Model-dependent tests and the applied verifier explicitly skip if the weights
are absent. A missing model in a direct probe invocation raises an error naming
the download command. Download access can depend on network availability and
Hub limits; do not substitute an unpinned revision silently.

The two printed positive probabilities are approximately 0.999872 for the
positive example and 0.000198 for the negative example. These scores are not
evidence that confidence is calibrated on arbitrary reviews.

## Applied exercise

See [instructions](applied/INSTRUCTIONS.md) and the
[results template](applied/RESULTS.md). The exercise probes the frozen classifier
and selects a threshold using eight sentences before reporting results on
twelve distinct test sentences. It does not fine-tune model parameters.
On this small original set, the selected threshold stays 0.5 and both test
accuracies are 1.0. The empty error list does not establish real-world reliability.

Run `./scripts/check-all.sh` for the full checks. The checkpoint was present
when the printed model-dependent results were verified.
