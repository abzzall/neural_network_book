# Chapter 14 validation

Verified on 2026-10-16 with Python 3.14.4, NumPy 2.5.4 and PyTorch
2.14.1+cu132 executing on CPU. The pretrained examples use Transformers
5.20.1, huggingface-hub 2.2.0, tokenizers 0.23.3 and safetensors 0.8.0.
Exact declared dependencies are in the repository requirements.

## Numerical checks

- NumPy attention outputs and all Q/K/V derivatives match PyTorch for masked
  and unmasked cases with distinct query and key lengths. Selected derivatives
  also match independent central differences.
- Bias-free multi-head output matches PyTorch after copying all projection
  matrices. Shape errors and fully forbidden query rows are rejected.
- Position encodings handle even and odd feature widths. Unmasked attention
  is permutation-equivariant; adding fixed positions changes that behaviour.
- Changing future input tokens leaves the causal model's earlier logits exactly
  unchanged, with zero gradient through the distinct future token embeddings.
- Character-model training cross-entropy falls from 2.958524 to 0.065127.
  This fits an original repeated text; no language-generalisation claim is made.

## Pretrained model

Model: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
Revision: `714eb0fa89d2f80546fda750413ed43d93601a13`.
The revision's model card declares Apache 2.0 licensing. Original probe
sentences and the toy character corpus are released under the repository MIT
licence. No external labelled dataset or pretrained weights are committed.

The explicit download fetched the checkpoint into the ignored data directory.
The local classifier printed positive probabilities 0.999872 and 0.000198 for
the chapter's positive and negative examples. The applied probe selected 0.5,
with selection accuracy 1.0 and test accuracy 1.0 both before and after
selection; its test error list is empty. The eight selection sentences and
twelve test sentences were not revised to manufacture errors or an improvement.
These results do not establish probability calibration or representative
review performance.

Offline integration was also executed:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q chapter-14-transformers/test_pretrained.py
```

Both tests passed using local files. Without the checkpoint, model-dependent
tests explicitly skip and direct script execution explains how to download it.
Routine tests and the results verifier do not initiate network access.

## Applied starter and script checks

All demonstrations are run as subprocesses from outside the repository.
The unfinished starter produces three failures and one passing dataset test
with the checkpoint present. Completing the numbered functions reproduces the
full solution exactly. An independent temporary copy passes its three
non-model acceptance tests and explicitly skips its one integration test at
the copy's absent cache path. The original full solution's model-dependent
acceptance test was executed with the real checkpoint present.

The results verifier accepts a freshly recorded JSON and rejects a corrupted
test accuracy. The main chapter tests passed all 16 tests with weights present.
