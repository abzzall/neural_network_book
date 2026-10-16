# Chapter 22 applied exercise — linear probe, full fine-tuning, or LoRA?

## 1. Outcome

Adapt the pretrained sentence encoder of chapter 15 to spam detection with the
same 200 labelled messages in three ways, and compare macro-F1 on the same
deduplicated test set with the number of trainable parameters and the training
time: a linear probe on frozen embeddings, full fine-tuning, and LoRA on the
attention query and value projections.

## 2. Verified versions and data

Python 3.14.4, PyTorch 2.14.1 (CPU), transformers 5.20.1, scikit-learn 1.9.1,
17 October 2026. About 2 minutes. Download the data and the encoder first:

```bash
.venv/bin/python chapter-15-nlp/spam_data.py --download
.venv/bin/python chapter-15-nlp/encoder.py --download
```

## 3. Steps

1. `LoRALinear`: keep the base layer; create `A` (r x in, small random) and `B`
   (out x r, zeros); the output is the base output plus `alpha / r` times
   `x A^T B^T`.
2. `add_lora`: freeze every parameter of the model, then replace each linear
   layer whose name ends with `query` or `value` by a `LoRALinear` around it.
3. `trainable`: count the parameters that require gradients.
4. In `lora_score`: add LoRA to the encoder, create the two-class head, and
   build AdamW over the trainable encoder parameters and the head.
5. Run the tests and the script; fill `RESULTS.md`.

```bash
.venv/bin/python -m pytest chapter-22-pretrained/applied/based-on
.venv/bin/python chapter-22-pretrained/applied/based-on/adapt_compare.py
```

## 4. Verification

```bash
.venv/bin/python chapter-22-pretrained/applied/full-code/verify_results.py chapter-22-pretrained/applied/RESULTS.md
```

Counts must match exactly; scores may differ by 0.015 across platforms.
