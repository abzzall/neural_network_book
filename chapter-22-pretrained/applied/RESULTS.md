# Chapter 22 applied exercise — results

| Quantity | Value |
|---|---|
| `encoder_parameters` | |
| `full_trainable` | |
| `lora_trainable` | |
| `probe_macro_f1` | |
| `full_macro_f1_mean` | |
| `lora_macro_f1_mean` | |
| `lora_macro_f1_min` | |

Seconds per seed on your machine (not checked): full ____, LoRA ____.

## What to explain

1. LoRA trains about 0.3% of the parameters of full fine-tuning. How close is
   its score, and what does it save if not much time?
2. Why does `B` start at zero? What would go wrong if both `A` and `B` were random?
3. A company serves 50 customer-specific classifiers. Compare storing 50 fully
   fine-tuned copies with storing 50 LoRA adapters.
4. When would you still choose the linear probe?
