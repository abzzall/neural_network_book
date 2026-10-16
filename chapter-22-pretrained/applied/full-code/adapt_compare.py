"""Three ways to adapt a pretrained encoder to spam detection with 200 labels.

Linear probe (frozen encoder + logistic regression), full fine-tuning, and LoRA
on the attention query and value projections. Same data and test set as the
chapter 15 exercise; fine-tuning results are means over seeds 42, 43, 44.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'chapter-15-nlp'))
sys.path.insert(0, str(ROOT / 'chapter-15-nlp' / 'applied' / 'full-code'))
from encoder import load_encoder
from spam_compare import fine_tuned_score, frozen_score, labelled_subset, logits
from spam_data import load, split


class LoRALinear(nn.Module):
    """Step 1: a frozen linear layer plus a trainable rank-r update, eq. (22.2)."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # update starts at 0

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T


def add_lora(model, names=('query', 'value'), r=8, alpha=16):
    """Step 2: freeze every weight, then wrap the named linear layers with LoRA."""
    for p in model.parameters():
        p.requires_grad_(False)
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.split('.')[-1] in names]
    for name, module in targets:
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, LoRALinear(module, r, alpha))
    return len(targets)


def trainable(*modules):
    """Step 3: count parameters that receive gradients."""
    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)


def lora_score(train_x, train_y, test_x, test_y, seed, epochs=3, lr=1e-3):
    """Step 4: train only LoRA weights and the head; return macro-F1 and counts."""
    torch.manual_seed(seed)
    tokenizer, model = load_encoder()
    add_lora(model)
    head = nn.Linear(model.config.hidden_size, 2)
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]
                                  + list(head.parameters()), lr=lr)
    targets, order = torch.tensor(train_y), torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(train_x), generator=order).split(16):
            loss = nn.functional.cross_entropy(
                logits(tokenizer, model, head, [train_x[i] for i in idx]), targets[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()
    with torch.inference_mode():
        predicted = torch.cat([logits(tokenizer, model, head, test_x[i:i + 64]).argmax(1)
                               for i in range(0, len(test_x), 64)])
    return f1_score(test_y, predicted.numpy(), average='macro'), trainable(model, head)


def run(seeds=(42, 43, 44)):
    torch.set_num_threads(4)
    train_x, test_x, train_y, test_y = split(*load())
    sub_x, sub_y = labelled_subset(train_x, train_y)
    _, encoder = load_encoder()
    out = {'encoder_parameters': sum(p.numel() for p in encoder.parameters()),
           'probe_macro_f1': frozen_score(sub_x, sub_y, test_x, test_y)}
    for name, fit in (('full', lambda s: (fine_tuned_score(sub_x, sub_y, test_x, test_y, s),)),
                      ('lora', lambda s: lora_score(sub_x, sub_y, test_x, test_y, s))):
        start, rows = time.perf_counter(), [fit(s) for s in seeds]
        scores = [r[0] for r in rows]
        out[f'{name}_macro_f1_mean'] = float(np.mean(scores))
        out[f'{name}_macro_f1_min'] = float(np.min(scores))
        out[f'{name}_seconds_per_seed'] = (time.perf_counter() - start) / len(seeds)
        if name == 'lora':
            out['lora_trainable'] = rows[0][1]
    out['full_trainable'] = out['encoder_parameters'] + 384 * 2 + 2
    return out


if __name__ == '__main__':
    print(json.dumps(run(), indent=1))
