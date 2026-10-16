"""Masked supervision, validation-based depth choice and a smoothing diagnostic."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from depth_experiment import run

def masked_loss(logits, labels, training_ids):
    return torch.nn.functional.cross_entropy(logits[training_ids], labels[training_ids])


def choose_depth(records):
    depths = sorted({record['depth'] for record in records})
    def criterion(depth):
        losses = [r['validation_loss'] for r in records if r['depth'] == depth]
        return sum(losses) / len(losses), depth
    return min(depths, key=criterion)


def node_variance(embeddings):
    return ((embeddings - embeddings.mean(0)) ** 2).mean()


if __name__ == '__main__':
    import json
    records = [run(seed, depth, loss_fn=masked_loss) for seed in (42, 43, 44)
               for depth in (0, 1, 2, 4, 8)]
    print(json.dumps(dict(selected_depth=choose_depth(records), runs=records), indent=2))
