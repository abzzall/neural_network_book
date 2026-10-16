"""Masked supervision, validation-based depth choice and a smoothing diagnostic."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from depth_experiment import run

def masked_loss(logits, labels, training_ids):
    raise NotImplementedError("Complete masked_loss")


def choose_depth(records):
    raise NotImplementedError("Complete choose_depth")


def node_variance(embeddings):
    raise NotImplementedError("Complete node_variance")


if __name__ == '__main__':
    import json
    records = [run(seed, depth, loss_fn=masked_loss) for seed in (42, 43, 44)
               for depth in (0, 1, 2, 4, 8)]
    print(json.dumps(dict(selected_depth=choose_depth(records), runs=records), indent=2))
