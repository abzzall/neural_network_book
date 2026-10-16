"""Complete the three segmentation functions, then run the fixed experiment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.nn import functional as F
from segment_shapes import run

def class_dice(actual, predicted):
    raise NotImplementedError("Complete class_dice")


def segmentation_loss(logits, masks):
    raise NotImplementedError("Complete segmentation_loss")


def choose_threshold(probabilities, masks):
    raise NotImplementedError("Complete choose_threshold")


if __name__ == '__main__':
    import json
    print(json.dumps([run(s, loss_fn=segmentation_loss,
                         threshold_fn=choose_threshold, score_fn=class_dice)
                      for s in (42, 43, 44)], indent=2))
