"""Acceptance checks for the guided exercise."""
import importlib.util
from pathlib import Path
import pytest
import torch
spec = importlib.util.spec_from_file_location('exercise_under_test', Path(__file__).with_name('vision_exercise.py'))
exercise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exercise)

def test_dice_and_empty_class():
    assert exercise.class_dice(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1, 1])) == pytest.approx([2/3, 4/5])
    assert exercise.class_dice(torch.zeros(3), torch.zeros(3)) == [1., 1.]

def test_loss_has_finite_useful_gradient():
    logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
    masks = torch.ones_like(logits)
    loss = exercise.segmentation_loss(logits, masks)
    loss.backward()
    assert torch.isfinite(loss) and (logits.grad < 0).all()

def test_validation_threshold():
    probabilities = torch.tensor([.1, .1, .9, .9])
    masks = torch.tensor([0, 0, 1, 1])
    assert exercise.choose_threshold(probabilities, masks) == .5
