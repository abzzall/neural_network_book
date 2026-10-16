"""Acceptance checks for the guided exercise."""
import importlib.util
from pathlib import Path
import pytest
import torch
spec = importlib.util.spec_from_file_location('exercise_under_test', Path(__file__).with_name('graph_exercise.py'))
exercise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exercise)

def test_only_training_nodes_have_supervised_gradient():
    logits = torch.zeros(4, 2, requires_grad=True)
    loss = exercise.masked_loss(logits, torch.tensor([0, 1, 0, 1]), torch.tensor([0, 2]))
    loss.backward()
    assert logits.grad[[1, 3]].abs().sum() == 0
    assert logits.grad[[0, 2]].abs().sum() > 0

def test_selection_ignores_test_accuracy_and_breaks_ties_by_depth():
    records = [dict(depth=1, validation_loss=.4, test_accuracy=1),
               dict(depth=2, validation_loss=.2, test_accuracy=0)]
    assert exercise.choose_depth(records) == 2
    records[0]['validation_loss'] = .2
    assert exercise.choose_depth(records) == 1

def test_variance_detects_equal_embeddings():
    assert exercise.node_variance(torch.ones(4, 3)) == 0
    assert exercise.node_variance(torch.eye(4)) > 0
