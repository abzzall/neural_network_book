"""Tests for the chapter 8 applied exercise. The same tests ship in based-on/."""

import numpy as np


def test_build_options(load_script):
    m = load_script("regularise")
    names = [type(layer).__name__ for layer in m.build(dropout=0.5, batchnorm=True).layers]
    assert names == ["Dense", "BatchNorm", "ReLU", "Dropout", "Dense"]


def test_every_setting_overfits_the_training_set(load_script):
    m = load_script("regularise")
    train, val, test = m.load_splits()
    assert len(train[1]) == 200
    nets = m.train_all(train, val)
    assert set(nets) == {"baseline", "L2", "dropout", "early stopping", "batch norm"}
    assert all(m.accuracy(net, train) == 1.0 for net in nets.values())
    assert len(nets["early stopping"].losses) < 200


def test_choice_uses_validation_accuracy(load_script):
    m = load_script("regularise")
    train, val, _ = m.load_splits()
    nets = m.train_all(train, val)
    best = m.choose(nets, val)
    assert m.accuracy(nets[best], val) == max(m.accuracy(n, val) for n in nets.values())
