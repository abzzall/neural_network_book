"""Tests for the chapter 9 applied exercise. The same tests ship in based-on/."""

import torch


def test_architecture(load_script):
    m = load_script("port")
    model = m.build()
    assert [type(layer).__name__ for layer in model] == ["Linear", "ReLU", "Dropout", "Linear"]
    assert model(torch.zeros(3, 64)).shape == (3, 10)


def test_same_seed_same_model(load_script):
    m = load_script("port")
    train, val, _ = m.load_splits()
    a, b = m.train(train, seed=0, epochs=3), m.train(train, seed=0, epochs=3)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_accuracy_uses_evaluation_mode(load_script):
    m = load_script("port")
    train, val, _ = m.load_splits()
    model = m.train(train, seed=0, epochs=3)
    assert m.accuracy(model, val) == m.accuracy(model, val)      # dropout is off
    assert not model.training


def test_seed_spread_is_small_and_matches_chapter_8(load_script):
    m = load_script("port")
    train, val, _ = m.load_splits()
    accs = m.seed_study(train, val)
    assert accs.std() < 0.01
    assert abs(accs.mean() - 0.9381) < 3 * accs.std() + 0.005   # chapter 8's nnbook result
