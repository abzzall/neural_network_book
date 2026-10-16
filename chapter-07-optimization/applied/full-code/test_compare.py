"""Tests for the chapter 7 applied exercise. The same tests ship in based-on/."""


def test_four_settings(load_script):
    m = load_script("compare")
    assert set(m.settings()) == {"full-batch GD", "mini-batch SGD", "SGD + momentum", "Adam"}


def test_epochs_to(load_script):
    m = load_script("compare")
    assert m.epochs_to([1.0, 0.5, 0.04, 0.03], target=0.05) == 2
    assert m.epochs_to([1.0, 0.5], target=0.05) is None


def test_momentum_reaches_the_target_first(load_script):
    m = load_script("compare")
    X_train, X_test, y_train, y_test = m.load_split()
    nets = m.train_all(X_train, y_train)
    reached = {name: m.epochs_to(net.losses) for name, net in nets.items()}
    assert reached["full-batch GD"] is None
    assert reached["SGD + momentum"] < reached["mini-batch SGD"]
    assert all(m.accuracy(net, X_test, y_test) > 0.93 for net in nets.values())
