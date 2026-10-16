"""Tests for the chapter 12 applied exercise. The same tests ship in based-on/."""

import torch


def test_shapes_and_sizes(load_script):
    m = load_script("cnn_vs_mlp")
    x = torch.zeros(4, 1, 8, 8)
    assert m.mlp()(x).shape == m.cnn()(x).shape == (4, 10)
    assert m.n_params(m.mlp()) == 2410 and m.n_params(m.cnn()) == 6090


def test_random_shift_moves_a_pixel_by_at_most_one(load_script):
    m = load_script("cnn_vs_mlp")
    x = torch.zeros(1, 1, 8, 8)
    x[0, 0, 4, 4] = 1.0
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        r, c = torch.nonzero(m.random_shift(x, g)[0, 0])[0].tolist()
        assert abs(r - 4) <= 1 and abs(c - 4) <= 1


def test_cnn_is_more_robust_to_shifts(load_script):
    m = load_script("cnn_vs_mlp")
    mlp_plain, mlp_shifted = m.evaluate(m.mlp, m.fit, seeds=(0,))
    cnn_plain, cnn_shifted = m.evaluate(m.cnn, m.fit, seeds=(0,))
    aug_plain, aug_shifted = m.evaluate(m.cnn, m.fit_augmented, seeds=(0,))
    assert cnn_plain > mlp_plain and cnn_shifted > mlp_shifted + 0.1
    assert aug_shifted > cnn_shifted
