"""Tests for the chapter 1 applied exercise. The same tests ship in based-on/."""

import numpy as np


def test_weighted_sums_of_the_first_day(load_script):
    m = load_script("neuron")
    assert np.isclose(m.weighted_sums()[0], 0.65)


def test_decision_flips_exactly_at_the_flip_bias(load_script):
    m = load_script("neuron")
    flips = m.flip_bias()
    for day, bias in enumerate(flips):
        assert m.decisions(bias + 1e-9)[day] == 1
        assert m.decisions(bias - 1e-9)[day] == 0


def test_more_bias_never_means_fewer_irrigation_days(load_script):
    m = load_script("neuron")
    counts = [m.irrigation_days(b) for b in (-1.0, 0.0, 0.5, 1.0, 1.5, 3.0)]
    assert counts == sorted(counts)
