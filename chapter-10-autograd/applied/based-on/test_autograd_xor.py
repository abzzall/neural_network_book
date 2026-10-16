"""Check the exercise steps, all first-step gradients and learned XOR outputs."""

import numpy as np
import pytest


def test_hidden_and_prediction(load_script):
    m = load_script('autograd_xor')
    p = [m.Value(0.) for _ in range(17)]
    assert [h.data for h in m.hidden_values(p, 1., -1.)] == [0.] * 4
    assert m.predict(p, 1., -1.).data == 0.5
    assert m.objective(p).data == 0.25
    p[2].data, p[12].data = 2., 1.
    assert m.hidden_values(p, 0., 0.)[0].data == 2.
    assert m.predict(p, 0., 0.).data == pytest.approx(1 / (1 + np.exp(-2.)))


def test_clear_then_update(load_script):
    m = load_script('autograd_xor')
    p = [m.Value(2.)]
    p[0].grad = 3.
    m.update(p, 0.5)
    assert p[0].data == 0.5
    m.clear_gradients(p)
    assert p[0].grad == 0.


def test_training_reduces_loss_and_matches_torch(load_script):
    m = load_script('autograd_xor')
    first, final, difference, probabilities = m.run()
    assert first == pytest.approx(0.329236, abs=1e-6)
    assert final < 0.001
    assert difference < 1e-12
    np.testing.assert_array_equal(np.array(probabilities) >= 0.5, [0, 1, 1, 0])
    assert np.mean((np.array(probabilities) - [0, 1, 1, 0]) ** 2) == pytest.approx(final)
