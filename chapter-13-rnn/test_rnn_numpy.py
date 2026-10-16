"""Check BPTT against PyTorch and central differences on every parameter."""

import numpy as np
import pytest
import torch
from torch import nn


def test_forward_and_all_gradients(load_script):
    m = load_script('rnn_numpy')
    net = m.RNN(2, 3, 2)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 2, 2))
    h0 = rng.normal(size=(2, 3))
    targets = rng.integers(2, size=(4, 2))
    loss = net.loss(x, targets, h0)
    grads, dx, dh0 = net.backward()
    recurrent, output = nn.RNN(2, 3).double(), nn.Linear(3, 2).double()
    with torch.no_grad():
        recurrent.weight_ih_l0.copy_(torch.from_numpy(net.Wx.T))
        recurrent.weight_hh_l0.copy_(torch.from_numpy(net.Wh.T))
        recurrent.bias_ih_l0.copy_(torch.from_numpy(net.bh))
        recurrent.bias_hh_l0.zero_()
        output.weight.copy_(torch.from_numpy(net.Wy.T))
        output.bias.copy_(torch.from_numpy(net.by))
    xt = torch.tensor(x, requires_grad=True)
    ht = torch.tensor(h0[None], requires_grad=True)
    hidden, _ = recurrent(xt, ht)
    logits = output(hidden)
    expected = nn.functional.cross_entropy(logits.reshape(-1, 2), torch.tensor(targets.ravel()))
    expected.backward()
    assert loss == pytest.approx(expected.item(), abs=1e-12)
    np.testing.assert_allclose(net.forward(x, h0)[1], hidden.detach().numpy(), atol=1e-12)
    references = [recurrent.weight_ih_l0.grad.T, recurrent.weight_hh_l0.grad.T,
                  recurrent.bias_ih_l0.grad, output.weight.grad.T, output.bias.grad]
    for analytic, reference in zip(grads, references):
        np.testing.assert_allclose(analytic, reference.numpy(), atol=1e-12)
    np.testing.assert_allclose(dx, xt.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(dh0, ht.grad.numpy()[0], atol=1e-12)
    for parameter, analytic in zip(net.parameters(), grads):
        for idx in np.ndindex(parameter.shape):
            original = parameter[idx]
            parameter[idx] = original + 1e-6
            upper = net.loss(x, targets, h0)
            parameter[idx] = original - 1e-6
            lower = net.loss(x, targets, h0)
            parameter[idx] = original
            assert analytic[idx] == pytest.approx((upper - lower) / 2e-6, abs=1e-8)


def test_inputs_and_stable_loss(load_script):
    m = load_script('rnn_numpy')
    net = m.RNN(1, 2, 2)
    with pytest.raises(ValueError, match='shape'):
        net.forward(np.zeros((0, 1, 1)))
    with pytest.raises(ValueError, match='h0'):
        net.forward(np.zeros((1, 1, 1)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match='integer'):
        net.loss(np.zeros((1, 1, 1)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match='vocabulary'):
        net.loss(np.zeros((1, 1, 1)), np.array([[2]]))
    net.by[:] = [1000., -1000.]
    assert np.isfinite(net.loss(np.zeros((1, 1, 1)), np.array([[1]])))
