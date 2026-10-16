"""Check teaching engines against independent PyTorch derivatives and failures."""

import numpy as np
import pytest
import torch


def test_scalar_operations(load_script):
    Value = load_script('value').Value
    for data in (-1.2, 0.3, 1.8):
        x = Value(data)
        t = torch.tensor(data, dtype=torch.double, requires_grad=True)
        y = ((x * x + 2 * x).exp() / 3 + x.relu()) ** 2
        z = ((t * t + 2 * t).exp() / 3 + t.relu()) ** 2
        y.backward()
        z.backward()
        assert y.data == pytest.approx(z.item())
        assert x.grad == pytest.approx(t.grad.item())
        y.backward()
        assert x.grad == pytest.approx(2 * t.grad.item())


def test_scalar_leaf_accumulates(load_script):
    x = load_script('value').Value(2)
    x.backward()
    x.backward()
    assert x.grad == 2


def test_broadcast_and_matrix(load_script):
    m = load_script('tiny_tensor')
    a = np.arange(6.).reshape(2, 3) / 5
    b = np.arange(6.).reshape(3, 2) / 7
    x, w, bias = m.Tensor(a, True), m.Tensor(b, True), m.Tensor([1., -1.], True)
    xt, wt, bt = [torch.tensor(v, requires_grad=True) for v in (a, b, np.array([1., -1.]))]
    y, yt = m.relu(x @ w + bias), torch.relu(xt @ wt + bt)
    y.sum().backward()
    yt.sum().backward()
    np.testing.assert_allclose(y.data, yt.detach().numpy())
    for u, v in ((x, xt), (w, wt), (bias, bt)):
        np.testing.assert_allclose(u.grad, v.grad.numpy())


def test_hooks_retention_and_accumulation(load_script):
    m = load_script('tiny_tensor')
    x, xt = m.Tensor([2., 3.], True), torch.tensor([2., 3.], dtype=torch.double, requires_grad=True)
    y, yt = x * x, xt * xt
    assert x.is_leaf and not y.is_leaf and y.grad_fn is not None
    y.retain_grad()
    yt.retain_grad()
    h, ht = y.register_hook(lambda g: g * 2), yt.register_hook(lambda g: g * 2)
    for retain in (True, False):
        y.backward(np.array([1., 2.]), retain_graph=retain)
        yt.backward(torch.tensor([1., 2.], dtype=torch.double), retain_graph=retain)
    np.testing.assert_allclose(x.grad, xt.grad.numpy())
    np.testing.assert_allclose(y.grad, yt.grad.numpy())
    h.remove()
    ht.remove()
    with pytest.raises(RuntimeError, match='second time'):
        y.backward(np.ones(2))
    with pytest.raises(RuntimeError, match='second time'):
        yt.backward(torch.ones(2))


@pytest.mark.parametrize('backend', ['tiny', 'torch'])
def test_error_contracts(load_script, backend):
    m = load_script('tiny_tensor') if backend == 'tiny' else torch
    make = (lambda v, grad=True: m.Tensor(v, grad)) if backend == 'tiny' else (
        lambda v, grad=True: torch.tensor(v, dtype=torch.double, requires_grad=grad))
    x = make([2., 3.])
    with pytest.raises(RuntimeError, match='scalar outputs'):
        (x * x).backward()
    with pytest.raises(RuntimeError, match='[Ss]hape|[Mm]ismatch'):
        (x * x).backward(np.ones(3) if backend == 'tiny' else torch.ones(3))
    with pytest.raises(RuntimeError, match='leaf Variable'):
        x.add_(1)
    with pytest.raises(RuntimeError, match='requires_grad|requires grad'):
        make(2., False).retain_grad()
    with pytest.raises(RuntimeError, match='require.*grad'):
        make(2., False).register_hook(lambda g: g)
    with m.no_grad():
        with m.no_grad():
            assert not (x * x).requires_grad
        assert not (x + x).requires_grad
    assert (x * x).requires_grad
    y = x * x
    detached = x.detach()
    detached.add_(1)
    assert not detached.requires_grad and detached.is_leaf
    with pytest.raises(RuntimeError, match='inplace operation'):
        y.sum().backward()
    with pytest.raises(RuntimeError, match='does not require grad'):
        detached.sum().backward()


def test_relu_saves_output_not_input(load_script):
    m = load_script('tiny_tensor')
    x = m.Tensor([-1., 2.], True)
    y = m.relu(x)
    x.detach().add_(5)
    y.sum().backward()
    np.testing.assert_array_equal(x.grad, [0., 1.])
    y = m.relu(x)
    y.detach().add_(1)
    with pytest.raises(RuntimeError, match='inplace operation'):
        y.sum().backward()


def test_constant_multiplier_does_not_save_input(load_script):
    m = load_script('tiny_tensor')
    x = m.Tensor(2., True)
    y = x * 3
    x.detach().add_(1)
    y.backward()
    assert x.grad == 3
