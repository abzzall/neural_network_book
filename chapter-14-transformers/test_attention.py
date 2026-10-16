"""Check attention values, backward rules, head layouts and causal behaviour."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.mark.parametrize('masked', [False, True])
def test_attention_and_derivatives(load_script, masked):
    m = load_script('attention_numpy')
    rng = np.random.default_rng(7)
    arrays = [rng.normal(size=s) for s in [(2, 2, 3, 4), (2, 2, 5, 4), (2, 2, 5, 2)]]
    allowed = np.arange(5)[None, :] <= np.arange(3)[:, None] if masked else None
    out, cache = m.attention(*arrays, allowed)
    arriving = rng.normal(size=out.shape)
    actual = m.backward(cache, arriving)
    tensors = [torch.tensor(a, requires_grad=True) for a in arrays]
    expected = F.scaled_dot_product_attention(*tensors,
        attn_mask=None if allowed is None else torch.tensor(allowed), dropout_p=0.)
    gradients = torch.autograd.grad(expected, tensors, torch.tensor(arriving))
    np.testing.assert_allclose(out, expected.detach(), rtol=1e-12, atol=1e-12)
    for a, b in zip(actual, gradients):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)
    # An independent central difference checks one entry in each input.
    for i, a in enumerate(arrays):
        index = (0, 0, 0, 0)
        original = a[index]
        values = []
        for change in (1e-6, -1e-6):
            a[index] = original + change
            values.append((m.attention(*arrays, allowed)[0] * arriving).sum())
        a[index] = original
        assert actual[i][index] == pytest.approx((values[0] - values[1]) / 2e-6, abs=1e-8)


def test_shapes_and_empty_mask(load_script):
    m = load_script('attention_numpy')
    x = np.ones((3, 2))
    with pytest.raises(ValueError, match='allowed key'):
        m.attention(x, x, x, np.zeros((3, 3), dtype=bool))
    with pytest.raises(ValueError, match='incompatible'):
        m.attention(x, np.ones((3, 4)), x)
    with pytest.raises(ValueError, match='divisible'):
        m.multihead(x[None], *([np.eye(2)] * 4), heads=3)


def test_multihead_layout_and_mask(load_script):
    m = load_script('attention_numpy')
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 4, 6))
    weights = [rng.normal(size=(6, 6)) for _ in range(4)]
    allowed = np.tri(4, dtype=bool)
    actual = m.multihead(x, *weights, heads=2, allowed=allowed)
    model = torch.nn.MultiheadAttention(6, 2, bias=False, batch_first=True).double()
    with torch.no_grad():
        model.in_proj_weight.copy_(torch.tensor(np.concatenate(weights[:3], axis=1).T))
        model.out_proj.weight.copy_(torch.tensor(weights[3].T))
    tensor = torch.tensor(x)
    expected, _ = model(tensor, tensor, tensor, attn_mask=torch.tensor(~allowed))
    np.testing.assert_allclose(actual, expected.detach(), rtol=1e-12, atol=1e-12)


def test_positions_and_permutations(load_script):
    m = load_script('position_demo')
    for width in (1, 4, 5):
        positions = m.sinusoidal(6, width)
        assert positions.shape == (6, width)
        np.testing.assert_array_equal(positions[0, ::2], 0.)
        np.testing.assert_array_equal(positions[0, 1::2], 1.)
        assert not np.allclose(positions[0], positions[1])
    x = np.random.default_rng(42).normal(size=(3, 4))
    p = [2, 0, 1]
    a, _ = m.attention(x, x, x)
    b, _ = m.attention(x[p], x[p], x[p])
    np.testing.assert_allclose(b, a[p], atol=1e-14)
    pos = m.sinusoidal(3, 4)
    a, _ = m.attention(x + pos, x + pos, x + pos)
    b, _ = m.attention(x[p] + pos, x[p] + pos, x[p] + pos)
    assert not np.allclose(b, a[p])


def test_no_future_leakage(load_script):
    m = load_script('tiny_transformer')
    torch.manual_seed(42)
    model = m.CharacterTransformer(10).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    changed = ids.clone()
    changed[:, 3:] = torch.tensor([8, 9])
    torch.testing.assert_close(model(ids)[:, :3], model(changed)[:, :3], rtol=0, atol=0)
    # Earlier logits cannot differentiate through future token embeddings.
    model.zero_grad(set_to_none=True)
    model(ids)[0, 1].sum().backward()
    assert torch.count_nonzero(model.tokens.weight.grad[3:6]) == 0
