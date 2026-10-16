"""Check the LoRA layer, its insertion, parameter counting, and the measured comparison."""

from pathlib import Path
import sys

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'chapter-15-nlp'))
import encoder
import spam_data

READY = spam_data.PATH.is_file() and (encoder.CACHE / 'model.safetensors').is_file()


def test_lora_starts_as_the_base_layer(load_script):
    m = load_script('adapt_compare')
    base = nn.Linear(6, 4)
    layer = m.LoRALinear(base, r=2, alpha=4)
    x = torch.randn(3, 6)
    torch.testing.assert_close(layer(x), base(x))
    assert layer.A.shape == (2, 6) and layer.B.shape == (4, 2)
    with torch.no_grad():
        layer.B.fill_(1.0)
    expected = base(x) + 2.0 * (x @ layer.A.T) @ layer.B.T
    torch.testing.assert_close(layer(x), expected)


def test_add_lora_freezes_and_wraps(load_script):
    m = load_script('adapt_compare')
    model = nn.Sequential()
    model.add_module('block', nn.Module())
    model.block.query, model.block.value, model.block.other = (nn.Linear(4, 4) for _ in range(3))
    assert m.add_lora(model, r=2) == 2
    assert isinstance(model.block.query, m.LoRALinear) and isinstance(model.block.other, nn.Linear)
    assert m.trainable(model) == 2 * (2 * 4 + 4 * 2)          # only A and B of two layers


@pytest.mark.skipif(not READY, reason='download the SMS data and the chapter 15 encoder')
def test_minilm_counts_and_reference_run(load_script):
    m = load_script('adapt_compare')
    marker = load_script('verify_results')
    _, model = encoder.load_encoder()
    assert m.add_lora(model) == 12                              # 6 layers x query, value
    results = m.run()
    for name, expected in marker.REFERENCE.items():
        assert marker.matches(name, results[name], expected), name
