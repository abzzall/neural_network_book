"""LoRA arithmetic, distillation, and CLIP zero-shot (skipped without the model)."""

from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clip_model

CLIP = (clip_model.CACHE / 'model.safetensors').is_file()


def run(script, cwd):
    return subprocess.run([sys.executable, str(HERE / script)], cwd=cwd, capture_output=True,
                          text=True, check=True, timeout=600).stdout.splitlines()


def test_lora_numpy(tmp_path):
    assert run('lora_numpy.py', tmp_path) == [
        'full fine-tuning: 147456 trainable weights in this layer',
        'LoRA rank 8: 6144 trainable weights (4.17% of full)',
        'output unchanged at the start: True',
        'merged layer gives the same output: True',
        'rank of the update: 8']


def test_distillation_loss_is_zero_for_identical_logits(load_script):
    d = load_script('distillation')
    z = torch.randn(5, 10)
    assert d.distillation_loss(z, z).item() == pytest.approx(0.0, abs=1e-6)
    assert d.distillation_loss(z, z + torch.randn(5, 10)).item() > 0


def test_distillation_script(tmp_path):
    lines = run('distillation.py', tmp_path)
    assert lines[0] == 'parameters: teacher 19210, student 1210'
    for line in lines[1:]:
        acc = {k: float(v) for k, v in re.findall(r'([a-z0-9 ,]+?) (\d\.\d{4})(?:, |$)',
                                                   line.split(': ', 1)[1])}
        assert acc['distilled, no labels'] > acc['100 labels'] + 0.03
        assert acc['distilled, no labels'] >= acc['teacher'] - 0.02


def test_missing_clip_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match='--download'):
        clip_model.load_clip(directory=tmp_path)


@pytest.mark.skipif(not CLIP, reason='run clip_model.py --download first')
def test_clip_zero_shot(tmp_path):
    lines = run('clip_zero_shot.py', tmp_path)
    first = lines[0]
    assert 'colour 1.000' in first
    assert float(first.split('both ')[1].split(',')[0]) == pytest.approx(0.889, abs=0.03)
    assert lines[1] == '  20 x red square -> red triangle'
