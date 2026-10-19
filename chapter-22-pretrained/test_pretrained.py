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
import nf4_numpy
import transfer_eurosat

CLIP = (clip_model.CACHE / 'model.safetensors').is_file()
try:
    import torchvision  # noqa: F401  optional: chapter-22-pretrained/requirements.txt
    EUROSAT = (transfer_eurosat.EUROSAT / 'EuroSAT_RGB.zip').is_file()
except ImportError:
    EUROSAT = False


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


def test_nf4_levels_and_zero_block():
    assert len(nf4_numpy.NF4) == 16
    assert nf4_numpy.NF4[0] == pytest.approx(-1)
    assert nf4_numpy.NF4[-1] == pytest.approx(1)
    assert 0 in nf4_numpy.NF4
    values = torch.zeros(67).numpy()
    codes, scales, size = nf4_numpy.quantise(values, nf4_numpy.NF4)
    restored = nf4_numpy.dequantise(codes, scales, nf4_numpy.NF4, values.shape, size)
    assert restored.shape == values.shape
    assert restored.tolist() == values.tolist()


def test_nf4_beats_uniform_for_normal_weights():
    rng = __import__('numpy').random.default_rng(0)
    values = rng.normal(size=4099).astype('float32')
    errors = []
    for levels in (nf4_numpy.UNIFORM, nf4_numpy.NF4):
        codes, scales, size = nf4_numpy.quantise(values, levels)
        restored = nf4_numpy.dequantise(codes, scales, levels, values.shape, size)
        errors.append(((restored - values) ** 2).mean())
    assert errors[1] < errors[0]


def test_eurosat_split_is_disjoint_and_reproducible(tmp_path):
    for name in ('a', 'b'):
        folder = tmp_path / name
        folder.mkdir()
        for index in range(7):
            (folder / f'{index}.jpg').touch()
    first = transfer_eurosat.stratified_files(tmp_path, 2, 3, seed=4)
    second = transfer_eurosat.stratified_files(tmp_path, 2, 3, seed=4)
    assert first == second
    classes, train, test = first
    assert classes == ['a', 'b']
    assert len(train) == 4 and len(test) == 6
    assert set(train).isdisjoint(test)


def test_eurosat_archive_rejects_path_traversal(tmp_path):
    import zipfile
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('../outside.txt', 'bad')
    with pytest.raises(ValueError, match='unsafe archive member'):
        transfer_eurosat.safe_extract(archive, tmp_path / 'destination')


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


@pytest.mark.skipif(not EUROSAT, reason='install torchvision and run transfer_eurosat.py --download first')
def test_eurosat_transfer_gain(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'transfer_eurosat.py'), '--device', 'cpu'],
                           cwd=tmp_path, capture_output=True, text=True, check=True,
                           timeout=1800).stdout.splitlines()
    assert lines[0] == 'classes: 10, train: 200, test: 1000'
    random_acc, pretrained_acc = float(lines[1].split()[-1]), float(lines[2].split()[-1])
    assert abs(random_acc - 0.5430) <= 0.02 and abs(pretrained_acc - 0.8500) <= 0.02
    assert pretrained_acc - random_acc > 0.25
