"""Geometry, optimisation and script-level checks for the vision examples."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def imports(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location('ch17_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boxes_nms_and_average_precision():
    m = load('iou_nms')
    boxes = [[0, 0, 2, 2], [.2, 0, 2.2, 2], [6, 0, 8, 2], [3, 0, 5, 2]]
    scores = [.9, .8, .7, .6]
    np.testing.assert_allclose(m.pairwise_iou(boxes[:1], boxes[1:2]), [[9 / 11]])
    assert m.nms(boxes, scores, .5) == [0, 2, 3]
    assert m.nms(boxes[:2], [1, 1], .5) == [0]
    ap = load('average_precision').average_precision
    result = ap(boxes, scores, [boxes[0], boxes[3]])
    assert result[0] == pytest.approx(.75)
    assert result[1].tolist() == [1, 0, 0, 1]
    keep = m.nms(boxes, scores, .5)
    assert ap(np.array(boxes)[keep], np.array(scores)[keep], [boxes[0], boxes[3]])[0] == pytest.approx(5 / 6)
    assert ap(np.empty((0, 4)), [], [boxes[0]])[0] == 0
    assert ap(boxes, scores, np.empty((0, 4)))[0] == 0
    with pytest.raises(ValueError):
        m.pairwise_iou([[1, 0, 0, 2]], boxes)


def test_unet_pair_augmentation_and_dice_gradients():
    torch.set_num_threads(1)
    model = load('tiny_unet').TinyUNet()
    x = torch.randn(2, 1, 31, 33)
    assert model(x).shape == x.shape
    shapes = load('vision_shapes')
    image, mask = shapes.make_shapes(2, 10)
    flipped_x, flipped_y = shapes.flip_pair(image, mask, True, True)
    torch.testing.assert_close(flipped_x, image.flip([-1, -2]))
    torch.testing.assert_close(flipped_y, mask.flip([-1, -2]))
    metrics = load('segmentation_metrics')
    assert metrics.class_dice(mask, mask) == [1., 1.]
    assert metrics.class_dice(torch.zeros(2), torch.zeros(2)) == [1., 1.]
    logits = torch.randn(2, 1, 3, 3, dtype=torch.float64, requires_grad=True)
    targets = (torch.rand_like(logits) > .5).double()
    assert torch.autograd.gradcheck(lambda z: metrics.segmentation_loss(z, targets), (logits,))


def test_vit_and_contrastive_invariants():
    model = load('tiny_vit').TinyViT()
    x = torch.randn(2, 1, 8, 8)
    assert model(x).shape == (2, 10)
    assert model.positions.shape[1] == 17
    loss = load('contrastive_toy').info_nce
    collapsed = torch.ones(3, 4)
    assert loss(collapsed, collapsed).item() == pytest.approx(np.log(5))
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    torch.testing.assert_close(loss(a, b), loss(b, a))
    with pytest.raises(ValueError):
        loss(a, b, 0)


@pytest.mark.parametrize('name,expected', [
    ('detection_demo', 'AP'), ('unet_shapes', 'output'),
    ('vit_digits', 'test accuracy:'), ('contrastive_toy', 'final contrastive loss:')])
def test_reader_scripts(name, expected, tmp_path):
    result = subprocess.run([sys.executable, str(HERE / (name + '.py'))],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert not result.stderr
    assert expected.lower() in result.stdout.lower()
    if name == 'vit_digits':
        assert float(result.stdout.split('test accuracy:')[1]) > .85


def test_pretrained_detector_local_only(tmp_path):
    if importlib.util.find_spec('torchvision') is None:
        pytest.skip('optional torchvision not installed')
    detector = load('pretrained_detector')
    with pytest.raises(FileNotFoundError, match='--download'):
        detector.run(tmp_path)
    cache = Path(os.environ.get('NNBOOK_VISION_CACHE', str(detector.DEFAULT_CACHE)))
    if not (cache / detector.URL.rsplit('/', 1)[1]).exists():
        pytest.skip('optional checkpoint is not cached')
    result = subprocess.run([sys.executable, str(HERE / 'pretrained_detector.py'),
                             '--cache', str(cache)], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert 'output schema valid: True' in result.stdout


@pytest.mark.parametrize('relative', ['segment_shapes.py', 'applied/full-code/vision_exercise.py'])
def test_segmentation_experiment_as_reader(relative, tmp_path):
    result = subprocess.run([sys.executable, str(HERE / relative)], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    runs = actual['runs'] if isinstance(actual, dict) else actual
    reference = json.loads((HERE / 'applied/full-code/reference_results.json').read_text())
    assert [run['threshold'] for run in runs] == [.6, .3, .5]
    for run, expected in zip(runs, reference):
        np.testing.assert_allclose(run['dice'], expected['dice'], atol=.025, rtol=0)
