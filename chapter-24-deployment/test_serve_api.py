"""The service answers health checks, predicts test digits, and rejects bad input."""

from pathlib import Path
from hashlib import sha256
import importlib.util
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
READY = all(importlib.util.find_spec(name) is not None
            for name in ('fastapi', 'onnxruntime', 'httpx'))
pytestmark = pytest.mark.skipif(not READY, reason='optional: requirements-deployment.txt '
                                'is not installed')


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import serve_api
    serve_api.loaded_model.cache_clear()
    with TestClient(serve_api.app) as client:
        yield client
    serve_api.loaded_model.cache_clear()


def test_health(client):
    assert client.get('/health').json() == {'status': 'ok'}


@pytest.mark.skipif(not (HERE.parent / 'outputs' / 'digits_cnn.onnx').is_file(),
                    reason='optional: run export_onnx.py first')
def test_predicts_test_digits(client):
    from digit_model import load_images
    _, _, X_test, y_test = load_images()
    hits = 0
    version = 'sha256:' + sha256((HERE.parent / 'outputs' / 'digits_cnn.onnx').read_bytes()).hexdigest()
    for x, y in zip(X_test[:50], y_test[:50]):
        reply = client.post('/predict', json={'pixels': x.flatten().tolist()})
        assert reply.status_code == 200 and 0 < reply.json()['probability'] <= 1
        assert reply.json()['model'] == version
        hits += reply.json()['digit'] == int(y)
    assert hits >= 47


def test_rejects_bad_input(client):
    assert client.post('/predict', json={'pixels': [0.5] * 63}).status_code == 422
    assert client.post('/predict', json={'pixels': [2.0] * 64}).status_code == 422
    assert client.post('/predict', json={'image': []}).status_code == 422


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity', '0.5', True,
                                  None, -0.1, 1.1])
def test_rejects_invalid_pixels_before_loading_model(client, monkeypatch, value):
    import serve_api
    def forbidden_load():
        pytest.fail('invalid input must be rejected before model loading')
    monkeypatch.setattr(serve_api, 'loaded_model', forbidden_load)
    reply = client.post('/predict', json={'pixels': [value] + [0] * 63})
    assert reply.status_code == 422


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_schema_rejects_nonfinite_numbers(value):
    from pydantic import ValidationError
    from serve_api import Image
    with pytest.raises(ValidationError):
        Image(pixels=[value] + [0.0] * 63)


def test_nonfinite_model_output_returns_a_json_error(client, monkeypatch):
    import serve_api
    class BadRuntime:
        def run(self, *args):
            return [np.full((1, 10), np.nan, dtype=np.float32)]
    monkeypatch.setattr(serve_api, 'loaded_model', lambda: (BadRuntime(), 'test'))
    reply = client.post('/predict', json={'pixels': [0] * 64})
    assert reply.status_code == 503
    assert reply.json() == {'detail': 'model produced nonfinite scores'}


def test_version_identifies_loaded_snapshot_not_mutable_path(tmp_path, monkeypatch):
    import serve_api
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'first weights')
    monkeypatch.setattr(serve_api, 'MODEL', path)
    seen = []
    def fake_session(weights, **kwargs):
        seen.append(weights)
        path.write_bytes(b'replacement weights')
        return object()
    monkeypatch.setattr(serve_api.ort, 'InferenceSession', fake_session)
    serve_api.loaded_model.cache_clear()
    try:
        first = serve_api.loaded_model()
        assert first[1] == 'sha256:' + sha256(b'first weights').hexdigest()
        assert serve_api.loaded_model() == first
        assert seen == [b'first weights']
        serve_api.loaded_model.cache_clear()
        second = serve_api.loaded_model()
        assert second[1] == 'sha256:' + sha256(b'replacement weights').hexdigest()
        assert second[1] != first[1]
    finally:
        serve_api.loaded_model.cache_clear()
