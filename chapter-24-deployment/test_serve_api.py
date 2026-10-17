"""The service answers health checks, predicts test digits, and rejects bad input."""

from pathlib import Path
import importlib.util
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
READY = (importlib.util.find_spec('fastapi') is not None
         and importlib.util.find_spec('onnxruntime') is not None
         and importlib.util.find_spec('httpx') is not None
         and (HERE.parent / 'outputs' / 'digits_cnn.onnx').is_file())
pytestmark = pytest.mark.skipif(not READY, reason='optional: requirements-deployment.txt '
                                'and export_onnx.py')


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import serve_api
    return TestClient(serve_api.app)


def test_health(client):
    assert client.get('/health').json() == {'status': 'ok', 'model': 'digits-cnn-1'}


def test_predicts_test_digits(client):
    from digit_model import load_images
    _, _, X_test, y_test = load_images()
    hits = 0
    for x, y in zip(X_test[:50], y_test[:50]):
        reply = client.post('/predict', json={'pixels': x.flatten().tolist()})
        assert reply.status_code == 200 and 0 < reply.json()['probability'] <= 1
        hits += reply.json()['digit'] == int(y)
    assert hits >= 47


def test_rejects_bad_input(client):
    assert client.post('/predict', json={'pixels': [0.5] * 63}).status_code == 422
    assert client.post('/predict', json={'pixels': [2.0] * 64}).status_code == 422
    assert client.post('/predict', json={'image': []}).status_code == 422
