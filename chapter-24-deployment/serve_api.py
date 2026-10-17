"""A prediction service for the ONNX digit model, with input validation.

Run: uvicorn serve_api:app (needs uvicorn); the tests use FastAPI's TestClient.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL = Path(__file__).resolve().parents[1] / 'outputs' / 'digits_cnn.onnx'
VERSION = 'digits-cnn-1'
app = FastAPI(title='Digit classifier')
session = None


class Image(BaseModel):
    pixels: list[float] = Field(min_length=64, max_length=64,
                                description='8x8 grey values, row by row, in 0..1')


@app.get('/health')
def health():
    return {'status': 'ok', 'model': VERSION}


@app.post('/predict')
def predict(image: Image):
    global session
    if session is None:                               # load once, on first use
        session = ort.InferenceSession(str(MODEL), providers=['CPUExecutionProvider'])
    x = np.array(image.pixels, dtype=np.float32)
    if x.min() < 0 or x.max() > 1:
        raise HTTPException(status_code=422, detail='pixel values must lie in [0, 1]')
    logits = session.run(None, {'image': x.reshape(1, 1, 8, 8)})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return {'digit': int(p.argmax()), 'probability': round(float(p.max()), 4),
            'model': VERSION}
