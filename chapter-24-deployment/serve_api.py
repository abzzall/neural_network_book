"""A prediction service for the ONNX digit model, with input validation.

Run: uvicorn serve_api:app (needs uvicorn); the tests use FastAPI's TestClient.
"""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL = Path(__file__).resolve().parents[1] / 'outputs' / 'digits_cnn.onnx'
app = FastAPI(title='Digit classifier')
Pixel = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=1)]


class Image(BaseModel):
    pixels: list[Pixel] = Field(min_length=64, max_length=64,
                                description='8x8 grey values, row by row, in 0..1')


@lru_cache(maxsize=1)
def loaded_model():
    """Cache one self-contained ONNX snapshot; restart to deploy new weights."""
    weights = MODEL.read_bytes()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(weights, sess_options=options,
                                   providers=['CPUExecutionProvider'])
    return runtime, 'sha256:' + sha256(weights).hexdigest()


@app.get('/health')
def health():
    return {'status': 'ok'}                          # liveness, not model readiness


@app.post('/predict')
def predict(image: Image):
    runtime, version = loaded_model()                 # only after schema validation
    x = np.array(image.pixels, dtype=np.float32)
    logits = runtime.run(None, {'image': x.reshape(1, 1, 8, 8)})[0][0]
    if not np.isfinite(logits).all():
        raise HTTPException(status_code=503, detail='model produced nonfinite scores')
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return {'digit': int(p.argmax()), 'probability': round(float(p.max()), 4),
            'model': version}
