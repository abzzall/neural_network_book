"""Export the CNN to ONNX, check it in ONNX Runtime, and quantise it to int8."""

from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
import torch

from digit_model import accuracy, load_images, trained_model

OUT = Path(__file__).resolve().parents[1] / 'outputs'


def export(model, path):
    """Trace the model with torch.export and write an ONNX file with a dynamic batch."""
    example = torch.zeros(1, 1, 8, 8)
    program = torch.onnx.export(model, (example,), dynamo=True, input_names=['image'],
                                dynamic_shapes=({0: 'batch'},), output_names=['logits'])
    program.save(path)


def onnx_logits(path, images):
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    return session.run(None, {'image': images.numpy()})[0]


if __name__ == '__main__':
    torch.set_num_threads(1)
    model = trained_model()
    _, _, X_test, y_test = load_images()
    with torch.inference_mode():
        reference = model(X_test).numpy()
    fp32, int8 = str(OUT / 'digits_cnn.onnx'), str(OUT / 'digits_cnn_int8.onnx')
    prepared = str(OUT / 'digits_cnn_prepared.onnx')
    export(model, fp32)
    quant_pre_process(fp32, prepared)                 # shape inference and graph clean-up
    quantize_dynamic(prepared, int8, weight_type=QuantType.QInt8)
    print(f'PyTorch test accuracy: {accuracy(model, X_test, y_test):.4f}')
    for name, path in (('ONNX float32', fp32), ('ONNX int8', int8)):
        logits = onnx_logits(path, X_test)
        acc = (logits.argmax(1) == y_test.numpy()).mean()
        print(f'{name}: {Path(path).stat().st_size} bytes, accuracy {acc:.4f}, '
              f'max |logit difference| {np.abs(logits - reference).max():.2e}')
    session = ort.InferenceSession(fp32, providers=['CPUExecutionProvider'])
    for batch in (1, 64):
        x = X_test[:batch].numpy()
        start = time.perf_counter()
        for _ in range(200):
            session.run(None, {'image': x})
        per_call = (time.perf_counter() - start) / 200
        print(f'batch {batch:2d}: {per_call * 1e3:.3f} ms per call, '
              f'{batch / per_call:,.0f} images per second')
