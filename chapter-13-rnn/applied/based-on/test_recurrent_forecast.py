"""Check forecasting chronology, train-only scaling and the held-out comparison."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


def test_windows_and_no_future_leak(load_script):
    m = load_script('recurrent_forecast')
    series = m.make_series()
    x, y, baseline, origins, scale = m.make_windows(series)
    assert x.shape == (392, 28, 1) and y.shape == baseline.shape == (392,)
    np.testing.assert_array_equal(origins, np.arange(28, 420))
    assert scale == pytest.approx(series[:260].std())
    np.testing.assert_allclose(x[0, :, 0], (series[:28] - series[:260].mean()) / scale, atol=1e-7)
    np.testing.assert_allclose(y.numpy(), series[origins], rtol=1e-7)
    np.testing.assert_allclose(baseline.numpy(), series[origins - 7], rtol=1e-7)
    changed = series.copy()
    changed[350:] += 10000
    future_x, _, _, _, future_scale = m.make_windows(changed)
    assert future_scale == scale
    torch.testing.assert_close(x[origins <= 350], future_x[origins <= 350])
    with pytest.raises(ValueError, match='lookback'):
        m.make_windows(series, lookback=2)


def test_test_targets_cannot_change_training(load_script):
    m = load_script('recurrent_forecast')
    torch.set_num_threads(1)
    x, y, baseline, origins, scale = m.make_windows(m.make_series())
    first, epoch = m.fit(x, y, baseline, origins, scale, epochs=3)
    altered = y.clone()
    altered[origins >= 340] = 100000
    second, second_epoch = m.fit(x, altered, baseline, origins, scale, epochs=3)
    assert epoch == second_epoch
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_finished_forecast_script(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('recurrent_forecast.py'))],
                            cwd=tmp_path, capture_output=True, text=True, check=True, timeout=60)
    values = {k: float(v) for k, v in (line.split(': ') for line in result.stdout.splitlines())}
    assert values['train_windows'] == 232
    assert 1 <= values['selected_epoch_seed42'] <= 250
    for name, reference in [('validation_seasonal_mae', 1.76248), ('validation_gru_mae', 0.407699),
                            ('test_seasonal_mae', 1.69621), ('test_gru_mae', 1.08933)]:
        assert values[name] == pytest.approx(reference, rel=0.02)
    assert values['test_gru_mae'] < values['test_seasonal_mae']
