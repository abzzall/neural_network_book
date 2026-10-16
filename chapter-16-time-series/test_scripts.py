"""Run the chapter's scripts as a reader would; data-dependent ones skip without data."""

from pathlib import Path
import importlib.util
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bike_data
import pretrained_forecast

DATA = bike_data.PATH.is_file()
CHRONOS = importlib.util.find_spec('chronos') is not None
MODEL = (pretrained_forecast.CACHE / 'model.safetensors').is_file()
needs_data = pytest.mark.skipif(not DATA, reason='run bike_data.py --download first')


def run(script, cwd):
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=cwd,
                            capture_output=True, text=True, check=True, timeout=300)
    return result.stdout.splitlines()


def test_mase_example(tmp_path):
    assert run('mase_example.py', tmp_path) == [
        'seasonal naive: forecast [11, 15, 20, 13], MAE 1.50, MASE 1.20',
        'naive: forecast [13, 13, 13, 13], MAE 3.25, MASE 2.60',
        'scale (in-sample seasonal-naive MAE): 1.25']


def test_missing_data_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match='--download'):
        bike_data.load(tmp_path / 'absent.csv')
    with pytest.raises(FileNotFoundError, match='--download'):
        pretrained_forecast.load_model(directory=tmp_path)


@needs_data
def test_data_and_autocorrelation(tmp_path):
    assert run('bike_data.py', tmp_path) == [
        'days: 731, from 2011-01-01 to 2012-12-31',
        'mean rentals 2011: 3405.8, 2012: 5599.9',
        'mean by weekday (Mon..Sun): 4338 4511 4549 4667 4690 4551 4229',
        'lowest day: 2012-10-29 with 22 rentals']
    assert run('components.py', tmp_path) == [
        'rentals            r1=+0.846 r2=+0.779 r7=+0.739 r14=+0.707 r28=+0.651',
        'first difference   r1=-0.291 r2=-0.109 r7=-0.010 r14=+0.075 r28=+0.013',
        'weekly difference  r1=+0.361 r2=+0.160 r7=-0.454 r14=+0.019 r28=+0.068',
        'approximate 95% band for white noise: +-0.072']


@needs_data
def test_baselines_linear_and_strategies(tmp_path):
    assert run('backtest.py', tmp_path) == [
        'origins: 25, horizon: 14 days',
        'naive           MAE  1385.8  MASE 1.585',
        'seasonal naive  MAE  1114.6  MASE 1.272',
        '7-day mean      MAE  1055.4  MASE 1.205',
        'Holt-Winters    MAE  1055.9  MASE 1.204']
    assert run('linear_direct.py', tmp_path) == [
        'linear, past values only MAE  1010.0  MASE 1.154',
        'linear, with calendar    MAE   974.0  MASE 1.114']
    assert run('strategies.py', tmp_path) == [
        'recursive MAE day 1: 1172.7  day 14: 1199.4  all days: 1009.7',
        'direct    MAE day 1: 1166.6  day 14: 1199.0  all days: 1010.0']


@needs_data
def test_tcn_backtest(tmp_path):
    lines = run('tcn_backtest.py', tmp_path)
    assert lines[0] == 'training windows: 478, receptive field: 63 days'
    mean = float(lines[-1].split('MASE ')[1].split()[0])
    assert mean == pytest.approx(1.082, abs=0.03)
    assert all(1.0 < float(line.split('MASE ')[1]) < 1.2 for line in lines[1:4])


def test_origins_are_sundays_from_july_2012(load_script):
    if not DATA:
        pytest.skip('run bike_data.py --download first')
    backtest = load_script('backtest')
    table = bike_data.load()
    starts = backtest.origins(len(table))
    assert str(table.index[starts[0]].date()) == '2012-07-01'
    assert {table.index[t].day_name() for t in starts} == {'Sunday'}


@pytest.mark.skipif(not (DATA and CHRONOS and MODEL),
                    reason='optional: install chronos-forecasting and download the model')
def test_pretrained_forecast(tmp_path):
    lines = run('pretrained_forecast.py', tmp_path)
    lines = [line for line in lines if not line.startswith('Loading')]
    assert float(lines[0].split('MASE ')[1]) == pytest.approx(1.135, abs=0.02)
    assert float(lines[1].split(': ')[1]) == pytest.approx(0.757, abs=0.03)


@needs_data
def test_figure_data(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'figure_data.py'), str(tmp_path)],
                           capture_output=True, text=True, check=True, timeout=120).stdout
    assert lines.startswith('first origin, day 1 quantiles: ')
    import numpy as np
    series = np.loadtxt(tmp_path / 'series.dat', skiprows=1)
    acf = np.loadtxt(tmp_path / 'acf.dat', skiprows=1)
    fan = np.loadtxt(tmp_path / 'fan.dat', skiprows=1)
    assert series.shape == (731, 3) and acf.shape == (29, 3) and fan.shape == (42, 5)
    assert acf[0, 1] == 1 and acf[1, 1] == pytest.approx(0.846, abs=1e-3)
    q = fan[28:, 2:]
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()
