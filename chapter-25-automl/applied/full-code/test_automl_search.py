"""Exhaustive search and successive halving on the digits, with an untouched test set."""

import pytest


@pytest.fixture(scope='module')
def search(request):
    import importlib.util
    path = request.path.parent / 'automl_search.py'
    spec = importlib.util.spec_from_file_location('automl_search', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidates_cover_the_grid(search):
    cands = search.candidates()
    assert len(cands) == 10 and len(set(cands)) == 10
    assert ((128, 64, 32), 1e-4) in cands


def test_halving_spends_less_on_a_toy_problem(search, monkeypatch):
    calls = []

    def fake_scores(cands, epochs, data):
        calls.append((len(cands), epochs))
        return [ord(c) / 1000 for c in cands]                 # later letters score higher

    monkeypatch.setattr(search, 'validation_scores', fake_scores)
    best, score, cost = search.successive_halving(list('abcdefghij'), None)
    assert calls == [(10, 25), (5, 50), (2, 100), (1, 200)]
    assert cost == 900 and best == 'j'
    best, score, cost = search.exhaustive(list('abcdefghij'), None)
    assert best == 'j' and cost == 2000


def test_final_test_is_used_only_by_the_selected_procedures(search, monkeypatch):
    seen = []
    monkeypatch.setattr(search, 'final_test', lambda c, d: seen.append(c) or 0.0)
    monkeypatch.setattr(search, 'validation_scores',
                        lambda cands, e, d: [float(i) for i in range(len(cands))])
    search.run()
    assert len(seen) == 2


def test_measured_results():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).with_name('automl_search.py')
    spec = importlib.util.spec_from_file_location('automl_search_real', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.run()
    assert out['exhaustive_epochs'] == 2000 and out['halving_epochs'] == 900
    assert 0 < out['exhaustive_actual_epochs'] <= out['exhaustive_epochs']
    assert 0 < out['halving_actual_epochs'] <= out['halving_epochs']
    assert out['exhaustive_fit_seconds'] > 0 and out['halving_fit_seconds'] > 0
    assert out['exhaustive_shape'] == out['halving_shape'] == [128, 64, 32]
    assert abs(out['exhaustive_validation'] - 0.9639) <= 0.02
    assert abs(out['exhaustive_test'] - 0.975) <= 0.02
