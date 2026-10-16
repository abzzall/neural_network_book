"""Check the label budget, the classifier's shapes, and the measured comparison."""

from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import encoder
import spam_data

DATA = spam_data.PATH.is_file()
MODEL = (encoder.CACHE / 'model.safetensors').is_file()


def test_labelled_subset_is_stratified(load_script):
    m = load_script('spam_compare')
    texts = [f'message {i}' for i in range(1000)]
    labels = [int(i % 8 == 0) for i in range(1000)]
    x, y = m.labelled_subset(texts, labels)
    assert len(x) == 200 and sum(y) == 25 and set(x) <= set(texts)
    assert m.labelled_subset(texts, labels) == (x, y)          # same seed, same sample


@pytest.mark.skipif(not MODEL, reason='run encoder.py --download first')
def test_logits_shape_and_padding_invariance(load_script):
    m = load_script('spam_compare')
    tokenizer, model = encoder.load_encoder()
    head = torch.nn.Linear(model.config.hidden_size, 2)
    with torch.inference_mode():
        alone = m.logits(tokenizer, model, head, ['Win a prize now'])
        padded = m.logits(tokenizer, model, head,
                          ['Win a prize now', 'a much longer message that forces padding'])
    assert padded.shape == (2, 2)
    torch.testing.assert_close(alone[0], padded[0], atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not (DATA and MODEL), reason='download data and model first')
def test_run_matches_reference(load_script):
    m = load_script('spam_compare')
    marker = load_script('verify_results')
    results = m.run()
    for name, expected in marker.REFERENCE.items():
        assert marker.matches(name, results[name], expected), name
    assert results['tfidf_macro_f1'] < results['frozen_macro_f1']
    assert results['frozen_macro_f1'] < results['finetuned_macro_f1_mean']
