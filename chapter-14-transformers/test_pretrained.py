"""Optional model integration runs locally; routine tests never download weights."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pretrained_probe

WEIGHTS = (pretrained_probe.CACHE / 'model.safetensors').is_file()


def test_missing_model_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match='--download'):
        pretrained_probe.load_classifier(directory=tmp_path)


@pytest.mark.skipif(not WEIGHTS, reason='download pinned model first')
def test_pipeline_and_scripts(tmp_path):
    classifier = pretrained_probe.load_classifier()
    probabilities = pretrained_probe.positive_scores(classifier,
        ['The film was wonderful.', 'The film was dreadful.'])
    assert probabilities[0] > 0.9 and probabilities[1] < 0.1
    tokenizer = classifier.tokenizer
    encoded = tokenizer(['A short film.', 'An unforgettable performance.'],
                        padding=True, return_tensors='pt')
    assert encoded['input_ids'].shape == encoded['attention_mask'].shape
    assert (encoded['attention_mask'] == 0).any()
    assert tokenizer.unk_token_id not in tokenizer.encode('An unforgettable film.')
    for path in [HERE / 'pretrained_probe.py',
                 HERE / 'applied/full-code/sentiment_probe.py']:
        result = subprocess.run([sys.executable, str(path)], cwd=tmp_path,
            capture_output=True, text=True, check=True, timeout=90)
        if path.name == 'pretrained_probe.py':
            probabilities = [float(line.split(': ')[1].split(' |')[0])
                             for line in result.stdout.splitlines()]
            assert probabilities[0] > 0.9 and probabilities[1] < 0.1
        else:
            assert 0 <= json.loads(result.stdout)['test_accuracy_after'] <= 1
