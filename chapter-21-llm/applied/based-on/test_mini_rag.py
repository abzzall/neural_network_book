"""Check chunking, retrieval, prompts, scoring, and the measured result."""

from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / 'chapter-15-nlp'))
import encoder
import lm

MODELS = ((lm.directory('qwen') / 'model.safetensors').is_file()
          and (encoder.CACHE / 'model.safetensors').is_file())


def test_chunks_and_scoring(load_script):
    m = load_script('mini_rag')
    assert m.chunks('first\nline\n\n\nsecond\n\n') == ['first\nline', 'second']
    assert m.correct('It has 5,574 messages.', ['5574'])
    assert m.correct('Apache License 2.0', ['apache']) and not m.correct('MIT', ['apache'])


def test_retrieve_picks_most_similar(load_script):
    m = load_script('mini_rag')
    passages = torch.nn.functional.normalize(torch.eye(4), dim=1)
    queries = torch.nn.functional.normalize(torch.tensor([[0.1, 0.9, 0.0, 0.3]]), dim=1)
    assert m.retrieve(queries, passages, k=2).tolist() == [[1, 3]]


@pytest.mark.skipif(not MODELS, reason='download the qwen model and the chapter 15 encoder')
def test_prompt_and_reference_run(load_script):
    m = load_script('mini_rag')
    marker = load_script('verify_results')
    tokenizer, _ = lm.load('qwen')
    text = m.prompt(tokenizer, 'Q?', ['passage one'])
    assert 'Context:\npassage one' in text and text.rstrip().endswith('assistant')
    results = m.run()
    for name, expected in marker.REFERENCE.items():
        assert marker.matches(name, results[name], expected), name
    assert results['correct_with_retrieval'] > results['correct_without_retrieval']
