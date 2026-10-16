"""INCOMPLETE exercise starter. Retrieval-augmented generation over facts about this book's examples.

A small instruction-tuned model answers 20 questions twice: from the question
alone, and with the two most similar corpus paragraphs in the prompt. Retrieval
uses the sentence encoder of chapter 15; generation is greedy, so runs repeat.
"""

import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                     # chapter 21
sys.path.insert(0, str(HERE.parents[2] / 'chapter-15-nlp'))
from encoder import embed, load_encoder
from lm import load

TOP_K = 2


def chunks(text):
    """Step 1: one passage per paragraph."""
    raise NotImplementedError("Step 1: complete chunks")


def retrieve(query_vectors, passage_vectors, k=TOP_K):
    """Step 2: indices of the k passages with the highest cosine similarity."""
    raise NotImplementedError("Step 2: complete retrieve")


def prompt(tokenizer, question, context=None):
    """Step 3: a chat prompt, with retrieved passages when given."""
    raise NotImplementedError("Step 3: complete prompt")


def correct(answer, accepted):
    """Step 4: the answer counts if it contains an accepted string."""
    raise NotImplementedError("Step 4: complete correct")


@torch.inference_mode()
def generate(tokenizer, model, text, max_new_tokens=40):
    ids = tokenizer(text, return_tensors='pt')
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def run():
    torch.set_num_threads(4)
    passages = chunks((HERE.parent / 'corpus.md').read_text())
    questions = json.loads((HERE.parent / 'questions.json').read_text())
    enc_tok, enc = load_encoder()
    p_vec = embed(enc_tok, enc, passages)
    q_vec = embed(enc_tok, enc, [q['q'] for q in questions])
    hits = retrieve(q_vec, p_vec)
    tokenizer, model = load('qwen')
    closed = rag = found = 0
    for i, item in enumerate(questions):
        context = [passages[j] for j in hits[i].tolist()]
        found += any(correct(c, item['a']) for c in context)     # answer was retrieved
        closed += correct(generate(tokenizer, model, prompt(tokenizer, item['q'])), item['a'])
        rag += correct(generate(tokenizer, model, prompt(tokenizer, item['q'], context)),
                       item['a'])
    return {'passages': len(passages), 'questions': len(questions),
            'retrieval_hits': found, 'correct_without_retrieval': closed,
            'correct_with_retrieval': rag}


if __name__ == '__main__':
    print(json.dumps(run(), indent=1))
