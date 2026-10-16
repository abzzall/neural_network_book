"""Search by meaning: TF-IDF matches words, a pretrained encoder matches ideas."""

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from encoder import embed, load_encoder

documents = [
    'How do I reset my forgotten password?',
    'The battery of my phone drains very quickly.',
    'Where can I see the status of my delivery?',
    'I was charged twice for the same order.',
    'The app crashes when I open the camera.',
]
queries = [
    'I cannot remember my login code',
    'my handset loses charge fast',
    'when will my parcel arrive',
    'you took money from my card two times',
]

tfidf = TfidfVectorizer().fit(documents + queries)
lexical = (tfidf.transform(queries) @ tfidf.transform(documents).T).toarray()
tokenizer, model = load_encoder()
semantic = (embed(tokenizer, model, queries) @ embed(tokenizer, model, documents).T)

for i, query in enumerate(queries):
    print(f'query: {query}')
    for name, scores in (('tf-idf', torch.as_tensor(lexical[i])), ('encoder', semantic[i])):
        best = int(scores.argmax())
        print(f'  {name:8s} -> doc {best + 1} (score {scores[best].item():.3f})')
correct = torch.arange(len(queries))
print('tf-idf correct:', int((torch.as_tensor(lexical).argmax(1) == correct).sum()), 'of 4')
print('encoder correct:', int((semantic.argmax(1) == correct).sum()), 'of 4')
