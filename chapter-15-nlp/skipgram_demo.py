"""Train skip-gram embeddings on a generated corpus and list nearest words."""

import numpy as np

from sgns import noise_distribution, pairs, sgd_step

rng = np.random.default_rng(42)
animals, people = ['cat', 'dog', 'horse', 'cow'], ['man', 'woman', 'boy', 'girl']
animal_verbs, people_verbs = ['runs', 'sleeps', 'eats'], ['reads', 'writes', 'speaks']
places = {'runs': 'field', 'sleeps': 'barn', 'eats': 'grass',
          'reads': 'book', 'writes': 'letter', 'speaks': 'school'}
text = []
for _ in range(1500):
    if rng.random() < 0.5:
        subject, verb = rng.choice(animals), rng.choice(animal_verbs)
    else:
        subject, verb = rng.choice(people), rng.choice(people_verbs)
    text.append(['the', subject, verb, 'near', 'the', places[verb]])

words = sorted({w for sentence in text for w in sentence})
index = {w: i for i, w in enumerate(words)}
sentences = [np.array([index[w] for w in s]) for s in text]
noise = noise_distribution(sentences, len(words))

dim, k, lr = 10, 5, 0.05
V = rng.normal(0, 0.1, (len(words), dim))        # centre-word vectors (the embeddings)
U = np.zeros((len(words), dim))                  # context-word vectors
all_pairs = np.array(list(pairs(sentences, window=2)))
for epoch in range(5):
    total = 0.0
    for center, context in all_pairs[rng.permutation(len(all_pairs))]:
        negatives = rng.choice(len(words), size=k, p=noise)
        total += sgd_step(V, U, center, context, negatives, lr)
    print(f'epoch {epoch + 1}: mean loss {total / len(all_pairs):.4f}')

E = V / np.linalg.norm(V, axis=1, keepdims=True)
for word in ('cat', 'woman'):
    sims = E @ E[index[word]]
    nearest = [words[j] for j in np.argsort(-sims) if j != index[word]][:3]
    print(f'nearest to {word}: {", ".join(nearest)}')
