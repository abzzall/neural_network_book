"""Train VAEs on digits; report test ELBO terms and which digits their samples look like."""

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from vae import loss_terms, sample, train

torch.set_num_threads(4)
images, labels = load_digits(return_X_y=True)
images = images / 16.0
train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.25,
                                                    random_state=0, stratify=labels)
judge = LogisticRegression(max_iter=5000).fit(train_x, train_y)    # a digit classifier
print(f'classifier accuracy on real test digits: {judge.score(test_x, test_y):.3f}')
print('latent  seed  test -ELBO  test KL  rarest digit among 1000 samples (count)')
for latent in (2, 8):
    for seed in (0, 1, 2):
        model = train(train_x, latent=latent, seed=seed)
        torch.manual_seed(1)                              # fixed sampling noise
        with torch.inference_mode():
            reconstruction, kl = loss_terms(model, torch.tensor(test_x, dtype=torch.float32))
        counts = np.bincount(judge.predict(sample(model, 1000)), minlength=10)
        print(f'{latent:6d}  {seed:4d}  {reconstruction + kl:10.2f}  {kl:7.2f}  '
              f'{counts.argmin()} ({counts.min()})')
