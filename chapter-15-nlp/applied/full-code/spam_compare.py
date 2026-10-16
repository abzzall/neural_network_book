"""Spam detection with 200 labels: TF-IDF vs frozen encoder vs fine-tuned encoder.

All three methods learn from the same 200 labelled messages and are scored on
the same deduplicated test set (macro-F1). Fine-tuning is repeated for three seeds.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from encoder import embed, load_encoder, mean_pool
from spam_data import load, split
from tfidf_baseline import baseline

LABELS, SEEDS, EPOCHS, BATCH, LEARNING_RATE = 200, (42, 43, 44), 3, 16, 5e-5


def labelled_subset(texts, labels, size=LABELS, seed=42):
    """Step 1: a stratified sample of `size` training messages."""
    subset_x, _, subset_y, _ = train_test_split(texts, labels, train_size=size,
                                                stratify=labels, random_state=seed)
    return subset_x, subset_y


def tfidf_score(train_x, train_y, test_x, test_y):
    predicted = baseline().fit(train_x, train_y).predict(test_x)
    return f1_score(test_y, predicted, average='macro')


def frozen_score(train_x, train_y, test_x, test_y):
    """Step 2: embed with the unchanged encoder, then fit logistic regression."""
    tokenizer, model = load_encoder()
    features = embed(tokenizer, model, train_x).numpy()
    classifier = LogisticRegression(class_weight='balanced', max_iter=2000)
    classifier.fit(features, train_y)
    predicted = classifier.predict(embed(tokenizer, model, test_x).numpy())
    return f1_score(test_y, predicted, average='macro')


def logits(tokenizer, model, head, texts):
    """Step 3: tokenise, encode, mean-pool, and apply the classification head."""
    batch = tokenizer(texts, padding=True, truncation=True, max_length=64,
                      return_tensors='pt')
    return head(mean_pool(model(**batch).last_hidden_state, batch['attention_mask']))


def fine_tuned_score(train_x, train_y, test_x, test_y, seed):
    torch.manual_seed(seed)
    tokenizer, model = load_encoder()
    head = torch.nn.Linear(model.config.hidden_size, 2)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()],
                                  lr=LEARNING_RATE)
    targets, order = torch.tensor(train_y), torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(train_x), generator=order).split(BATCH):
            # Step 4: cross-entropy on the batch, then one optimiser step.
            loss = torch.nn.functional.cross_entropy(
                logits(tokenizer, model, head, [train_x[i] for i in idx]), targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():                  # Step 5: predict the test set
        predicted = torch.cat([logits(tokenizer, model, head, test_x[i:i + 64]).argmax(1)
                               for i in range(0, len(test_x), 64)])
    return f1_score(test_y, predicted.numpy(), average='macro')


def run():
    torch.set_num_threads(4)
    train_x, test_x, train_y, test_y = split(*load())
    sub_x, sub_y = labelled_subset(train_x, train_y)
    tuned = [fine_tuned_score(sub_x, sub_y, test_x, test_y, s) for s in SEEDS]
    return {'labelled_messages': len(sub_y), 'labelled_spam': int(sum(sub_y)),
            'test_messages': len(test_y),
            'tfidf_macro_f1': tfidf_score(sub_x, sub_y, test_x, test_y),
            'frozen_macro_f1': frozen_score(sub_x, sub_y, test_x, test_y),
            'finetuned_macro_f1_mean': float(np.mean(tuned)),
            'finetuned_macro_f1_std': float(np.std(tuned))}


if __name__ == '__main__':
    print(json.dumps(run(), indent=1))
