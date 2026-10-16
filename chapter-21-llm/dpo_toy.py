"""Direct preference optimisation on one prompt with five candidate responses.

A hidden reward r ranks the responses. People compare pairs and prefer y_w over
y_l with probability sigmoid(r_w - r_l) (Bradley-Terry). DPO learns a policy from
these preferences alone; theory says it approaches pi* ~ pi_ref exp(r / beta).
"""

import numpy as np
import torch

BETA = 0.5
reward = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])          # hidden from the learner
ref_logits = torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0])         # the reference policy
log_ref = torch.log_softmax(ref_logits, dim=0)


def preferences(n, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(5, size=n), rng.integers(5, size=n)
    a, b = a[a != b], b[a != b]
    p_a = 1 / (1 + np.exp(-(reward[a] - reward[b]).numpy()))
    a_wins = rng.random(len(a)) < p_a
    return torch.tensor(np.where(a_wins, a, b)), torch.tensor(np.where(a_wins, b, a))


def dpo_loss(logits, winners, losers):
    """-log sigmoid(beta * [(log pi - log ref)(y_w) - (log pi - log ref)(y_l)]), eq. (21.7)."""
    ratio = torch.log_softmax(logits, dim=0) - log_ref
    return -torch.nn.functional.logsigmoid(BETA * (ratio[winners] - ratio[losers])).mean()


if __name__ == '__main__':
    winners, losers = preferences(20000)
    logits = ref_logits.clone().requires_grad_(True)            # start at the reference
    optimiser = torch.optim.Adam([logits], lr=0.05)
    for _ in range(2000):
        loss = dpo_loss(logits, winners, losers)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    learned = torch.softmax(logits.detach(), dim=0)
    target = torch.softmax(log_ref + reward / BETA, dim=0)       # pi* of eq. (21.6)
    show = lambda p: ' '.join(f'{v:.3f}' for v in p)
    print(f'preference pairs: {len(winners)}')
    print('reference policy:', show(log_ref.exp()))
    print('DPO policy:      ', show(learned))
    print('theoretical pi*: ', show(target))
    print(f'largest difference from pi*: {(learned - target).abs().max():.4f}')
    implied = BETA * (torch.log(learned) - log_ref)
    print('implied reward minus its mean:', show(implied - implied.mean()))
