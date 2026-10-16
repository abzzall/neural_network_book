# Chapter 19 applied exercise — results

| Quantity | Value |
|---|---|
| `final_mean` | |
| `final_min` | |
| `checkpoints_at_500` | |
| `worst_after_40k` | |

Run time per seed on your machine (not checked): ____ s.

## What to explain

1. All three seeds end at 500, yet one scored about 14 at 80,000 steps. What
   does that say about stopping training at a fixed step count?
2. Why does DQN need a replay buffer and a target network, when tabular
   Q-learning on the grid world needed neither?
3. Why must the time limit not be treated as a termination in the TD target?
4. REINFORCE used about 210,000-230,000 steps; DQN used 100,000. Which is more
   sample-efficient here, and what does REINFORCE gain in exchange?
