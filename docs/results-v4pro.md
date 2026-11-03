# Experiment 5 — measured scores

Model: `deepseek-v4-pro`; single turn, no tools; actions sampled from softmax, sticky 0.1, 10,000-step cap; 1,000 episodes per attempt.

## Per-attempt results

| game | attempt | statements | nonzero weights | score | max | mean length | reasoning tok | USD | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asterix | attempt1 | 153 | 1189 | 0.5 ± 0.0 | 5 | 97 | 57040 | — | ok |
| asterix | attempt2 | 160 | 618 | 2.9 ± 0.1 | 20 | 139 | 54553 | — | ok |
| asterix | attempt3 | 129 | 628 | 0.9 ± 0.0 | 11 | 96 | 91908 | — | ok |
| breakout | attempt1 | 22 | 184 | 4.5 ± 0.1 | 34 | 53 | 17973 | — | ok |
| breakout | attempt2 | 46 | 311 | 8.4 ± 0.3 | 62 | 98 | 29749 | — | ok |
| breakout | attempt3 | 17 | 107 | 8.3 ± 0.3 | 62 | 97 | 19040 | — | ok |
| seaquest | attempt1 | 395 | 1969 | 0.8 ± 0.0 | 6 | 156 | 66606 | — | ok |
| seaquest | attempt2 | 277 | 2356 | 0.3 ± 0.0 | 3 | 117 | 57532 | — | ok |
| seaquest | attempt3 | 200 | 1662 | 0.6 ± 0.0 | 5 | 61 | 75205 | — | ok |

## Against paper / baselines

| game | random¹ | PPO 100M | PPO 500M | Astra (paper) | this arm (mean) |
| --- | --- | --- | --- | --- | --- |
| asterix | 0.53 | 25.8 | 55.2 | 53.2 | 1.4 |
| breakout | 0.49 | 15.8 | 25.3 | 517.4 | 7.0 |
| seaquest | 0.1 | 5.6 | 88.9 | 53.1 | 0.6 |

¹ Random is measured with this harness (1,000 episodes); PPO and Astra columns are the paper's numbers, and the USD column is only filled where a cost estimate was recorded.

## Action mix over 500 steps (0 no-op / 1 left / 2 up / 3 right / 4 down / 5 fire)

| game | attempt | action mix |
| --- | --- | --- |
| asterix | attempt1 | 0:0.252 1:0.36 2:0.0 3:0.292 4:0.0 5:0.096 |
| asterix | attempt2 | 0:0.0 1:0.54 2:0.112 3:0.29 4:0.058 5:0.0 |
| asterix | attempt3 | 0:0.002 1:0.562 2:0.218 3:0.058 4:0.16 5:0.0 |
| breakout | attempt1 | 0:0.104 1:0.446 2:0.002 3:0.448 4:0.0 5:0.0 |
| breakout | attempt2 | 0:0.098 1:0.452 2:0.0 3:0.45 4:0.0 5:0.0 |
| breakout | attempt3 | 0:0.098 1:0.452 2:0.0 3:0.45 4:0.0 5:0.0 |
| seaquest | attempt1 | 0:0.028 1:0.382 2:0.018 3:0.384 4:0.162 5:0.026 |
| seaquest | attempt2 | 0:0.028 1:0.19 2:0.024 3:0.214 4:0.074 5:0.47 |
| seaquest | attempt3 | 0:0.5 1:0.016 2:0.07 3:0.0 4:0.41 5:0.004 |
