# Experiment 5 — measured scores

Model: `deepseek-flash`; single turn, no tools; actions sampled from softmax, sticky 0.1, 10,000-step cap; 1,000 episodes per attempt.

## Per-attempt results

| game | attempt | statements | nonzero weights | score | max | mean length | reasoning tok | USD | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asterix | attempt1 | 142 | 540 | 7.1 ± 0.2 | 43 | 287 | 69857 | — | ok |
| asterix | attempt2 | 217 | 612 | 3.3 ± 0.1 | 22 | 120 | 71139 | — | ok |
| asterix | attempt3 | 267 | 876 | 8.4 ± 0.3 | 55 | 252 | 94102 | — | ok |
| breakout | attempt1 | 154 | 775 | 12.6 ± 0.5 | 111 | 165 | 79389 | — | ok |
| breakout | attempt2 | 108 | 933 | 69.8 ± 2.3 | 525 | 1057 | 95833 | — | ok |
| breakout | attempt3 | 221 | 1559 | 13.9 ± 0.5 | 116 | 168 | 75857 | — | ok |
| seaquest | attempt1 | 382 | 3429 | 0.6 ± 0.0 | 5 | 70 | 108074 | — | ok |
| seaquest | attempt2 | 160 | 1355 | 0.0 ± 0.0 | 0 | 546 | 116127 | — | ok |
| seaquest | attempt3 | 309 | 2963 | 0.1 ± 0.0 | 4 | 4847 | 81353 | — | ok |

## Against paper / baselines

| game | random¹ | PPO 100M | PPO 500M | Astra (paper) | this arm (mean) |
| --- | --- | --- | --- | --- | --- |
| asterix | 0.53 | 25.8 | 55.2 | 53.2 | 6.2 |
| breakout | 0.49 | 15.8 | 25.3 | 517.4 | 32.1 |
| seaquest | 0.1 | 5.6 | 88.9 | 53.1 | 0.3 |

¹ Random is measured with this harness (1,000 episodes); PPO and Astra columns are the paper's numbers, and the USD column is only filled where a cost estimate was recorded.

## Action mix over 500 steps (0 no-op / 1 left / 2 up / 3 right / 4 down / 5 fire)

| game | attempt | action mix |
| --- | --- | --- |
| asterix | attempt1 | 0:0.906 1:0.004 2:0.04 3:0.004 4:0.046 5:0.0 |
| asterix | attempt2 | 0:0.722 1:0.058 2:0.09 3:0.058 4:0.072 5:0.0 |
| asterix | attempt3 | 0:0.766 1:0.072 2:0.046 3:0.062 4:0.054 5:0.0 |
| breakout | attempt1 | 0:0.16 1:0.418 2:0.0 3:0.422 4:0.0 5:0.0 |
| breakout | attempt2 | 0:0.028 1:0.446 2:0.028 3:0.444 4:0.032 5:0.022 |
| breakout | attempt3 | 0:0.174 1:0.406 2:0.0 3:0.42 4:0.0 5:0.0 |
| seaquest | attempt1 | 0:0.046 1:0.072 2:0.204 3:0.136 4:0.416 5:0.126 |
| seaquest | attempt2 | 0:0.0 1:0.0 2:0.992 3:0.0 4:0.008 5:0.0 |
| seaquest | attempt3 | 0:0.304 1:0.29 2:0.004 3:0.342 4:0.022 5:0.038 |
