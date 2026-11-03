# Experiment 5 — measured scores

Model: `openai/gpt-6-astra`; single turn, no tools; actions sampled from softmax, sticky 0.1, 10,000-step cap; 1,000 episodes per attempt.

## Per-attempt results

| game | attempt | statements | nonzero weights | score | max | mean length | reasoning tok | USD | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asterix | attempt1 | 110 | 1716 | 69.6 ± 1.3 | 268 | 1000 | 18646 | 1.151 | ok |
| asterix | attempt2 | 132 | 942 | 41.0 ± 1.0 | 213 | 1130 | 21520 | 1.437 | parse errors |
| asterix | attempt3 | 170 | 794 | 64.0 ± 1.2 | 310 | 985 | 15890 | 1.040 | ok |
| breakout | attempt1 | 244 | 4983 | 537.3 ± 9.5 | 842 | 6588 | 22228 | 1.473 | ok |
| breakout | attempt2 | 159 | 2421 | 406.4 ± 5.4 | 637 | 7840 | 19486 | 1.166 | ok |
| breakout | attempt3 | 190 | 1866 | 468.8 ± 5.1 | 662 | 8460 | 21453 | 1.411 | ok |
| seaquest | attempt1 | 135 | 548 | 34.4 ± 1.0 | 128 | 1171 | 23627 | 1.440 | ok |
| seaquest | attempt2 | 181 | 1269 | 14.1 ± 0.5 | 83 | 425 | 22066 | 1.448 | ok |
| seaquest | attempt3 | 177 | 1747 | 38.2 ± 1.0 | 152 | 1078 | 29776 | 1.719 | ok |

Total API cost for these runs (list price): **$12.28**


## Against paper / baselines

| game | random¹ | PPO 100M | PPO 500M | Astra (paper) | this arm (mean) |
| --- | --- | --- | --- | --- | --- |
| asterix | 0.53 | 25.8 | 55.2 | 53.2 | 58.2 |
| breakout | 0.49 | 15.8 | 25.3 | 517.4 | 470.8 |
| seaquest | 0.1 | 5.6 | 88.9 | 53.1 | 28.9 |

¹ Random is measured with this harness (1,000 episodes); PPO and Astra columns are the paper's numbers, and the USD column is only filled where a cost estimate was recorded.

## Action mix over 500 steps (0 no-op / 1 left / 2 up / 3 right / 4 down / 5 fire)

| game | attempt | action mix |
| --- | --- | --- |
| asterix | attempt1 | 0:0.88 1:0.0 2:0.062 3:0.0 4:0.058 5:0.0 |
| asterix | attempt2 | 0:0.988 1:0.0 2:0.006 3:0.0 4:0.006 5:0.0 |
| asterix | attempt3 | 0:0.884 1:0.0 2:0.058 3:0.0 4:0.058 5:0.0 |
| breakout | attempt1 | 0:0.58 1:0.21 2:0.0 3:0.21 4:0.0 5:0.0 |
| breakout | attempt2 | 0:0.484 1:0.268 2:0.0 3:0.248 4:0.0 5:0.0 |
| breakout | attempt3 | 0:0.48 1:0.264 2:0.0 3:0.256 4:0.0 5:0.0 |
| seaquest | attempt1 | 0:0.0 1:0.0 2:0.074 3:0.0 4:0.082 5:0.844 |
| seaquest | attempt2 | 0:0.0 1:0.038 2:0.134 3:0.03 4:0.142 5:0.656 |
| seaquest | attempt3 | 0:0.0 1:0.008 2:0.1 3:0.002 4:0.112 5:0.778 |
