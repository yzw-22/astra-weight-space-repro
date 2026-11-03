# Figures

Two figures that carry the argument of the reproduction, rendered from the run
artifacts in this repository. The script only reads `runs/*/*/attempt*/eval.json`
and `logs/random_baseline.json`; it writes into `figures/out/`.

```bash
python figures/make_figures.py                  # from the repository root
python figures/make_figures.py --repo . --out /tmp/figs
```

## `fig1_scores.png` — what the models score, and where the score comes from

* **A** scores per game on a log axis. Per game: the paper's two PPO baselines
  (grey band), the random policy measured here (grey dot), the paper's single
  Astra sample (hollow diamond) and the three models averaged over 3 attempts
  with ±1 SE. Log scale is mandatory: the spread is 0.1 → 517.
* **B** the same 27 attempts as score versus mean episode length. The dashed
  lines are constant reward per step (0.01 / 0.03 / 0.10). All nine Breakout
  attempts fall between 0.05 and 0.085, so on that game the horizontal spread —
  surviving longer — is what separates the models, not the per-step rate.

`fig1_scores.png` is the hero figure used in the top-level README and in
`docs/REPORT.md` §3.1.

## `fig2_depth_variance.png` — the mechanism and the noise (report only)

* **C** Breakout survival by look-ahead depth: brick-bounce-aware prediction
  (Astra) → wall-only projection (best Flash attempt) → tracking the current
  ball column (other attempts, all of V4-Pro). The multipliers are ≈ ×7.2 and
  ×6.3 per extra layer.
* **D** the three attempts behind every point, against the paper's single
  sample. This is the honesty panel: Astra's Breakout range (406–537) brackets
  the paper's 517.4, while Flash's spans 12.6–69.8.

## Style

Helvetica Neue, no top/right spines, light x-grid only, one colour per model
(teal / amber / violet) reused across every panel, direct labels instead of
legends wherever possible. Output is 300 dpi PNG plus SVG (text stays editable);
both are drawn from the same code path, so they never drift apart.

Type is sized for reading at 100 % zoom on a normal screen: 11.5 pt base,
10 pt ticks, 10.5 pt value labels, nothing below 9 pt, and all annotation grey
darkened to `#4B5563` so it survives compression on GitHub and on social
previews. The figures are 13.6 x 5.9 in and 12.6 x 5.6 in.
