"""Build docs/results.md from runs/: scores, sparsity, and behaviour diagnostics.

  python report.py --runs-root runs --out ../docs/results.md
"""
import argparse
import collections
import json
import pathlib

import numpy as np

from core import MinAtar6, MLPPolicy

GAMES = ["asterix", "breakout", "seaquest"]
# random baselines measured here (code/baseline_random.py); PPO and Astra from the paper
REFERENCE = {
    "breakout": {"ppo100": 15.8, "ppo500": 25.3, "astra": 517.4, "random": 0.49},
    "seaquest": {"ppo100": 5.6, "ppo500": 88.9, "astra": 53.1, "random": 0.10},
    "asterix": {"ppo100": 25.8, "ppo500": 55.2, "astra": 53.2, "random": 0.53},
}


def action_profile(npz, game, steps=500):
    pol = MLPPolicy.from_npz(npz)
    env = MinAtar6(game)
    env.seed(0)
    x = env.reset()
    rng = np.random.default_rng(0)
    cnt = collections.Counter()
    for _ in range(steps):
        a = pol.sample(x, rng)
        cnt[a] += 1
        x, _, term = env.step(a)
        if term:
            x = env.reset()
    tot = sum(cnt.values())
    return {a: round(cnt[a] / tot, 3) for a in range(6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="../docs/results.md")
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.runs_root)
    rows = []
    for g in GAMES:
        for d in sorted((root / g).glob("attempt*")):
            ev = d / "eval.json"
            if not ev.exists():
                continue
            e = json.loads(ev.read_text())
            p = json.loads((d / "parse.json").read_text())
            u = json.loads((d / "usage.json").read_text()) if (d / "usage.json").exists() else {}
            cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
            row = {
                "game": g, "attempt": d.name,
                "statements": p["statements"], "applied": p["applied"],
                "nonzero": p["nonzero_total"],
                "mean": e["mean"], "se": e["se"], "max": e.get("max"),
                "len": e.get("mean_length"),
                "truncated": p.get("truncated", u.get("truncated", False)),
                "reasoning_tokens": u.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens"),
                "completion_tokens": u.get("usage", {}).get("completion_tokens"),
                "wall_s": u.get("wall_s"),
                "cost_usd": (u.get("cost") or {}).get("usd", 0.0),
                "model": cfg.get("model") or u.get("model"),
            }
            if not args.no_profile and (d / "weights.npz").exists():
                row["profile"] = action_profile(d / "weights.npz", g)
            rows.append(row)

    models = sorted({r["model"] for r in rows if r.get("model")})
    model_str = " / ".join(models) if models else "unknown"
    lines = ["# Experiment 5 — measured scores", "",
             f"Model: `{model_str}`; single turn, no tools; actions sampled from softmax, "
             "sticky 0.1, 10,000-step cap; 1,000 episodes per attempt.", "",
             "## Per-attempt results", "",
             "| game | attempt | statements | nonzero weights | score | max | mean length | reasoning tok | USD | status |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        st = "truncated" if r["truncated"] else ("parse errors" if r["applied"] != r["statements"] else "ok")
        cost = f"{r['cost_usd']:.3f}" if r.get("cost_usd") else "—"
        lines.append(f"| {r['game']} | {r['attempt']} | {r['statements']} | {r['nonzero']} | "
                     f"{r['mean']:.1f} ± {r['se']:.1f} | {r['max']:.0f} | {r['len']:.0f} | "
                     f"{r['reasoning_tokens']} | {cost} | {st} |")

    total_cost = sum(r.get("cost_usd", 0) or 0 for r in rows)
    if total_cost:
        lines += ["", f"Total API cost for these runs (list price): **${total_cost:.2f}**", ""]

    lines += ["", "## Against paper / baselines", "",
              "| game | random¹ | PPO 100M | PPO 500M | Astra (paper) | this arm (mean) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for g in GAMES:
        vals = [r["mean"] for r in rows if r["game"] == g]
        mean = f"{np.mean(vals):.1f}" if vals else "—"
        ref = REFERENCE[g]
        lines.append(f"| {g} | {ref['random']} | {ref['ppo100']} | {ref['ppo500']} | "
                     f"{ref['astra']} | {mean} |")

    lines += ["", "¹ Random is measured with this harness (1,000 episodes); "
              "PPO and Astra columns are the paper's numbers, and the USD column is "
              "only filled where a cost estimate was recorded."]

    if not args.no_profile:
        lines += ["", "## Action mix over 500 steps (0 no-op / 1 left / 2 up / 3 right / 4 down / 5 fire)", "",
                  "| game | attempt | action mix |", "| --- | --- | --- |"]
        for r in rows:
            if "profile" in r:
                prof = " ".join(f"{k}:{v}" for k, v in r["profile"].items())
                lines.append(f"| {r['game']} | {r['attempt']} | {prof} |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
