"""Uniform-random baseline under the blog's protocol.

  python baseline_random.py --episodes 1000 --out ../logs/random_baseline.json

Same environment wrapper, sticky actions (p = 0.1) and 10,000-step cap as the
scored runs; the only difference is that actions are drawn uniformly instead of
from a written network.
"""
import argparse
import json

import numpy as np

from core import evaluate


class UniformRandom:
    def sample(self, x, rng):
        return int(rng.integers(0, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--games", nargs="+", default=["asterix", "breakout", "seaquest"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = {}
    for g in args.games:
        r = evaluate(g, UniformRandom(), episodes=args.episodes, seed0=args.seed0)
        out[g] = {k: r[k] for k in ("episodes", "mean", "se", "max", "mean_length")}
        print(f"{g}: {r['mean']:.3f} ± {r['se']:.3f} over {r['episodes']} episodes "
              f"(mean length {r['mean_length']:.0f})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
