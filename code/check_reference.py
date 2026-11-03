"""Sanity gate: does this harness reproduce the released exp5 artifacts exactly?

Checks
  1. prompt files are byte-identical to the prompts in the released logs
  2. replaying each released reply through the edits DSL reproduces the
     released weights in ref/exp5/networks/*.npz (max |diff| must be 0)
  3. (optional) scores the released weights under the blog's protocol

  python check_reference.py --episodes 100
"""
import argparse
import json
import pathlib

import numpy as np

import edits_dsl as dsl
from core import MLPPolicy, evaluate

HERE = pathlib.Path(__file__).resolve().parent
REF = HERE.parent / "ref" / "exp5"
GAMES = ["asterix", "breakout", "seaquest"]
EXPECTED_CHARS = {"breakout": 9973, "seaquest": 18266, "asterix": 10930}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0,
                    help="0 = skip scoring, >0 = episodes per game")
    ap.add_argument("--out", default=str(HERE / "out" / "reference_check.json"))
    args = ap.parse_args()

    report = {"prompts": {}, "weights": {}, "scores": {}}
    ok = True

    for g in GAMES:
        prompt = (HERE / "prompts" / f"{g}.md").read_bytes()
        released = (REF / f"{g}_message.txt").read_bytes()
        report["prompts"][g] = {"chars": len(prompt),
                                "expected": EXPECTED_CHARS[g],
                                "match": prompt == released,
                                "byte_identical_to_released": prompt == released}
        ok &= report["prompts"][g]["match"]

        reply = (REF / f"{g}_reply_answer.md").read_text()
        w, parse = dsl.apply_edits(reply)
        ref = np.load(REF / "networks" / f"{g}.npz")
        maxdiff = max(float(np.abs(w[k] - ref[dsl.KEYS[k]]).max()) for k in dsl.SHAPES)
        report["weights"][g] = {
            "statements": parse["statements"],
            "applied": parse["applied"],
            "errors": len(parse["errors"]),
            "nonzero": dsl.nonzero_counts(w),
            "max_abs_diff_vs_released_npz": maxdiff,
            "match": maxdiff == 0.0 and parse["applied"] == parse["statements"],
        }
        ok &= report["weights"][g]["match"]

        if args.episodes:
            res = evaluate(g, MLPPolicy(w), episodes=args.episodes)
            res.pop("scores", None)
            report["scores"][g] = res
            print(f"{g}: {res['mean']:.1f} ± {res['se']:.1f} "
                  f"({args.episodes} eps)")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {g: v["match"] for g, v in report[k].items()}
                      for k in ("prompts", "weights")}, indent=1))
    print("ALL CHECKS PASS" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
