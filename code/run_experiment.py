"""Run the exp5 protocol: ask an LLM for weights, parse the edits block, score it.

Examples
  # pipeline check without any API key (replays Astra's own reply)
  python run_experiment.py --game asterix --attempts 1 \
      --mock-reply ../ref/exp5/asterix_reply_answer.md --episodes 100

  # live pilot on DeepSeek
  export DEEPSEEK_API_KEY=...; python run_experiment.py --game asterix --attempts 3

  # everything
  python run_experiment.py --game all --attempts 3
"""
import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

import edits_dsl as dsl
import llm
from core import MLPPolicy, evaluate

HERE = pathlib.Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
GAMES = ["asterix", "breakout", "seaquest"]


def load_prompt(game):
    return (PROMPTS / f"{game}.md").read_text()


def run_one(game, run_dir, prompt, args, mock_reply=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.txt").write_text(prompt)
    (run_dir / "config.json").write_text(json.dumps({
        "game": game, "model": args.cfg["model"],
        "reasoning_effort": args.cfg.get("reasoning_effort"),
        "temperature": args.cfg["temperature"],
        "max_tokens": args.cfg["max_tokens"], "attempt": args.attempt,
        "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
        "prompt_chars": len(prompt), "started": int(time.time()),
    }, indent=1))

    if mock_reply:
        text = pathlib.Path(mock_reply).read_text()
        record = {"provider": "mock", "model": "mock", "text": text,
                  "reasoning": "", "finish_reason": "stop", "truncated": False,
                  "usage": {}, "wall_s": 0.0}
        (run_dir / "reply.md").write_text(text)
        (run_dir / "usage.json").write_text(json.dumps(
            {k: v for k, v in record.items() if k != "text"}, indent=1))
    else:
        record = llm.complete(prompt, config=args.cfg,
                              system=llm.DEFAULT_SYSTEM, log=print)
        llm.save_record(record, run_dir, prompt)
        text = record["text"]

    weights, report = dsl.apply_edits(text)
    report["truncated"] = bool(record.get("truncated"))
    report["nonzero"] = dsl.nonzero_counts(weights)
    report["nonzero_total"] = int(sum(report["nonzero"].values()))
    report["parameters"] = int(sum(v.size for v in weights.values()))
    (run_dir / "parse.json").write_text(json.dumps(report, indent=1))

    if not report["applied"]:
        (run_dir / "eval.json").write_text(json.dumps(
            {"error": "no statements applied", "mean": 0.0, "se": 0.0,
             "episodes": args.episodes}, indent=1))
        return {"game": game, "attempt": args.attempt, "status": "no_weight",
                "mean": 0.0, "se": 0.0, "statements": report["statements"],
                "nonzero": 0}

    npz = run_dir / "weights.npz"
    dsl.save_npz(weights, npz)

    t0 = time.time()
    policy = MLPPolicy(weights)
    res = evaluate(game, policy, episodes=args.episodes, seed0=args.seed0)
    res["wall_s"] = round(time.time() - t0, 1)
    res["weights"] = str(npz)
    res["statements"] = report["statements"]
    res["applied"] = report["applied"]
    res["nonzero_total"] = report["nonzero_total"]
    res.pop("scores", None)
    (run_dir / "eval.json").write_text(json.dumps(res, indent=1))
    return {"game": game, "attempt": args.attempt, "status": "ok",
            "mean": res["mean"], "se": res["se"],
            "statements": report["statements"],
            "nonzero": report["nonzero_total"],
            "cost_usd": (record.get("cost") or {}).get("usd", 0.0),
            "wall_s": res["wall_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="all", choices=GAMES + ["all"])
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--runs-root", default=str(HERE / "runs"))
    ap.add_argument("--mock-reply", default=None,
                    help="replay this reply instead of calling the API")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    games = GAMES if args.game == "all" else [args.game]
    root = pathlib.Path(args.runs_root)
    args.cfg = llm.default_config(args.model, args.temperature, args.max_tokens)

    if args.dry_run:
        for g in games:
            p = load_prompt(g)
            print(f"{g}: prompt {len(p)} chars")
        print("config:", {k: v for k, v in args.cfg.items() if k != "api_key"})
        return

    if not args.mock_reply and not llm.available():
        raise SystemExit("no API key: export DEEPSEEK_API_KEY=... "
                         "(or use --mock-reply for a pipeline check)")

    results = []
    for g in games:
        prompt = load_prompt(g)
        for k in range(1, args.attempts + 1):
            run_dir = root / g / f"attempt{k}"
            eval_file = run_dir / "eval.json"
            if args.resume and eval_file.exists():
                prev = json.loads(eval_file.read_text())
                print(f"[skip] {g} attempt{k}: mean={prev.get('mean')}")
                results.append({"game": g, "attempt": k, "status": "cached",
                                "mean": prev.get("mean", 0.0),
                                "se": prev.get("se", 0.0)})
                continue
            print(f"\n=== {g} attempt {k}/{args.attempts} "
                  f"(prompt {len(prompt)} chars, model {args.cfg['model']}) ===",
                  flush=True)
            args.attempt = k
            r = run_one(g, run_dir, prompt, args, args.mock_reply)
            results.append(r)
            print(f"--- {g} attempt{k}: {r['status']} mean={r['mean']:.1f}"
                  f" ± {r['se']:.1f} ({r['wall_s'] if 'wall_s' in r else 0}s)",
                  flush=True)

    # per-game summary
    for g in games:
        rows = [r for r in results if r["game"] == g]
        if not rows:
            continue
        means = [r["mean"] for r in rows]
        summary = {
            "game": g,
            "attempts": len(rows),
            "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows), 4),
            "mean": float(np.mean(means)),
            "se": float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0,
            "per_attempt": rows,
        }
        (root / g).mkdir(parents=True, exist_ok=True)
        (root / g / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"\n{g}: mean {summary['mean']:.1f} ± {summary['se']:.1f} "
              f"over {len(rows)} attempt(s)")


if __name__ == "__main__":
    main()
