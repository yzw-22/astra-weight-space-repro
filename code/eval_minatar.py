"""Evaluate a 600-32-32-6 tanh policy on a MinAtar game under the blog's protocol.

  python eval_minatar.py --game breakout --weights ref/exp5/networks/breakout.npz \
      --episodes 1000

Protocol: 6 standardized input channels, 6 actions sampled from the softmax,
sticky actions p=0.1, episode cap 10,000 steps, score = mean return.
"""
import argparse
import json
import time

from core import MLPPolicy, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True,
                    choices=["breakout", "asterix", "seaquest"])
    ap.add_argument("--weights", required=True, help="npz with the 6 tensors")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--keep-scores", action="store_true")
    args = ap.parse_args()

    policy = MLPPolicy.from_npz(args.weights)
    t0 = time.time()
    res = evaluate(args.game, policy, args.episodes, args.seed0,
                   log_every=args.log_every)
    res["weights"] = args.weights
    res["wall_s"] = round(time.time() - t0, 1)
    res["nonzero"] = {k: int((v != 0).sum()) for k, v in policy.w.items()}
    if not args.keep_scores:
        res.pop("scores")
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
