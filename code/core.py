"""Shared pieces of the MinAtar weight-space-programming reproduction.

  * MinAtar6   : MinAtar env wrapped to the protocol's 6 standardized input channels
  * MLPPolicy : 600 -> 32 -> 32 -> 6 tanh actor, weights as a dict of numpy arrays
  * evaluate  : the blog's scoring protocol (softmax sampling, sticky 0.1, 10k steps)
"""
import os
import pathlib

import numpy as np

# minatar imports pyplot at import time; keep its cache inside the workspace
os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(__file__).resolve().parent / ".mplcache"))

from minatar import Environment  # noqa: E402

# standardized slot -> native MinAtar channel names
# slot 0 player, 1 danger, 2 reward, 3 projectile, 4 trail, 5 gauge
CHANNEL_MAP = {
    "breakout": {0: ["paddle"], 2: ["brick"], 3: ["ball"], 4: ["trail"]},
    "asterix": {0: ["player"], 1: ["enemy"], 2: ["gold"], 4: ["trail"]},
    "seaquest": {
        0: ["sub_front", "sub_back"],
        1: ["enemy_fish", "enemy_sub", "enemy_bullet"],
        2: ["enemy_fish", "enemy_sub", "diver"],
        3: ["friendly_bullet"],
        4: ["trail", "sub_back"],
        5: ["oxygen_guage", "diver_guage"],
    },
}

SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32),
          "b2": (32,), "W3": (32, 6), "b3": (6,)}

NPZ_KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias",
            "W2": "actor/dense2/kernel", "b2": "actor/dense2/bias",
            "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}


class MinAtar6:
    """10x10 grid -> 600-vector with the paper's channel standardization."""

    def __init__(self, game, sticky_action_prob=0.1, difficulty_ramping=True):
        self.game = game
        self.env = Environment(game, sticky_action_prob=sticky_action_prob,
                               difficulty_ramping=difficulty_ramping)
        native = self.env.env.channels
        # precompute (slot, native index) pairs
        self.pairs = [(slot, native[name])
                      for slot, names in CHANNEL_MAP[game].items()
                      for name in names]

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        self.env.reset()
        return self.obs()

    def obs(self):
        state = self.env.state()
        obs = np.zeros((10, 10, 6), dtype=np.float64)
        for slot, ch in self.pairs:
            np.maximum(obs[:, :, slot], state[:, :, ch], out=obs[:, :, slot])
        return obs.reshape(-1)

    def step(self, action):
        r, terminal = self.env.act(action)
        return self.obs(), r, terminal

    def num_actions(self):
        return 6


class MLPPolicy:
    def __init__(self, weights):
        self.w = {k: np.asarray(weights[k], dtype=np.float64) for k in SHAPES}

    @classmethod
    def from_npz(cls, path):
        z = np.load(path)
        w = {}
        for short, long in NPZ_KEYS.items():
            w[short] = z[long] if long in z.files else z[short]
        return cls(w)

    def logits(self, x):
        h1 = np.tanh(x @ self.w["W1"] + self.w["b1"])
        h2 = np.tanh(h1 @ self.w["W2"] + self.w["b2"])
        return h2 @ self.w["W3"] + self.w["b3"]

    def sample(self, x, rng):
        logits = self.logits(x)
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        return int(rng.choice(len(p), p=p))

    def greedy(self, x):
        return int(np.argmax(self.logits(x)))


def run_episode(game, policy, seed, rng, max_steps=10_000):
    env = MinAtar6(game)
    env.seed(seed)
    x = env.reset()
    total, steps, terminal = 0.0, 0, False
    while steps < max_steps:
        x, r, terminal = env.step(policy.sample(x, rng))
        total += r
        steps += 1
        if terminal:
            break
    return total, steps


def evaluate(game, policy, episodes=1000, seed0=0, max_steps=10_000,
             rng=None, log_every=0, log=print):
    rng = np.random.default_rng(seed0 + 977) if rng is None else rng
    scores, lengths = [], []
    for i in range(episodes):
        s, n = run_episode(game, policy, seed0 + i, rng, max_steps)
        scores.append(s)
        lengths.append(n)
        if log_every and (i + 1) % log_every == 0:
            log(f"  {i + 1}/{episodes} mean={np.mean(scores):.1f}")
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "game": game,
        "episodes": episodes,
        "mean": float(scores.mean()),
        "se": float(scores.std(ddof=1) / np.sqrt(episodes)) if episodes > 1 else 0.0,
        "max": float(scores.max()),
        "mean_length": float(np.mean(lengths)),
        "scores": [float(s) for s in scores],
    }
