"""Figures for the weight-space-programming reproduction.

Reads the run artifacts of the public repo (runs/*/*/attempt*/eval.json) plus the
paper's baselines, and renders two figures:

  fig1_scores            A  scores on a log axis (3 games x 7 references)
                         B  score vs mean episode length, with iso-rate lines
  fig2_depth_variance    C  Breakout survival by look-ahead depth
                         D  the three samples behind every point

Usage:
  python figures/make_figures.py            # from the repository root
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

GAMES = ["breakout", "asterix", "seaquest"]
ARMS = ["astra", "flash", "v4pro"]
LABEL = {"astra": "Astra", "flash": "V4.1-Flash", "v4pro": "V4-Pro"}
GAME_LABEL = {"breakout": "Breakout", "asterix": "Asterix", "seaquest": "Seaquest"}

COLOR = {"astra": "#0F6E6E", "flash": "#D97706", "v4pro": "#6D4AA6"}
MARKER = {"astra": "o", "flash": "s", "v4pro": "^"}
INK = "#111827"
MUTED = "#6B7280"

# the paper's one-shot Astra scores and PPO baselines (blog table, 2026-11-02)
PAPER = {
    "breakout": {"ppo100": 15.8, "ppo500": 25.3, "astra": 517.4},
    "asterix": {"ppo100": 25.8, "ppo500": 55.2, "astra": 53.2},
    "seaquest": {"ppo100": 5.6, "ppo500": 88.9, "astra": 53.1},
}


def house_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": "#374151",
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
    })


def load_runs(repo):
    """{arm: {game: [ {mean, se, length}, ... ]}} from the stored artifacts."""
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for game in GAMES:
            rows = []
            for d in sorted((repo / "runs" / arm / game).glob("attempt*")):
                ev = d / "eval.json"
                if ev.exists():
                    e = json.loads(ev.read_text())
                    rows.append({"mean": e["mean"], "se": e["se"],
                                 "length": e["mean_length"]})
            out[arm][game] = rows
    return out


def load_random(repo):
    path = repo / "logs" / "random_baseline.json"
    if not path.exists():
        return {g: {"mean": np.nan} for g in GAMES}
    return json.loads(path.read_text())


def panel_letter(ax, letter, title, x=-0.02, y=1.16):
    ax.text(x, y, f"{letter}", transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=INK, va="top")
    ax.text(x + 0.045, y, title, transform=ax.transAxes, fontsize=10,
            color=INK, va="top")


def fig_scores(data, rand, out_dir):
    fig = plt.figure(figsize=(12.2, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.26,
                          left=0.105, right=0.985, top=0.855, bottom=0.135)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- A: scores, one lane per game --------------------------------------
    ax = ax_a
    ax.set_xscale("log")
    ax.set_xlim(0.04, 2000)
    ax.set_ylim(-0.90, 3.28)
    ax.set_yticks([0.08, 1.38, 2.68])
    ax.set_yticklabels([GAME_LABEL[g] for g in reversed(GAMES)], fontsize=9.5,
                       fontweight="semibold")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", which="major")
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())

    for row, game in zip([2.6, 1.3, 0.0], GAMES):
        ref = PAPER[game]
        lane_ref, lane_band, lane_model = row + 0.40, row + 0.14, row - 0.22
        ax.plot([ref["ppo100"], ref["ppo500"]], [lane_band, lane_band],
                color="#D1D5DB", lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([rand[game]["mean"]], [lane_band], s=26, marker="o",
                   color="#6B7280", zorder=2)
        ax.scatter([ref["astra"]], [lane_ref], marker="D", s=40,
                   facecolor="white", edgecolor="#6B7280", linewidth=1.1, zorder=4)
        ax.text(ref["astra"], lane_ref + 0.14, f"{ref['astra']:g}", ha="center",
                va="bottom", fontsize=7.8, color=MUTED)

        for arm in ARMS:
            rows = data[arm][game]
            m = float(np.mean([r["mean"] for r in rows]))
            se = float(np.std([r["mean"] for r in rows], ddof=1) / np.sqrt(len(rows)))
            ax.errorbar([m], [lane_model], xerr=[se], fmt=MARKER[arm], ms=7,
                        color=COLOR[arm], ecolor=COLOR[arm], elinewidth=1.2,
                        capsize=3, capthick=1.2, zorder=5, markeredgecolor="white",
                        markeredgewidth=0.6)
            ax.plot([m, m], [lane_model - 0.16, lane_model - 0.02], color=COLOR[arm],
                    lw=0.8, alpha=0.35, zorder=3)
            ax.text(m, lane_model - 0.19, f"{m:.1f}", ha="center", va="top",
                    fontsize=7.8, color=COLOR[arm], fontweight="semibold")

    ax.set_xlabel("score per episode (log scale)", labelpad=6)
    panel_letter(ax, "A", "Scores, 1,000 episodes \u00d7 3 attempts", x=-0.10, y=1.06)

    # ---- B: where the score comes from -------------------------------------
    ax = ax_b
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(35, 12000)
    ax.set_ylim(0.04, 1600)
    ax.grid(which="major", axis="both")

    xs = np.array([35, 12000])
    for rate in (0.01, 0.03, 0.1):
        ax.plot(xs, rate * xs, color="#D1D5DB", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(40, 150, "dashed lines: reward per step\n"
                     "0.01 \u00b7 0.03 \u00b7 0.10\n"
                     "Breakout attempts: 0.05\u20130.085",
            fontsize=7.2, color="#9CA3AF", va="top", ha="left", linespacing=1.6)

    for arm in ARMS:
        for game in GAMES:
            for r in data[arm][game]:
                ax.scatter(r["length"], max(r["mean"], 0.05), s=58, marker=MARKER[arm],
                           color=COLOR[arm], alpha=0.9, edgecolor="white",
                           linewidth=0.8, zorder=3)

    ax.annotate("survives 4,847 steps\nbut scores 0.15",
                xy=(4847, 0.147), xytext=(120, 0.055), fontsize=7.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color="#D1D5DB", lw=0.9,
                                connectionstyle="arc3,rad=-0.25"))

    ax.set_xlabel("mean episode length, in steps (log scale)", labelpad=6)
    ax.set_ylabel("score (log scale)", labelpad=6)
    panel_letter(ax, "B", "Score = survival \u00d7 reward per step",
                 x=-0.11, y=1.06)

    handles = [
        Line2D([], [], marker=MARKER["astra"], color=COLOR["astra"], ls="",
               ms=6, label="Astra"),
        Line2D([], [], marker=MARKER["flash"], color=COLOR["flash"], ls="",
               ms=6, label="V4.1-Flash"),
        Line2D([], [], marker=MARKER["v4pro"], color=COLOR["v4pro"], ls="",
               ms=6, label="V4-Pro"),
        Line2D([], [], marker="D", color="#6B7280", ls="", mfc="white", ms=6,
               label="paper (1 sample)"),
        Line2D([], [], color="#D1D5DB", lw=6, label="PPO 100M\u2013500M"),
        Line2D([], [], marker="o", color="#6B7280", ls="", ms=5, label="random"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.105, 0.99),
               ncol=6, handletextpad=0.4, columnspacing=1.6)

    fig.text(0.105, 0.028,
             "Sources: runs/ in this repository (3 attempts \u00d7 1,000 episodes); "
             "PPO and paper Astra from the paper; random measured with this harness.",
             fontsize=7.5, color="#9CA3AF")
    return save(fig, out_dir / "fig1_scores")


def fig_depth(data, out_dir):
    fig = plt.figure(figsize=(11.4, 5.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.16, 1.0], hspace=0.60,
                          wspace=0.16, left=0.175, right=0.975, top=0.84, bottom=0.15)
    ax_c = fig.add_subplot(gs[:, 0])
    axes_d = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    # ---- C: Breakout ladder -------------------------------------------------
    b = {arm: data[arm]["breakout"] for arm in ARMS}
    groups = [
        ("Astra\nbrick-bounce", [r["length"] for r in b["astra"]], "astra"),
        ("V4.1-Flash\nwall only",
         [r["length"] for r in b["flash"] if r["length"] > 500], "flash"),
        ("V4.1-Flash\ncurrent column",
         [r["length"] for r in b["flash"] if r["length"] <= 500], "flash"),
        ("V4-Pro\ncurrent column", [r["length"] for r in b["v4pro"]], "v4pro"),
    ]
    ax = ax_c
    ax.set_xscale("log")
    ax.set_xlim(40, 13000)
    ax.set_ylim(-0.7, len(groups) - 0.15)
    ypos = list(range(len(groups)))[::-1]
    ax.set_yticks(ypos)
    ax.set_yticklabels([g[0] for g in groups], fontsize=8.5, linespacing=1.4)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", which="major")
    ax.spines["left"].set_visible(False)

    means = []
    for y, (_, vals, arm) in zip(ypos, groups):
        vals = np.asarray(vals, dtype=float)
        ys = y + (np.linspace(-0.11, 0.11, len(vals)) if len(vals) > 1 else 0.0)
        ax.scatter(vals, ys, s=46, color=COLOR[arm], edgecolor="white",
                   linewidth=0.7, zorder=3)
        m = float(vals.mean())
        means.append(m)
        ax.plot([vals.min(), vals.max()], [y, y], color=COLOR[arm], lw=1.6,
                alpha=0.30, solid_capstyle="round", zorder=2)
        ax.text(vals.max() * 1.14, y, f"{m:,.0f}", fontsize=8.5, color=COLOR[arm],
                ha="left", va="center", fontweight="semibold")

    for i in range(1, len(means)):
        ratio = means[i - 1] / means[i]
        x = float(np.sqrt(means[i - 1] * means[i]))
        ax.annotate("", xy=(x, ypos[i - 1] - 0.30), xytext=(x, ypos[i] + 0.30),
                    arrowprops=dict(arrowstyle="<->", color="#B6BCC4", lw=1.0))
        ax.text(x * 1.07, (ypos[i - 1] + ypos[i]) / 2, f"\u00d7{ratio:.1f}",
                fontsize=8, color=MUTED, ha="left", va="center")

    ax.set_xlabel("mean episode length, in steps (log scale)", labelpad=8)
    panel_letter(ax, "C", "Breakout: survival tracks look-ahead depth",
                 x=-0.60, y=1.12)

    # ---- D: the three samples ----------------------------------------------
    limits = {"breakout": (0.03, 1300), "asterix": (0.03, 170), "seaquest": (0.03, 110)}
    for ax, game in zip(axes_d, GAMES):
        vals = {arm: [r["mean"] for r in data[arm][game]] for arm in ARMS}
        ax.set_yscale("log")
        ax.set_ylim(*limits[game])
        ax.set_xlim(-0.55, 3.05)
        ax.set_xticks([0, 1, 2])
        ax.grid(axis="y", which="major")
        paper = PAPER[game]["astra"]
        ax.axhline(paper, color="#B6BCC4", lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.text(3.0, paper * 1.12, f"paper {paper:g}", fontsize=7, color=MUTED,
                ha="right", va="bottom")

        for i, arm in enumerate(ARMS):
            v = np.asarray(vals[arm], dtype=float)
            floor = limits[game][0] * 1.5
            shown = np.where(v <= 0, floor, v)
            jitter = np.linspace(-0.11, 0.11, len(v)) if len(v) > 1 else np.zeros(len(v))
            ax.scatter(i + jitter, shown, s=30, color=COLOR[arm], alpha=0.9,
                       edgecolor="white", linewidth=0.6, zorder=3)
            for orig, s_, jj in zip(v, shown, jitter):
                if orig <= 0:
                    ax.text(i + jj + 0.09, s_, "0", fontsize=6.5, color=COLOR[arm],
                            va="center")
            m = float(v.mean())
            se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            ax.errorbar([i], [m], yerr=[se], fmt="none", ecolor=COLOR[arm],
                        elinewidth=1.3, capsize=4, capthick=1.3, zorder=4)
            ax.plot([i - 0.20, i + 0.20], [m, m], color=COLOR[arm], lw=1.8, zorder=5)

        ax.set_title(GAME_LABEL[game], fontsize=8.5, loc="left", pad=3,
                     color="#374151", fontweight="semibold")
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_visible(False)
        if game != GAMES[-1]:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticklabels(["Astra", "Flash", "Pro"], fontsize=8.5)

    axes_d[0].set_ylabel("score (log)", labelpad=4)
    panel_letter(axes_d[0], "D", "Every point is 3 samples", x=-0.36, y=1.62)

    fig.text(0.235, 0.025,
             "C: one dot per attempt, bar spans the observed range, label is the mean. "
             "D: dashed line = the paper's single sample; bar = mean of three attempts, "
             "whisker = standard error.",
             fontsize=7.5, color="#9CA3AF")
    return save(fig, out_dir / "fig2_depth_variance")


def save(fig, stem):
    made = []
    for ext in ("png", "svg"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.12, facecolor="white")
        made.append(str(path))
    plt.close(fig)
    return made


def main():
    ap = argparse.ArgumentParser()
    here = pathlib.Path(__file__).resolve().parent
    ap.add_argument("--repo", default=str(here.parent))
    ap.add_argument("--out", default=str(here / "out"))
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    house_style()
    data = load_runs(repo)
    rand = load_random(repo)

    for path in fig_scores(data, rand, out) + fig_depth(data, out):
        print(path)

    missing = [(a, g) for a in ARMS for g in GAMES if not data[a][g]]
    if missing:
        print("WARNING missing runs:", missing)


if __name__ == "__main__":
    main()
