"""
analysis/plot_cournot.py — Plots for the Cournot CURB sweep.

Reads results/cournot_n{N}/run_{K}/*/seed_*/steps.csv and produces:
  1. ci_training.png      CI over training, mean ± stderr per condition
  2. quantity_training.png Per-agent quantity over training (one subplot per condition)
  3. tv_signal.png         TV(j→j) over training for CURB conditions
  4. summary_bar.png       Final CI bar chart

Uses the schema written by cournot_sweep.py:
    steps.csv: step, q_agent0, q_agent1, ..., CI, tv0, tv1, ...
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Condition order + colors (for n=2 cournot sweep)
CONDS_ORDER = [
    "baseline",
    "imperfect_monitoring",
    "full_l01",
    "full_l02",
    "full_l05",
]

COND_COLORS = {
    "baseline":             "#444444",
    "imperfect_monitoring": "#9C27B0",
    "full_l01":             "#FFC107",
    "full_l02":             "#FF9800",
    "full_l05":             "#1976D2",
}

COND_LABELS = {
    "baseline":             "baseline",
    "imperfect_monitoring": "imperfect monitoring",
    "full_l01":             "CURB λ=0.1",
    "full_l02":             "CURB λ=0.2",
    "full_l05":             "CURB λ=0.5",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _color(c):
    return COND_COLORS.get(c, "#777777")


def _label(c):
    return COND_LABELS.get(c, c)


def _load_seeds(run_dir, cond_name, n_seeds):
    """Return (steps, ci_arr, q_arr, tv_arr) where _arr shape is (n_seeds_found, T)."""
    cis, qs, tvs = [], [], []
    steps = None
    for seed in range(n_seeds):
        path = os.path.join(run_dir, cond_name, f"seed_{seed}", "steps.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if steps is None:
            steps = df["step"].values
        L = min(len(df), len(steps))
        cis.append(df["CI"].values[:L])
        q_cols  = [c for c in df.columns if c.startswith("q_agent")]
        tv_cols = [c for c in df.columns if c.startswith("tv")]
        qs.append(df[q_cols].values[:L])
        tvs.append(df[tv_cols].values[:L])
    if not cis:
        return None, None, None, None
    L = min(len(c) for c in cis)
    return (steps[:L],
            np.stack([c[:L] for c in cis]),
            np.stack([q[:L] for q in qs]),
            np.stack([t[:L] for t in tvs]))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_ci_training(run_dir, n_seeds, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for cond in CONDS_ORDER:
        steps, ci, _, _ = _load_seeds(run_dir, cond, n_seeds)
        if ci is None:
            continue
        m   = ci.mean(0)
        sem = ci.std(0) / np.sqrt(ci.shape[0])
        ax.plot(steps / 1e6, m, color=_color(cond), lw=1.6, label=_label(cond))
        ax.fill_between(steps / 1e6, m - sem, m + sem,
                        color=_color(cond), alpha=0.18)
    ax.axhline(0.0, color="#888", ls="--", lw=0.7)
    ax.axhline(1.0, color="#444", ls=":",  lw=0.7)
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Collusion Index")
    ax.set_title("Cournot n=2: CI over training (mean ± SEM)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_quantity_training(run_dir, n_seeds, out_path, n=2):
    cols  = [c for c in CONDS_ORDER if os.path.isdir(os.path.join(run_dir, c))]
    ncols = min(3, len(cols))
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.0 * nrows),
                             sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    seller_colors = ["#1976D2", "#E91E63"][:n]

    nash_q, mono_q = 100.0, 75.0  # Calvano (2021) defaults for n=2

    for idx, cond in enumerate(cols):
        ax = axes[idx // ncols, idx % ncols]
        steps, _, q, _ = _load_seeds(run_dir, cond, n_seeds)
        if q is None:
            ax.set_visible(False)
            continue
        for i in range(min(n, q.shape[2])):
            m   = q[:, :, i].mean(0)
            std = q[:, :, i].std(0)
            ax.plot(steps / 1e6, m, color=seller_colors[i], lw=1.5,
                    label=f"Seller {i}")
            ax.fill_between(steps / 1e6, m - std, m + std,
                            color=seller_colors[i], alpha=0.15)
        ax.axhline(nash_q, color="#888", ls="--", lw=0.8, label="Nash q*=100")
        ax.axhline(mono_q, color="#444", ls=":",  lw=0.8, label="Mono q=75")
        ax.set_title(_label(cond))
        ax.set_xlabel("Training steps (M)")
        ax.grid(axis="y", alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("Quantity")
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    for j in range(len(cols), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle("Cournot n=2: per-agent quantity over training", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_tv_signal(run_dir, n_seeds, out_path):
    """TV penalty over training — only meaningful for CURB conditions."""
    curb_conds = [c for c in CONDS_ORDER if c.startswith("full_")]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for cond in curb_conds:
        steps, _, _, tv = _load_seeds(run_dir, cond, n_seeds)
        if tv is None:
            continue
        # Mean TV across agents and seeds
        tv_mean = tv.mean(axis=(0, 2))
        ax.plot(steps / 1e6, tv_mean, color=_color(cond), lw=1.6, label=_label(cond))
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Mean TV (across agents)")
    ax.set_title("Cournot n=2: TV signal over training (CURB conditions)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_summary_bar(sweep_csv, out_path):
    df    = pd.read_csv(sweep_csv)
    means = df.groupby("condition")["final_CI"].mean()
    sems  = df.groupby("condition")["final_CI"].sem()
    conds = [c for c in CONDS_ORDER if c in means.index]

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(len(conds))
    ax.bar(x, [means[c] for c in conds], yerr=[sems[c] for c in conds],
           color=[_color(c) for c in conds], capsize=4, width=0.65)
    ax.axhline(0.0, color="#888", ls="--", lw=0.7)
    ax.axhline(1.0, color="#444", ls=":",  lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(c) for c in conds], rotation=20, ha="right",
                       fontsize=9)
    ax.set_ylabel("Final CI (mean ± SEM)")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Cournot n=2: Final CI by condition")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Driver ────────────────────────────────────────────────────────────────────

def make_all_plots(run_dir, n=2, n_seeds=20):
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_ci_training(run_dir, n_seeds,       os.path.join(out_dir, "ci_training.png"))
    plot_quantity_training(run_dir, n_seeds, os.path.join(out_dir, "quantity_training.png"), n=n)
    plot_tv_signal(run_dir, n_seeds,         os.path.join(out_dir, "tv_signal.png"))

    sweep_csv = os.path.join(run_dir, "sweep_results.csv")
    if os.path.exists(sweep_csv):
        plot_summary_bar(sweep_csv, os.path.join(out_dir, "summary_bar.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n",       type=int, default=2)
    parser.add_argument("--seeds",   type=int, default=20)
    args = parser.parse_args()
    make_all_plots(args.run_dir, n=args.n, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
