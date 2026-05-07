"""
analysis/plot_dqn_calvano.py — Plots for DQN Calvano sweep.

Figures
-------
1. ci_training.pdf   — CI over training, mean ± stderr
2. tv_signal.pdf     — TV distance (CURB conditions)
3. summary_bar.pdf   — Final CI bar chart

Usage
-----
  python plot_dqn_calvano.py --run_dir ../results/dqn_calvano/run_1
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "baseline_dqn":  "#555555",
    "curb_l01_dqn":  "#90CAF9",
    "curb_l02_dqn":  "#1976D2",
    "curb_l05_dqn":  "#FF6F00",
}
LABELS = {
    "baseline_dqn":  "DQN Baseline",
    "curb_l01_dqn":  "DQN + CURB λ=0.1",
    "curb_l02_dqn":  "DQN + CURB λ=0.2",
    "curb_l05_dqn":  "DQN + CURB λ=0.5",
}
ORDER = ["baseline_dqn", "curb_l01_dqn", "curb_l02_dqn", "curb_l05_dqn"]


def _color(c): return COLORS.get(c, "#AAAAAA")
def _label(c): return LABELS.get(c, c)


def _load_ts(run_dir, condition, seeds):
    records = {}
    for seed in seeds:
        path = os.path.join(run_dir, condition, f"seed_{seed}", "steps.csv")
        if not os.path.exists(path):
            continue
        records[seed] = pd.read_csv(path)
    if not records:
        return None
    L     = min(len(df) for df in records.values())
    ref   = next(iter(records.values()))
    steps = ref["step"].values[:L]
    ci    = np.stack([df["CI"].values[:L] for df in records.values()])
    tv_cols = [c for c in ref.columns if c.startswith("tv")]
    tv = np.stack([df[tv_cols].values[:L].mean(1)
                   for df in records.values()]) if tv_cols else None
    return {"steps": steps, "ci": ci, "tv": tv}


def _shade(ax, steps, data, color, label, lw=1.8):
    m  = data.mean(0)
    se = data.std(0) / np.sqrt(data.shape[0])
    ax.plot(steps, m, color=color, label=label, linewidth=lw)
    ax.fill_between(steps, m - se, m + se, color=color, alpha=0.15)


def plot_ci_training(run_dir, n, seeds, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for cond in ORDER:
        data = _load_ts(run_dir, cond, seeds)
        if data is None:
            continue
        _shade(ax, data["steps"], data["ci"], _color(cond), _label(cond))
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Monopoly")
    ax.axhline(0.0, color="grey",  lw=0.8, ls="--", label="Nash")
    ax.set_xlabel("Training steps"); ax.set_ylabel("Collusion Index (CI)")
    ax.set_title(f"DQN Calvano n={n}: CI during training")
    ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.set_ylim(-0.1, 1.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


def plot_tv_signal(run_dir, n, seeds, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for cond in [c for c in ORDER if "curb" in c]:
        data = _load_ts(run_dir, cond, seeds)
        if data is None or data["tv"] is None:
            continue
        _shade(ax, data["steps"], data["tv"], _color(cond), _label(cond))
    ax.set_xlabel("Training steps"); ax.set_ylabel("TV distance")
    ax.set_title(f"DQN Calvano n={n}: TV signal (CURB conditions)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


def plot_summary_bar(sweep_csv, n, out_path):
    df    = pd.read_csv(sweep_csv)
    means = df.groupby("condition")["final_CI"].mean()
    sems  = df.groupby("condition")["final_CI"].sem()
    conds = [c for c in ORDER if c in means.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(conds)), [means[c] for c in conds],
           yerr=[sems[c] for c in conds],
           color=[_color(c) for c in conds], capsize=4, width=0.6)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Monopoly")
    ax.axhline(0.0, color="grey",  lw=0.8, ls="--", label="Nash")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([_label(c) for c in conds], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Final CI"); ax.set_ylim(-0.05, 1.15)
    ax.set_title(f"DQN Calvano n={n}: Final CI by Condition")
    ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


def make_all_plots(run_dir, n, n_seeds):
    out_dir   = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    seeds     = list(range(n_seeds))
    sweep_csv = os.path.join(run_dir, "sweep_results.csv")

    plot_ci_training(run_dir, n, seeds, os.path.join(out_dir, "ci_training.pdf"))
    plot_tv_signal(run_dir, n, seeds,   os.path.join(out_dir, "tv_signal.pdf"))
    if os.path.exists(sweep_csv):
        plot_summary_bar(sweep_csv, n, os.path.join(out_dir, "summary_bar.pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--n",       type=int, default=2)
    parser.add_argument("--seeds",   type=int, default=20)
    args = parser.parse_args()
    make_all_plots(args.run_dir, args.n, args.seeds)


if __name__ == "__main__":
    main()
