"""
analysis/plot_neurips_extra.py — paper-ready CI training plots for Cournot and
DQN-Calvano runs. Same NeurIPS-style sizing/legend conventions as
plot_neurips_calvano.py.

Outputs:
  <run_dir>/plots/neurips/ci_training.{pdf,png}
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7,
    "lines.linewidth":  1.4,
    "axes.linewidth":   0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

FIGSIZE    = (3.9, 2.7)
SAVEFIG_KW = dict(bbox_inches="tight", pad_inches=0.03)


# ── Per-experiment configs ────────────────────────────────────────────────────

COURNOT_ORDER = ["baseline", "imperfect_monitoring",
                 "full_l01", "full_l02", "full_l05"]
COURNOT_LABELS = {
    "baseline":             "No intervention",
    "imperfect_monitoring": "Imperfect monitoring",
    "full_l01":             "CURB $\\lambda{=}0.1$",
    "full_l02":             "CURB $\\lambda{=}0.2$",
    "full_l05":             "CURB $\\lambda{=}0.5$",
}
COURNOT_COLORS = {
    "baseline":             "#444444",
    "imperfect_monitoring": "#9C27B0",
    "full_l01":             "#90CAF9",
    "full_l02":             "#1976D2",
    "full_l05":             "#FF9800",
}

DQN_ORDER  = ["baseline_dqn", "curb_l01_dqn", "curb_l02_dqn", "curb_l05_dqn"]
DQN_LABELS = {
    "baseline_dqn": "No intervention",
    "curb_l01_dqn": "CURB $\\lambda{=}0.1$",
    "curb_l02_dqn": "CURB $\\lambda{=}0.2$",
    "curb_l05_dqn": "CURB $\\lambda{=}0.5$",
}
DQN_COLORS = {
    "baseline_dqn": "#444444",
    "curb_l01_dqn": "#90CAF9",
    "curb_l02_dqn": "#1976D2",
    "curb_l05_dqn": "#FF9800",
}


def _load_ci(run_dir, condition, n_seeds):
    """Returns (steps, ci_arr shape (seeds, T)) or (None, None)."""
    cis, steps = [], None
    for seed in range(n_seeds):
        p = os.path.join(run_dir, condition, f"seed_{seed}", "steps.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if steps is None:
            steps = df["step"].values
        L = min(len(df), len(steps))
        cis.append(df["CI"].values[:L])
    if not cis:
        return None, None
    L = min(len(c) for c in cis)
    return steps[:L], np.stack([c[:L] for c in cis])


def _shade(ax, x, arr, color, label, lw=1.4):
    m  = arr.mean(0)
    se = arr.std(0) / np.sqrt(arr.shape[0])
    ax.plot(x, m, color=color, label=label, linewidth=lw)
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.18)


def _plot_ci(run_dir, n_seeds, order, labels, colors, out_pdf, out_png):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cond in order:
        steps, ci = _load_ci(run_dir, cond, n_seeds)
        if ci is None:
            continue
        _shade(ax, steps / 1e6, ci, colors[cond], labels[cond])
    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.6)
    ax.axhline(0.0, color="grey",  lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("Training steps (×$10^6$)")
    ax.set_ylabel("Collusion Index")
    ax.set_ylim(-0.05, 1.30)
    ax.legend(loc="upper left", frameon=False,
              handlelength=1.3, columnspacing=0.6)
    fig.tight_layout()
    fig.savefig(out_pdf, **SAVEFIG_KW)
    fig.savefig(out_png, dpi=200, **SAVEFIG_KW)
    plt.close(fig)
    print(f"Saved {out_pdf} and {out_png}")


# ── Drivers ───────────────────────────────────────────────────────────────────

def make_cournot(run_dir, n_seeds=20):
    out = os.path.join(run_dir, "plots", "neurips")
    os.makedirs(out, exist_ok=True)
    _plot_ci(run_dir, n_seeds, COURNOT_ORDER, COURNOT_LABELS, COURNOT_COLORS,
             os.path.join(out, "ci_training.pdf"),
             os.path.join(out, "ci_training.png"))


def make_dqn(run_dir, n_seeds=20):
    out = os.path.join(run_dir, "plots", "neurips")
    os.makedirs(out, exist_ok=True)
    _plot_ci(run_dir, n_seeds, DQN_ORDER, DQN_LABELS, DQN_COLORS,
             os.path.join(out, "ci_training.pdf"),
             os.path.join(out, "ci_training.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--kind",    choices=["cournot", "dqn"], required=True)
    p.add_argument("--seeds",   type=int, default=20)
    args = p.parse_args()
    if args.kind == "cournot":
        make_cournot(args.run_dir, n_seeds=args.seeds)
    else:
        make_dqn(args.run_dir, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
