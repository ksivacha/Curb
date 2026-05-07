"""
analysis/plot_jsd_ablation.py — plots for the divergence-metric ablation.

Reads results/jsd_ablation/run_{N}/{condition}/seed_{S}/steps.csv and
sweep_results.csv. Produces:

  ci_training.png   — CI over training, all 10 conditions, mean ± SEM
  summary_bar.png   — final CI bar chart, grouped by λ × metric
  by_lambda.png     — three side-by-side panels (one per λ) comparing
                      TV / JSD-norm / JSD-raw CI trajectories

Usage
-----
  python plot_jsd_ablation.py --run_dir ../results/jsd_ablation/run_1
  python plot_jsd_ablation.py --run_dir ../results/jsd_ablation/run_1 --seeds 20
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Style ────────────────────────────────────────────────────────────────────

LAM_TAGS = ["l01", "l02", "l05"]                  # column / panel order
LAM_VALUES = {"l01": 0.1, "l02": 0.2, "l05": 0.5}

METRIC_COLORS = {
    "tv":       "#1976D2",   # blue
    "jsd_norm": "#E91E63",   # pink/red
    "jsd_raw":  "#FF9800",   # orange
}
METRIC_LABELS = {
    "tv":       "TV",
    "jsd_norm": "JSD (normalised)",
    "jsd_raw":  "JSD (raw nats)",
}
BASELINE_COLOR = "#444444"


def _condition_metric(name):
    """Map a condition name back to its metric family for plotting."""
    if name == "baseline":           return "baseline"
    if name.startswith("tv_full_"):  return "tv"
    if name.startswith("jsd_norm_"): return "jsd_norm"
    if name.startswith("jsd_raw_"):  return "jsd_raw"
    return None


def _condition_lam_tag(name):
    for t in LAM_TAGS:
        if name.endswith("_" + t):
            return t
    return None


# ── Time-series loading ──────────────────────────────────────────────────────

def _load_seeds(run_dir, condition, n_seeds):
    """Returns (steps, ci_array shape (seeds_found, T)) or (None, None)."""
    cis = []
    steps = None
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


def _shade(ax, x, arr, color, label, lw=1.5, ls="-"):
    m   = arr.mean(0)
    sem = arr.std(0) / np.sqrt(arr.shape[0])
    ax.plot(x, m, color=color, label=label, linewidth=lw, linestyle=ls)
    ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.15)


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_ci_training(run_dir, n_seeds, out_path):
    """All conditions on one plot. Different λ → different line styles."""
    fig, ax = plt.subplots(figsize=(10, 4.2))

    # baseline reference
    steps, ci = _load_seeds(run_dir, "baseline", n_seeds)
    if ci is not None:
        _shade(ax, steps, ci, BASELINE_COLOR, "baseline", lw=1.8, ls="-")

    # solid → l01, dashed → l02, dotted → l05
    lam_styles = {"l01": "-", "l02": "--", "l05": ":"}
    for metric in ["tv", "jsd_norm", "jsd_raw"]:
        for tag in LAM_TAGS:
            cond = (f"{metric}_full_{tag}" if metric == "tv"
                    else f"{metric}_full_{tag}")
            steps, ci = _load_seeds(run_dir, cond, n_seeds)
            if ci is None:
                continue
            _shade(ax, steps, ci,
                   METRIC_COLORS[metric],
                   f"{METRIC_LABELS[metric]} λ={LAM_VALUES[tag]}",
                   ls=lam_styles[tag])

    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0.0, color="grey",  lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Collusion Index")
    ax.set_title("CURB divergence ablation: CI over training (mean ± SEM)")
    ax.set_ylim(-0.1, 1.2)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


def plot_by_lambda(run_dir, n_seeds, out_path):
    """One panel per λ — direct TV vs JSD-norm vs JSD-raw comparison at matched λ."""
    fig, axes = plt.subplots(1, len(LAM_TAGS),
                             figsize=(4.0 * len(LAM_TAGS), 3.6),
                             sharey=True)
    for ax, tag in zip(axes, LAM_TAGS):
        # baseline reference on each panel
        steps, ci = _load_seeds(run_dir, "baseline", n_seeds)
        if ci is not None:
            _shade(ax, steps, ci, BASELINE_COLOR, "baseline", lw=1.4)
        for metric in ["tv", "jsd_norm", "jsd_raw"]:
            cond = f"{metric}_full_{tag}"
            steps, ci = _load_seeds(run_dir, cond, n_seeds)
            if ci is None:
                continue
            _shade(ax, steps, ci, METRIC_COLORS[metric],
                   METRIC_LABELS[metric])
        ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(0.0, color="grey",  lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("Training steps")
        ax.set_title(f"λ = {LAM_VALUES[tag]}")
        ax.set_ylim(-0.1, 1.2)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Collusion Index")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("CURB: TV vs JSD at matched λ", y=1.02)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_summary_bar(sweep_csv, out_path):
    """Grouped bar: x = λ, three bars per group (TV / JSD-norm / JSD-raw)."""
    df = pd.read_csv(sweep_csv)
    if df.empty:
        print(f"empty sweep csv: {sweep_csv}"); return

    # Aggregate baseline as a horizontal reference
    base_mean = df.loc[df.condition == "baseline", "final_CI"].mean()

    # Group: for each λ tag, get mean+sem per metric
    metrics = ["tv", "jsd_norm", "jsd_raw"]
    means = {m: [] for m in metrics}
    sems  = {m: [] for m in metrics}
    for tag in LAM_TAGS:
        for m in metrics:
            cond = f"{m}_full_{tag}"
            sub = df[df.condition == cond]["final_CI"]
            means[m].append(sub.mean()  if len(sub) else np.nan)
            sems [m].append(sub.sem()   if len(sub) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(LAM_TAGS))
    width = 0.27
    for k, m in enumerate(metrics):
        ax.bar(x + (k - 1) * width, means[m], width,
               yerr=sems[m], capsize=3,
               color=METRIC_COLORS[m], label=METRIC_LABELS[m])

    if not np.isnan(base_mean):
        ax.axhline(base_mean, color=BASELINE_COLOR, lw=1.2, ls="--",
                   label=f"baseline ({base_mean:.2f})")
    ax.axhline(0.0, color="grey",  lw=0.8, ls=":")
    ax.axhline(1.0, color="black", lw=0.8, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={LAM_VALUES[t]}" for t in LAM_TAGS])
    ax.set_ylabel("Final CI (mean ± SEM)")
    ax.set_title("CURB divergence ablation: final CI by λ × metric")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


# ── Driver ──────────────────────────────────────────────────────────────────

def make_all_plots(run_dir, n_seeds=20):
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_ci_training(run_dir, n_seeds, os.path.join(out_dir, "ci_training.png"))
    plot_by_lambda  (run_dir, n_seeds, os.path.join(out_dir, "by_lambda.png"))
    sweep_csv = os.path.join(run_dir, "sweep_results.csv")
    if os.path.exists(sweep_csv):
        plot_summary_bar(sweep_csv, os.path.join(out_dir, "summary_bar.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--seeds",   type=int, default=20)
    args = p.parse_args()
    make_all_plots(args.run_dir, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
