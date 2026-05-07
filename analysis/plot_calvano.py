"""
analysis/plot_calvano.py — Plots for Calvano CURB sweep.

Mirrors Market_env/tv_intervention/plot.py structure.

Figures
-------
1. ci_training.pdf     — CI over training, mean ± stderr, all conditions
2. price_training.pdf  — Mean price over training
3. tv_signal.pdf       — TV distance over training (CURB conditions)
4. summary_bar.pdf     — Final CI bar chart

Usage
-----
  python plot_calvano.py --run_dir ../results/calvano_n2/run_1 --n 2
  python plot_calvano.py --run_dir ../results/calvano_n2/run_1 --n 2 --seeds 50
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "baseline":     "#555555",
    "belief_only":  "#7B1FA2",
    "tv_only_l005": "#B0BEC5",
    "tv_only_l01":  "#78909C",
    "tv_only_l02":  "#546E7A",
    "tv_only_l05":  "#37474F",
    "full_l005":    "#90CAF9",
    "full_l01":     "#1976D2",
    "full_l02":     "#E91E63",
    "full_l05":     "#FF6F00",
}
LABELS = {
    "baseline":     "Baseline",
    "belief_only":  "Belief only",
    "tv_only_l005": "TV λ=0.05",
    "tv_only_l01":  "TV λ=0.10",
    "tv_only_l02":  "TV λ=0.20",
    "tv_only_l05":  "TV λ=0.50",
    "full_l005":    "CURB λ=0.05",
    "full_l01":     "CURB λ=0.10",
    "full_l02":     "CURB λ=0.20",
    "full_l05":     "CURB λ=0.50",
}
KEY_CONDITIONS = ["baseline", "full_l005", "full_l01", "full_l02", "full_l05"]


def _color(c):  return COLORS.get(c, "#AAAAAA")
def _label(c):  return LABELS.get(c, c)


# ── Time-series loading ───────────────────────────────────────────────────────

def _load_ts(run_dir, condition, seeds):
    records = {}
    for seed in seeds:
        path = os.path.join(run_dir, condition, f"seed_{seed}", "steps.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        records[seed] = df
    if not records:
        return None
    L   = min(len(df) for df in records.values())
    ref = next(iter(records.values()))
    steps = ref["step"].values[:L]
    ci    = np.stack([df["CI"].values[:L]      for df in records.values()])
    price_cols = [c for c in ref.columns if c.startswith("price_")]
    price = np.stack([df[price_cols].values[:L].mean(1) for df in records.values()])
    tv_cols = [c for c in ref.columns if c.startswith("tv")]
    tv = np.stack([df[tv_cols].values[:L].mean(1) for df in records.values()]) if tv_cols else None
    return {"steps": steps, "ci": ci, "price": price, "tv": tv}


def _shade(ax, steps, data, color, label, lw=1.5):
    m  = data.mean(0)
    se = data.std(0) / np.sqrt(data.shape[0])
    ax.plot(steps, m, color=color, label=label, linewidth=lw)
    ax.fill_between(steps, m - se, m + se, color=color, alpha=0.15)


# ── Individual plots ──────────────────────────────────────────────────────────

def plot_ci_training(run_dir, n, seeds, out_path, conditions=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    conds = conditions or list(COLORS.keys())
    for cond in conds:
        data = _load_ts(run_dir, cond, seeds)
        if data is None:
            continue
        _shade(ax, data["steps"], data["ci"], _color(cond), _label(cond))
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Monopoly")
    ax.axhline(0.0, color="grey",  lw=0.8, ls="--", label="Nash")
    ax.set_xlabel("Training steps"); ax.set_ylabel("Collusion Index (CI)")
    ax.set_title(f"Calvano n={n}: CI during training")
    ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.set_ylim(-0.1, 1.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


def plot_price_training(run_dir, n, seeds, out_path, conditions=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    conds = conditions or KEY_CONDITIONS
    for cond in conds:
        data = _load_ts(run_dir, cond, seeds)
        if data is None:
            continue
        _shade(ax, data["steps"], data["price"], _color(cond), _label(cond))
    ax.set_xlabel("Training steps"); ax.set_ylabel("Mean price")
    ax.set_title(f"Calvano n={n}: Mean price during training")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


def plot_tv_signal(run_dir, n, seeds, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    curb_conds = [c for c in COLORS if c.startswith("full_")]
    for cond in curb_conds:
        data = _load_ts(run_dir, cond, seeds)
        if data is None or data["tv"] is None:
            continue
        _shade(ax, data["steps"], data["tv"], _color(cond), _label(cond))
    ax.set_xlabel("Training steps"); ax.set_ylabel("TV distance")
    ax.set_title(f"Calvano n={n}: TV signal (CURB conditions)")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


def plot_summary_bar(sweep_csv, n, out_path):
    df   = pd.read_csv(sweep_csv)
    means = df.groupby("condition")["final_CI"].mean()
    sems  = df.groupby("condition")["final_CI"].sem()
    conds  = [c for c in list(COLORS.keys()) if c in means.index]
    values = [means[c] for c in conds]
    errors = [sems[c]  for c in conds]
    colors = [_color(c) for c in conds]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(conds)), values, yerr=errors, color=colors, capsize=4, width=0.6)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Monopoly")
    ax.axhline(0.0, color="grey",  lw=0.8, ls="--", label="Nash")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([_label(c) for c in conds], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Final Collusion Index"); ax.set_ylim(-0.05, 1.15)
    ax.set_title(f"Calvano n={n}: Final CI by Condition")
    ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


# ── Focused ablation plots ────────────────────────────────────────────────────

def plot_lambda_ablation(run_dir, n, seeds, out_path):
    """CI training: baseline vs full CURB at each λ — shows effect of penalty strength."""
    conds = ["baseline", "full_l005", "full_l01", "full_l02", "full_l05"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for cond in conds:
        data = _load_ts(run_dir, cond, seeds)
        if data is None:
            continue
        _shade(ax, data["steps"], data["ci"], _color(cond), _label(cond))
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Monopoly")
    ax.axhline(0.0, color="grey",  lw=0.8, ls="--", label="Nash")
    ax.set_xlabel("Training steps"); ax.set_ylabel("Collusion Index (CI)")
    ax.set_title(f"Calvano n={n}: CURB λ ablation")
    ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.set_ylim(-0.1, 1.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


def plot_component_ablation(run_dir, n, seeds, out_path, lam="l02"):
    """CI training: baseline, TV-only, belief-only, full CURB at one λ — shows component contributions."""
    conds = ["baseline", f"tv_only_{lam}", "belief_only", f"full_{lam}"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for cond in conds:
        data = _load_ts(run_dir, cond, seeds)
        if data is None:
            continue
        _shade(ax, data["steps"], data["ci"], _color(cond), _label(cond))
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Monopoly")
    ax.axhline(0.0, color="grey",  lw=0.8, ls="--", label="Nash")
    ax.set_xlabel("Training steps"); ax.set_ylabel("Collusion Index (CI)")
    lam_label = "0." + lam[1:] if lam.startswith("l0") else lam[1:]
    ax.set_title(f"Calvano n={n}: Component ablation (λ={lam_label})")
    ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.set_ylim(-0.1, 1.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved {out_path}")


# ── make_all_plots (called by sweep) ─────────────────────────────────────────

def make_all_plots(run_dir, n, n_seeds):
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    seeds       = list(range(n_seeds))
    sweep_csv   = os.path.join(run_dir, "sweep_results.csv")

    plot_ci_training(run_dir, n, seeds,    os.path.join(out_dir, "ci_training.png"))
    plot_price_training(run_dir, n, seeds, os.path.join(out_dir, "price_training.png"))
    plot_tv_signal(run_dir, n, seeds,      os.path.join(out_dir, "tv_signal.png"))
    plot_lambda_ablation(run_dir, n, seeds,    os.path.join(out_dir, "lambda_ablation.png"))
    plot_component_ablation(run_dir, n, seeds, os.path.join(out_dir, "component_ablation.png"))
    if os.path.exists(sweep_csv):
        plot_summary_bar(sweep_csv, n, os.path.join(out_dir, "summary_bar.png"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",  required=True)
    parser.add_argument("--n",        type=int, default=2)
    parser.add_argument("--seeds",    type=int, default=50)
    args = parser.parse_args()
    make_all_plots(args.run_dir, args.n, args.seeds)


if __name__ == "__main__":
    main()
