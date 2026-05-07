"""
analysis/plot_neurips_jsd.py — paper-ready summary-bar plot for the
JSD-ablation experiment (TV vs JSD-norm vs JSD-raw at matched λ).

Output: <run_dir>/plots/neurips/summary_bar.{pdf,png}
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# NeurIPS-style rc tweaks (same as plot_neurips_calvano)
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

LAM_TAGS   = ["l01", "l02", "l05"]
LAM_VALUES = {"l01": 0.1, "l02": 0.2, "l05": 0.5}

METRIC_COLORS = {
    "tv":       "#1976D2",
    "jsd_norm": "#E91E63",
    "jsd_raw":  "#FF9800",
}
METRIC_LABELS = {
    "tv":       "TV",
    "jsd_norm": "JSD (norm.)",
    "jsd_raw":  "JSD (raw)",
}
BASELINE_COLOR = "#444444"


def plot_summary_bar(sweep_csv, out_path_pdf, out_path_png):
    df = pd.read_csv(sweep_csv)
    if df.empty:
        print(f"empty sweep csv: {sweep_csv}"); return

    base_mean = df.loc[df.condition == "baseline", "final_CI"].mean()

    metrics = ["tv", "jsd_norm", "jsd_raw"]
    means = {m: [] for m in metrics}
    sems  = {m: [] for m in metrics}
    for tag in LAM_TAGS:
        for m in metrics:
            cond = f"{m}_full_{tag}"
            sub  = df[df.condition == cond]["final_CI"]
            means[m].append(sub.mean()  if len(sub) else np.nan)
            sems [m].append(sub.sem()   if len(sub) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(LAM_TAGS))
    width = 0.27
    for k, m in enumerate(metrics):
        ax.bar(x + (k - 1) * width, means[m], width,
               yerr=sems[m], capsize=2.5,
               color=METRIC_COLORS[m],
               edgecolor="white", linewidth=0.4,
               label=METRIC_LABELS[m])

    if not np.isnan(base_mean):
        ax.axhline(base_mean, color=BASELINE_COLOR, lw=1.0, ls="--",
                   label=f"No intervention ({base_mean:.2f})")
    ax.axhline(0.0, color="grey",  lw=0.6, ls=":")
    ax.axhline(1.0, color="black", lw=0.6, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\lambda{{=}}{LAM_VALUES[t]}$" for t in LAM_TAGS])
    ax.set_ylabel("Final CI (mean $\\pm$ SEM)")
    ax.set_ylim(-0.05, 1.20)
    ax.legend(loc="upper right", frameon=False,
              handlelength=1.3, columnspacing=0.6, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path_pdf, **SAVEFIG_KW)
    fig.savefig(out_path_png, dpi=200, **SAVEFIG_KW)
    plt.close(fig)
    print(f"Saved {out_path_pdf} and {out_path_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()
    out_dir = os.path.join(args.run_dir, "plots", "neurips")
    os.makedirs(out_dir, exist_ok=True)
    plot_summary_bar(
        os.path.join(args.run_dir, "sweep_results.csv"),
        os.path.join(out_dir, "summary_bar.pdf"),
        os.path.join(out_dir, "summary_bar.png"),
    )


if __name__ == "__main__":
    main()
