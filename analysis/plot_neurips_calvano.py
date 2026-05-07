"""
analysis/plot_neurips_calvano.py — paper-ready versions of the lambda and
component ablation plots from plot_calvano.py.

Produces compact, NeurIPS-aligned subfigure-style plots for inclusion as
two-column \\subfigure[]{...} or two side-by-side \\begin{figure}{...} in
a NeurIPS LaTeX document.

Output: <run_dir>/plots/neurips/{lambda_ablation,component_ablation}.{pdf,png}

Typical LaTeX usage:
    \\begin{figure}[t]
      \\centering
      \\begin{subfigure}{0.49\\textwidth}
        \\includegraphics[width=\\linewidth]{lambda_ablation.pdf}
        \\caption{}
      \\end{subfigure}\\hfill
      \\begin{subfigure}{0.49\\textwidth}
        \\includegraphics[width=\\linewidth]{component_ablation.pdf}
        \\caption{}
      \\end{subfigure}
    \\end{figure}
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_calvano import _load_ts, COLORS, LABELS as _LABELS

# Local label overrides for paper-ready figures
LABELS = dict(_LABELS)
LABELS["baseline"] = "No intervention"

# ── NeurIPS-style global rc tweaks ───────────────────────────────────────────

plt.rcParams.update({
    "font.size":        9,        # readable when scaled to ~2.7" in LaTeX
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

FIGSIZE = (3.9, 2.7)   # ~7:5 aspect, fits 0.49\textwidth nicely
SAVEFIG_KW = dict(bbox_inches="tight", pad_inches=0.03)


def _color(c): return COLORS.get(c, "#AAAAAA")
def _label(c): return LABELS.get(c, c)


def _shade(ax, steps, data, color, label, lw=1.4):
    m  = data.mean(0)
    se = data.std(0) / np.sqrt(data.shape[0])
    ax.plot(steps / 1e6, m, color=color, label=label, linewidth=lw)
    ax.fill_between(steps / 1e6, m - se, m + se, color=color, alpha=0.18)


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_lambda_ablation(run_dir, n, seeds, out_path_pdf, out_path_png):
    conds = ["baseline", "full_l005", "full_l01", "full_l02", "full_l05"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cond in conds:
        data = _load_ts(run_dir, cond, seeds)
        if data is None:
            continue
        _shade(ax, data["steps"], data["ci"], _color(cond), _label(cond))

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.6)
    ax.axhline(0.0, color="grey",  lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("Training steps (×$10^6$)")
    ax.set_ylabel("Collusion Index")
    ax.set_ylim(-0.05, 1.30)
    ax.legend(loc="upper left", frameon=False, ncol=1,
              handlelength=1.3, columnspacing=0.6)
    fig.tight_layout()
    fig.savefig(out_path_pdf, **SAVEFIG_KW)
    fig.savefig(out_path_png, dpi=200, **SAVEFIG_KW)
    plt.close(fig)
    print(f"Saved {out_path_pdf} and {out_path_png}")


def plot_component_ablation(run_dir, n, seeds, out_path_pdf, out_path_png,
                            lam="l02"):
    """λ tag like 'l02' → λ=0.2; 'l05' → λ=0.5."""
    conds = ["baseline", f"tv_only_{lam}", "belief_only", f"full_{lam}"]
    # Reverse `str(lam).replace(".", "")`: "l02" → "0.2", "l005" → "0.05".
    rest    = lam[1:] if lam.startswith("l") else lam
    lam_val = float(rest[0] + "." + rest[1:])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cond in conds:
        data = _load_ts(run_dir, cond, seeds)
        if data is None:
            continue
        _shade(ax, data["steps"], data["ci"], _color(cond), _label(cond))

    ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.6)
    ax.axhline(0.0, color="grey",  lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("Training steps (×$10^6$)")
    ax.set_ylabel("Collusion Index")
    ax.set_ylim(-0.05, 1.30)
    ax.legend(loc="upper left", frameon=False, ncol=1,
              handlelength=1.3, columnspacing=0.6)
    fig.tight_layout()
    fig.savefig(out_path_pdf, **SAVEFIG_KW)
    fig.savefig(out_path_png, dpi=200, **SAVEFIG_KW)
    plt.close(fig)
    print(f"Saved {out_path_pdf} and {out_path_png}")


# ── Driver ──────────────────────────────────────────────────────────────────

def make_neurips_plots(run_dir, n=2, seeds=20, lam="l02"):
    out_dir = os.path.join(run_dir, "plots", "neurips")
    os.makedirs(out_dir, exist_ok=True)
    seed_iter = range(seeds)

    plot_lambda_ablation(
        run_dir, n, seed_iter,
        os.path.join(out_dir, "lambda_ablation.pdf"),
        os.path.join(out_dir, "lambda_ablation.png"),
    )
    plot_component_ablation(
        run_dir, n, seed_iter,
        os.path.join(out_dir, "component_ablation.pdf"),
        os.path.join(out_dir, "component_ablation.png"),
        lam=lam,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--n",       type=int, default=2)
    p.add_argument("--seeds",   type=int, default=20)
    p.add_argument("--lam",     default="l02",
                   help="λ tag for component ablation (e.g. l02, l05)")
    args = p.parse_args()
    make_neurips_plots(args.run_dir, n=args.n, seeds=args.seeds, lam=args.lam)


if __name__ == "__main__":
    main()
