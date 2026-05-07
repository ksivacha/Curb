"""
analysis/plot_neurips_forced_defection.py — paper-ready forced-defection plot.

Two side-by-side panels (baseline vs TV-only) showing per-agent prices
around step 50 (forced defection event), smoothed and zoomed for clarity.

Output: <run_dir>/plots/neurips/forced_defection.{pdf,png}
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.calvano_bertrand import CalvanoBertrandEnv

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

FIGSIZE        = (7.8, 2.8)         # two panels side-by-side ≈ full text width
SAVEFIG_KW     = dict(bbox_inches="tight", pad_inches=0.03)
DEFECT_STEP    = 50
SMOOTH_WINDOW  = 5
ZOOM_RANGE     = (40, 90)

CONDITIONS = ["baseline", "tv_only_l05"]
LABELS = {"baseline": "No intervention", "tv_only_l05": "TV-only ($\\lambda{=}0.5$)"}
COLORS = {
    ("baseline",    0): "#D32F2F",
    ("baseline",    1): "#1976D2",
    ("tv_only_l05", 0): "#D32F2F",
    ("tv_only_l05", 1): "#1976D2",
}


def _smooth(arr1d, w):
    """Causal (right-anchored) moving average — value at t uses only t-w+1..t.
    Avoids leaking events forward in time visually (e.g. forced defection at t=50
    should not affect the plotted curve at t<50)."""
    if w <= 1 or len(arr1d) < w:
        return arr1d
    out = np.empty_like(arr1d, dtype=float)
    for i in range(len(arr1d)):
        lo = max(0, i - w + 1)
        out[i] = arr1d[lo : i + 1].mean()
    return out


def _shade(ax, x, arr, color, label, lw=1.4):
    s   = np.stack([_smooth(arr[i], SMOOTH_WINDOW) for i in range(arr.shape[0])])
    m   = s.mean(0); sem = s.std(0) / np.sqrt(s.shape[0])
    ax.plot(x, m, color=color, label=label, linewidth=lw)
    ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.18)


def _load(run_dir, condition, n_seeds):
    out = []
    for seed in range(n_seeds):
        p = os.path.join(run_dir, condition, f"seed_{seed}",
                         "defection_prices.csv")
        if not os.path.exists(p):
            continue
        out.append(pd.read_csv(p)[["price_agent0", "price_agent1"]].values)
    if not out:
        return None
    L = min(len(a) for a in out)
    return np.stack([a[:L] for a in out])


def make_plot(run_dir, n_seeds, out_pdf, out_png):
    ref         = CalvanoBertrandEnv(n=2)
    nash, mono  = float(ref.nash_price), float(ref.mono_price)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        prices = _load(run_dir, cond, n_seeds)
        if prices is None:
            ax.set_visible(False); continue
        t = np.arange(prices.shape[1])
        _shade(ax, t, prices[:, :, 0], COLORS[(cond, 0)], "Agent 0 (defector)")
        _shade(ax, t, prices[:, :, 1], COLORS[(cond, 1)], "Agent 1 (responder)")
        ax.axvline(DEFECT_STEP, color="grey", lw=0.7, ls="--", alpha=0.7)
        ax.axhline(nash, color="#444",  lw=0.6, ls=":")
        ax.axhline(mono, color="black", lw=0.6, ls="--")
        ax.set_xlim(*ZOOM_RANGE)
        ax.set_xlabel("Step")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(LABELS[cond])

    axes[0].set_ylabel("Price")
    axes[0].legend(loc="lower right", frameon=False, handlelength=1.3)
    fig.tight_layout()
    fig.savefig(out_pdf, **SAVEFIG_KW)
    fig.savefig(out_png, dpi=200, **SAVEFIG_KW)
    plt.close(fig)
    print(f"Saved {out_pdf} and {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--seeds",   type=int, default=20)
    args = p.parse_args()
    out = os.path.join(args.run_dir, "plots", "neurips")
    os.makedirs(out, exist_ok=True)
    make_plot(args.run_dir, args.seeds,
              os.path.join(out, "forced_defection.pdf"),
              os.path.join(out, "forced_defection.png"))


if __name__ == "__main__":
    main()
