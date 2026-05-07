"""
analysis/plot_forced_defection.py — plots for the forced-defection experiment.

Reads results/forced_defection/run_{N}/<condition>/seed_{S}/defection_prices.csv
and produces:

  forced_defection.png     — two side-by-side panels (baseline vs CURB).
                             Each shows mean ± SEM agent prices over 200 steps,
                             with the forced-defection step marked.
  forced_defection_overlay.png
                            — same data, all four lines overlaid on one axis
                              (4 lines = {baseline, curb} × {agent0, agent1}).

Conventions
-----------
  Agent 0 = forced defector (red shades)
  Agent 1 = responder (blue shades)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure the project root is on sys.path so we can import envs.* when this
# script is invoked directly (as opposed to via train experiment).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.calvano_bertrand import CalvanoBertrandEnv

CONDITIONS = ["baseline", "tv_only_l05"]
COND_LABELS = {
    "baseline":    "Baseline",
    "tv_only_l05": "TV-only (λ=0.5)",
}

# colors per (condition, agent)
LINE_COLORS = {
    ("baseline",    0): "#D32F2F",   # red       (defector under baseline)
    ("baseline",    1): "#1976D2",   # blue      (responder under baseline)
    ("tv_only_l05", 0): "#F8BBD0",   # pink      (defector under TV-only)
    ("tv_only_l05", 1): "#90CAF9",   # light blue (responder under TV-only)
}

DEFECT_STEP = 50
SMOOTH_WINDOW = 5         # moving-average window applied to each seed trace
ZOOM_RANGE    = (40, 90)  # x-axis limits for paper-style zoom around defection


# ── Load ────────────────────────────────────────────────────────────────────

def _load_defection_prices(run_dir, condition, n_seeds):
    """Returns array shape (seeds_found, T, 2) with [step, agent0, agent1]."""
    out = []
    for seed in range(n_seeds):
        p = os.path.join(run_dir, condition, f"seed_{seed}",
                         "defection_prices.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        out.append(df[["price_agent0", "price_agent1"]].values)
    if not out:
        return None
    L = min(len(a) for a in out)
    return np.stack([a[:L] for a in out])


def _smooth(arr1d: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-average smoothing along 1D. Edges shrink the window."""
    if window <= 1 or len(arr1d) < window:
        return arr1d
    kernel = np.ones(window) / window
    out = np.convolve(arr1d, kernel, mode="same")
    # Fix edge artifacts where convolution sums fewer-than-window values
    half = window // 2
    for i in range(half):
        out[i] = arr1d[: i + half + 1].mean()
        out[-1 - i] = arr1d[-(i + half + 1):].mean()
    return out


def _shade(ax, x, arr, color, label, lw=1.6, ls="-"):
    """arr shape (n_seeds, T). Smooths per-seed, then plots mean ± SEM."""
    smoothed = np.stack([_smooth(arr[s], SMOOTH_WINDOW)
                         for s in range(arr.shape[0])])
    m   = smoothed.mean(0)
    sem = smoothed.std(0) / np.sqrt(smoothed.shape[0])
    ax.plot(x, m, color=color, label=label, linewidth=lw, linestyle=ls)
    ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.18)


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_two_panel(run_dir, n_seeds, out_path):
    ref_env = CalvanoBertrandEnv(n=2)
    nash_p, mono_p = float(ref_env.nash_price), float(ref_env.mono_price)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        prices = _load_defection_prices(run_dir, cond, n_seeds)
        if prices is None:
            ax.set_visible(False)
            continue
        t = np.arange(prices.shape[1])
        # agent 0 (forced defector)
        _shade(ax, t, prices[:, :, 0],
               LINE_COLORS[(cond, 0)], "Agent 0 (defector)", lw=1.6)
        # agent 1 (responder)
        _shade(ax, t, prices[:, :, 1],
               LINE_COLORS[(cond, 1)], "Agent 1 (responder)", lw=1.6)

        ax.axvline(DEFECT_STEP, color="grey", lw=0.8, ls="--", alpha=0.7,
                   label=f"forced defection (t={DEFECT_STEP})")
        ax.axhline(nash_p, color="#444", lw=0.7, ls=":",  label="Nash price")
        ax.axhline(mono_p, color="black", lw=0.7, ls="--", label="Mono price")
        ax.set_title(COND_LABELS.get(cond, cond))
        ax.set_xlabel("Step")
        ax.set_xlim(*ZOOM_RANGE)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Forced-defection response (mean ± SEM across seeds)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_overlay(run_dir, n_seeds, out_path):
    ref_env = CalvanoBertrandEnv(n=2)
    nash_p, mono_p = float(ref_env.nash_price), float(ref_env.mono_price)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for cond in CONDITIONS:
        prices = _load_defection_prices(run_dir, cond, n_seeds)
        if prices is None:
            continue
        t = np.arange(prices.shape[1])
        ls = "-" if cond == "baseline" else "--"
        _shade(ax, t, prices[:, :, 0],
               LINE_COLORS[(cond, 0)],
               f"{COND_LABELS[cond]} — Agent 0", ls=ls)
        _shade(ax, t, prices[:, :, 1],
               LINE_COLORS[(cond, 1)],
               f"{COND_LABELS[cond]} — Agent 1", ls=ls)

    ax.axvline(DEFECT_STEP, color="grey", lw=0.8, ls="--", alpha=0.7,
               label=f"forced defection (t={DEFECT_STEP})")
    ax.axhline(nash_p, color="#444", lw=0.7, ls=":",  label="Nash price")
    ax.axhline(mono_p, color="black", lw=0.7, ls="--", label="Mono price")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.set_title("Forced-defection response: baseline vs TV-only")
    ax.set_xlim(*ZOOM_RANGE)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Driver ──────────────────────────────────────────────────────────────────

def make_all_plots(run_dir, n_seeds=20):
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_two_panel(run_dir, n_seeds, os.path.join(out_dir, "forced_defection.png"))
    plot_overlay  (run_dir, n_seeds, os.path.join(out_dir, "forced_defection_overlay.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--seeds",   type=int, default=20)
    args = p.parse_args()
    make_all_plots(args.run_dir, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
