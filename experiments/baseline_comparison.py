"""
experiments/baseline_comparison.py — 4-condition baseline comparison.

Conditions
----------
  baseline   : train in standard Calvano, test in standard Calvano
  curb_l05   : train with CURBWrapper(lam=0.5), test in standard Calvano
  pdp        : train with BuyBoxWrapper(pdp, γ=1), test in standard Calvano
  dpdp       : train with BuyBoxWrapper(dpdp, γ=1, adv=0.3), test in standard Calvano

All conditions tested by running 100k greedy (ε=0) steps in base CalvanoBertrandEnv.
Q-tables transfer directly since state space (last joint price indices) is unchanged.

Output
------
results/baseline_comparison/run_N/
  comparison_results.csv   — per-seed: condition, seed, price_agent{i}, CI
  summary_table.txt        — mean ± std table (also printed to stdout)

Usage
-----
  python baseline_comparison.py --seeds 50 --workers 50
  python baseline_comparison.py --seeds 2 --smoke
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.calvano_bertrand import CalvanoBertrandEnv
from wrappers.curb import CURBWrapper
from wrappers.buybox import BuyBoxWrapper
from agents.tabular_q import TabularQAgent

TRAIN_STEPS = 3_000_000
TEST_STEPS  = 100_000


# ── Condition definitions ─────────────────────────────────────────────────────

CONDITIONS = [
    {"name": "baseline",  "train": "base",  "lam": 0.0, "dp_type": None,   "adv": None},
    {"name": "full_l02",  "train": "curb",  "lam": 0.2, "dp_type": None,   "adv": None},
    {"name": "full_l05",  "train": "curb",  "lam": 0.5, "dp_type": None,   "adv": None},
    {"name": "pdp",       "train": "buybox","lam": 0.0, "dp_type": "pdp",  "adv": None},
    {"name": "dpdp",      "train": "buybox","lam": 0.0, "dp_type": "dpdp", "adv": 0.3},
]


# ── Greedy test ───────────────────────────────────────────────────────────────

TRACE_STEPS = 500   # per-step price trace saved for visualisation


def _greedy_test(agent: TabularQAgent, n_steps: int = TEST_STEPS) -> dict:
    """
    Run greedy (ε=0) episode in a fresh standard CalvanoBertrandEnv.
    Returns mean_prices (full window), CI, and the first TRACE_STEPS prices
    per agent for step-level plotting.
    """
    test_env = CalvanoBertrandEnv(n=agent.n)
    obs, _   = test_env.reset()
    sk       = agent._state_key(obs)

    prices_log  = np.zeros((n_steps, agent.n))
    profits_log = np.zeros((n_steps, agent.n))

    for t in range(n_steps):
        actions = agent._greedy(sk)
        obs2, rewards, _, _, info = test_env.step(actions)
        prices_log[t]  = info.get("prices", test_env.price_grid[actions])
        profits_log[t] = rewards
        sk = agent._state_key(obs2)

    mean_prices = prices_log.mean(0)
    mean_pi     = profits_log.mean(0)
    ci          = test_env.collusion_index(mean_pi)

    return {
        "mean_prices": mean_prices,
        "ci":          ci,
        "price_trace": prices_log[:TRACE_STEPS].copy(),   # (TRACE_STEPS, n)
    }


# ── Per-seed worker ───────────────────────────────────────────────────────────

def _worker(args):
    cond, seed, n_steps = args
    np.random.seed(seed)

    agent_cfg = {"alpha": 0.15, "delta": 0.95, "beta": 4e-6,
                 "tstable": 100_000,
                 "use_belief": cond["train"] == "curb",
                 "K_belief":   100,
                 "M_belief":   5,
                 "seed":       seed}

    base_env = CalvanoBertrandEnv(n=2)

    if cond["train"] == "base":
        train_env = base_env
    elif cond["train"] == "curb":
        curb_cfg  = {"lam": cond["lam"], "tv_threshold": 0.0, "W": 500}
        train_env = CURBWrapper(base_env, curb_cfg)
    else:  # buybox
        bbx_cfg   = {"dp_type": cond["dp_type"], "gamma": 1.0}
        if cond["adv"] is not None:
            bbx_cfg["adv"] = cond["adv"]
        train_env = BuyBoxWrapper(base_env, bbx_cfg)

    agent = TabularQAgent(train_env, agent_cfg)
    train_r = agent.run(train_env, n_steps, log_every=50_000)

    result = _greedy_test(agent)
    result["train_steps"] = train_r["steps"]
    result["train_prices"] = train_r["prices"]
    result["train_ci"]    = train_r["ci"]

    print(f"  [{cond['name']:<10}] seed={seed:3d}  CI={result['ci']:.3f}  "
          f"p0={result['mean_prices'][0]:.4f}  p1={result['mean_prices'][1]:.4f}",
          flush=True)

    return cond, seed, result


# ── File saving ───────────────────────────────────────────────────────────────

def _save_seed(cond, seed, r, run_dir, n=2):
    """Write per-seed steps.csv with training-time price and CI trajectories,
    plus greedy_prices.csv with the first TRACE_STEPS of the greedy test."""
    seed_dir = os.path.join(run_dir, cond["name"], f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    steps  = r["train_steps"]
    prices = r["train_prices"]
    cis    = r["train_ci"]
    fields = ["step"] + [f"price_agent{i}" for i in range(n)] + ["CI"]
    with open(os.path.join(seed_dir, "steps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t_idx, step in enumerate(steps):
            row = {"step": int(step),
                   "CI":   round(float(cis[t_idx]), 6)}
            for i in range(n):
                row[f"price_agent{i}"] = round(float(prices[t_idx, i]), 6) \
                    if t_idx < len(prices) else 0.0
            w.writerow(row)

    # Greedy-test per-step prices for the seller-trajectory plots
    trace = r.get("price_trace")
    if trace is not None:
        with open(os.path.join(seed_dir, "greedy_prices.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + [f"price_agent{i}" for i in range(n)])
            for t in range(len(trace)):
                w.writerow([t] + [round(float(trace[t, i]), 6) for i in range(n)])


def _save_results(all_results, run_dir, n=2):
    fields = ["condition", "seed"] + [f"price_agent{i}" for i in range(n)] + ["CI"]
    path   = os.path.join(run_dir, "comparison_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cond, seed, r in all_results:
            row = {"condition": cond["name"], "seed": seed, "CI": round(r["ci"], 6)}
            for i in range(n):
                row[f"price_agent{i}"] = round(float(r["mean_prices"][i]), 6)
            w.writerow(row)
    print(f"Saved {path}")
    return path


def _print_table(all_results, n=2) -> str:
    from collections import defaultdict
    by_cond = defaultdict(lambda: {"prices": [], "ci": []})
    for cond, seed, r in all_results:
        by_cond[cond["name"]]["prices"].append(r["mean_prices"])
        by_cond[cond["name"]]["ci"].append(r["ci"])

    order = [c["name"] for c in CONDITIONS]
    p_headers = "  ".join(f"{'p'+str(i):>14}" for i in range(n))
    header = f"{'Condition':<12}  {p_headers}  {'CI':>14}"
    sep    = "-" * len(header)
    lines  = [header, sep]

    for name in order:
        d   = by_cond[name]
        ps  = np.array(d["prices"])   # (seeds, n)
        cis = np.array(d["ci"])
        row = f"{name:<12}"
        for i in range(n):
            row += f"  {ps[:,i].mean():6.4f}+/-{ps[:,i].std():6.4f}"
        row += f"  {cis.mean():6.4f}+/-{cis.std():6.4f}"
        lines.append(row)

    table = "\n".join(lines)
    print("\n" + table + "\n")
    return table


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_price_training(run_dir, n_seeds, n=2):
    """One subplot per condition; two lines per subplot (one per seller);
    mean across seeds with 1-sigma band. Reads per-seed steps.csv files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    conds_order = [c["name"] for c in CONDITIONS]

    # Reference prices for the Nash/monopoly lines.
    ref_env    = CalvanoBertrandEnv(n=n)
    nash_price = float(ref_env.nash_price)
    mono_price = float(ref_env.mono_price)

    ncols = min(3, len(conds_order))
    nrows = (len(conds_order) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.2 * nrows),
                             sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    colors = ["#1976D2", "#E91E63", "#FF9800", "#4CAF50"][:n]

    for idx, cond_name in enumerate(conds_order):
        ax = axes[idx // ncols, idx % ncols]
        cond_dir = os.path.join(run_dir, cond_name)
        if not os.path.isdir(cond_dir):
            ax.set_visible(False)
            continue

        series = {i: [] for i in range(n)}
        steps  = None
        for seed in range(n_seeds):
            path = os.path.join(cond_dir, f"seed_{seed}", "steps.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if steps is None:
                steps = df["step"].values
            L = min(len(df), len(steps))
            for i in range(n):
                series[i].append(df[f"price_agent{i}"].values[:L])

        if steps is None or not series[0]:
            ax.set_visible(False)
            continue

        L = min(len(series[0][0]), len(steps))
        steps = steps[:L]

        for i in range(n):
            arr = np.stack([s[:L] for s in series[i]])
            m   = arr.mean(0)
            s   = arr.std(0)
            ax.plot(steps / 1e6, m, color=colors[i], lw=1.5,
                    label=f"Seller {i}")
            ax.fill_between(steps / 1e6, m - s, m + s,
                            color=colors[i], alpha=0.15)

        ax.axhline(nash_price, color="#888", ls="--", lw=0.8, label="Nash")
        ax.axhline(mono_price, color="#444", ls=":",  lw=0.8, label="Monopoly")
        ax.set_title(cond_name)
        ax.set_xlabel("Training steps (M)")
        ax.grid(axis="y", alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("Price")
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    # Hide unused cells.
    for j in range(len(conds_order), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "price_training.png"), dpi=150)
    plt.close(fig)


def _plot_greedy_seller_traces(run_dir, n_seeds, n=2):
    """
    Two figures, one per seller. Each shows price step-by-step (greedy ε=0
    rollout in standard CalvanoBertrandEnv) with a line per condition.
    Mean across seeds, ±1σ band. Reads per-seed greedy_prices.csv.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    conds_order = [c["name"] for c in CONDITIONS]
    cond_colors = {"baseline": "#444444",
                   "full_l02": "#26A69A",
                   "full_l05": "#1976D2",
                   "pdp":      "#E91E63",
                   "dpdp":     "#FF9800"}

    ref_env    = CalvanoBertrandEnv(n=n)
    nash_price = float(ref_env.nash_price)
    mono_price = float(ref_env.mono_price)

    # Aggregate price traces per condition, per seller
    traces = {c: {i: [] for i in range(n)} for c in conds_order}
    n_steps = None
    for cond_name in conds_order:
        cond_dir = os.path.join(run_dir, cond_name)
        if not os.path.isdir(cond_dir):
            continue
        for seed in range(n_seeds):
            path = os.path.join(cond_dir, f"seed_{seed}", "greedy_prices.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if n_steps is None:
                n_steps = len(df)
            for i in range(n):
                traces[cond_name][i].append(df[f"price_agent{i}"].values[:n_steps])

    if n_steps is None:
        print("[plot_greedy_seller_traces] no greedy_prices.csv found, skipping")
        return

    steps = np.arange(n_steps)

    for seller in range(n):
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        for cond_name in conds_order:
            seed_runs = traces[cond_name][seller]
            if not seed_runs:
                continue
            arr = np.stack(seed_runs)              # (n_seeds, n_steps)
            m   = arr.mean(0)
            s   = arr.std(0)
            color = cond_colors.get(cond_name, "#777777")
            ax.plot(steps, m, color=color, lw=1.6, label=cond_name)
            ax.fill_between(steps, m - s, m + s, color=color, alpha=0.15)
        ax.axhline(nash_price, color="#888", ls="--", lw=0.8, label="Nash")
        ax.axhline(mono_price, color="#444", ls=":",  lw=0.8, label="Monopoly")
        ax.set_xlabel("Greedy test step")
        ax.set_ylabel(f"Price (Seller {seller})")
        ax.set_title(f"Seller {seller}: greedy-rollout prices, mean±σ across seeds")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"greedy_seller{seller}.png"), dpi=150)
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    type=int, default=50)
    parser.add_argument("--workers",  type=int, default=50)
    parser.add_argument("--steps",    type=int, default=None)
    parser.add_argument("--smoke",    action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds   = 2
        args.workers = 2
        args.steps   = 50_000

    n_steps = args.steps or TRAIN_STEPS

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "baseline_comparison"
    )
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(d.split("_")[1]) for d in os.listdir(base_dir)
                if d.startswith("run_") and d.split("_")[1].isdigit()]
    run_num = max(existing, default=0) + 1
    run_dir = os.path.join(base_dir, f"run_{run_num}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving to: {run_dir}\n")

    tasks = [(cond, seed, n_steps)
             for cond in CONDITIONS
             for seed in range(args.seeds)]

    print(f"Conditions: {len(CONDITIONS)}  Seeds: {args.seeds}  "
          f"Total: {len(tasks)}  Workers: {args.workers}  Train steps: {n_steps:,}\n")

    t0 = time.time()
    all_results = []
    done = 0
    try:
        with mp.Pool(min(args.workers, len(tasks))) as pool:
            for cond, seed, r in pool.imap_unordered(_worker, tasks):
                _save_seed(cond, seed, r, run_dir)
                all_results.append((cond, seed, r))
                done += 1
                if done % 5 == 0 or done == len(tasks):
                    print(f"  [progress] {done}/{len(tasks)} seeds saved "
                          f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
    except KeyboardInterrupt:
        print(f"\n[interrupted] {done}/{len(tasks)} seeds saved to {run_dir}", flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")

    if all_results:
        _save_results(all_results, run_dir)
        table = _print_table(all_results)
        with open(os.path.join(run_dir, "summary_table.txt"), "w") as f:
            f.write(table + "\n")
        print(f"Summary table saved to {run_dir}/summary_table.txt")

        try:
            _plot_price_training(run_dir, args.seeds)
            _plot_greedy_seller_traces(run_dir, args.seeds)
            print(f"Plots saved to {run_dir}/plots/")
        except Exception as e:
            print(f"[plot] {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
