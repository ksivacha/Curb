"""
experiments/dqn_calvano_sweep.py — DQN sweep: baseline vs CURB (λ sweep).

Conditions
----------
  baseline_dqn   : DQN in standard CalvanoBertrandEnv
  curb_l01_dqn   : DQN + CURBWrapper(λ=0.1)
  curb_l02_dqn   : DQN + CURBWrapper(λ=0.2)
  curb_l05_dqn   : DQN + CURBWrapper(λ=0.5)

Output mirrors calvano_sweep.py so plot_calvano.py can be reused directly.

results/dqn_calvano/run_N/
  sweep_results.csv
  plots/
  <condition>/seed_<s>/steps.csv  sessions.csv

Usage
-----
  python dqn_calvano_sweep.py --seeds 20 --workers 20
  python dqn_calvano_sweep.py --seeds 2 --smoke
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.calvano_bertrand import CalvanoBertrandEnv
from wrappers.curb import CURBWrapper
from agents.dqn_agent import MultiAgentDQN

LAM_VALUES = [0.1, 0.2, 0.5]
TRAIN_STEPS = 2_000_000


# ── Conditions ────────────────────────────────────────────────────────────────

def all_conditions():
    conds = [{"name": "baseline_dqn", "lam": 0.0, "use_curb": False, "use_belief": False}]
    for lam in LAM_VALUES:
        tag = str(lam).replace(".", "")
        conds.append({"name": f"curb_l{tag}_dqn", "lam": lam, "use_curb": True, "use_belief": True})
    return conds


# ── Per-seed worker ───────────────────────────────────────────────────────────

def _worker(args):
    import torch
    torch.set_num_threads(1)

    n, cond, seed, n_steps, log_every = args
    np.random.seed(seed)

    curb_cfg = {"lam": cond["lam"], "tv_threshold": 0.0, "W": 500}
    dqn_cfg  = {"lr": 0.001, "buf_size": 10_000, "batch": 32,
                "gamma": 0.95, "C": 100,
                "use_belief": cond["use_belief"], "K_belief": 100, "M_belief": 5,
                "seed": seed}

    base_env = CalvanoBertrandEnv(n=n)
    env      = CURBWrapper(base_env, curb_cfg) if cond["use_curb"] else base_env
    agent    = MultiAgentDQN(env, dqn_cfg)
    r        = agent.run(env, n_steps, log_every)

    print(f"  [{cond['name']:<20}] seed={seed:3d}  CI={r['final_ci']:.3f}", flush=True)
    return cond, seed, r


# ── File saving (mirrors calvano_sweep._save_seed) ────────────────────────────

def _save_seed(cond, seed, r, run_dir, n):
    seed_dir = os.path.join(run_dir, cond["name"], f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    p_cols  = [f"price_agent{i}" for i in range(n)]
    tv_cols = [f"tv{i}" for i in range(n)]
    with open(os.path.join(seed_dir, "steps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step"] + p_cols + ["CI"] + tv_cols)
        w.writeheader()
        prices = r["prices"]
        tv     = r["tv"]
        for t_idx, step in enumerate(r["steps"]):
            row = {"step": int(step),
                   "CI":   round(float(r["ci"][t_idx]), 6)}
            for i in range(n):
                row[f"price_agent{i}"] = round(float(prices[t_idx, i]), 6) if t_idx < len(prices) else 0.0
                row[f"tv{i}"]          = round(float(tv[t_idx, i]),     6) if t_idx < len(tv)     else 0.0
            w.writerow(row)

    with open(os.path.join(seed_dir, "sessions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "final_CI", "converged", "t_conv"])
        w.writeheader()
        w.writerow({"seed":     seed,
                    "final_CI": round(r["final_ci"], 4),
                    "converged": r["converged"],
                    "t_conv":    r["t_conv"]})


def _save_sweep(all_results, run_dir):
    with open(os.path.join(run_dir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "lam", "use_curb",
                                          "seed", "final_CI", "converged", "t_conv"])
        w.writeheader()
        for cond, seed, r in all_results:
            w.writerow({"condition": cond["name"],
                        "lam":       cond["lam"],
                        "use_curb":  cond["use_curb"],
                        "seed":      seed,
                        "final_CI":  round(r["final_ci"], 4),
                        "converged": r["converged"],
                        "t_conv":    r["t_conv"]})


def _print_summary(all_results):
    by_cond = defaultdict(list)
    for cond, seed, r in all_results:
        by_cond[cond["name"]].append(r["final_ci"])

    base_ci = np.mean(by_cond.get("baseline_dqn", [0]))
    print(f"\n{'Condition':<24} {'CI':>6} {'+/-':>5}  {'vs base':>8}")
    print("-" * 48)
    for name in ["baseline_dqn"] + sorted(
            [n for n in by_cond if n != "baseline_dqn"],
            key=lambda x: np.mean(by_cond[x])):
        cis   = by_cond[name]
        delta = np.mean(cis) - base_ci
        tag   = "--" if name == "baseline_dqn" else f"{'v' if delta < 0 else '^'}{abs(delta):.3f}"
        print(f"  {name:<22} {np.mean(cis):>6.3f} {np.std(cis):>5.3f}  {tag:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    type=int, default=20)
    parser.add_argument("--workers",  type=int, default=20)
    parser.add_argument("--steps",    type=int, default=None)
    parser.add_argument("--log_every",type=int, default=10_000)
    parser.add_argument("--smoke",    action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds     = 2
        args.workers   = 2
        args.steps     = 50_000
        args.log_every = 5_000

    n_steps = args.steps or TRAIN_STEPS
    n       = 2

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "dqn_calvano"
    )
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(d.split("_")[1]) for d in os.listdir(base_dir)
                if d.startswith("run_") and d.split("_")[1].isdigit()]
    run_num = max(existing, default=0) + 1
    run_dir = os.path.join(base_dir, f"run_{run_num}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving to: {run_dir}\n")

    conds = all_conditions()

    # Save run config for provenance
    import json
    cfg_dump = {
        "n":            n,
        "n_steps":      n_steps,
        "seeds":        args.seeds,
        "workers":      args.workers,
        "log_every":    args.log_every,
        # DQN hyperparameters (matches dqn_cfg in _worker)
        "lr":           1e-3,
        "buf_size":     10_000,
        "batch":        32,
        "gamma":        0.95,
        "C":            100,
        "beta":         4e-6,
        "hidden":       32,
        "K_belief":     100,
        "M_belief":     5,
        "K_prices":     15,
        "W_tv":         500,
        "tv_threshold": 0.0,
        "tv_lag":       1,
        "conditions":   [c["name"] for c in conds],
        "lam_values":   sorted({c["lam"] for c in conds}),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg_dump, f, indent=2)

    tasks = [(n, cond, seed, n_steps, args.log_every)
             for cond in conds
             for seed in range(args.seeds)]

    print(f"Conditions: {len(conds)}  Seeds: {args.seeds}  "
          f"Total: {len(tasks)}  Workers: {args.workers}  Steps: {n_steps:,}\n")

    t0 = time.time()
    all_results = []
    done = 0
    try:
        with mp.Pool(min(args.workers, len(tasks))) as pool:
            for cond, seed, r in pool.imap_unordered(_worker, tasks):
                _save_seed(cond, seed, r, run_dir, n)
                all_results.append((cond, seed, r))
                done += 1
                if done % 5 == 0 or done == len(tasks):
                    print(f"  [progress] {done}/{len(tasks)} seeds saved "
                          f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
    except KeyboardInterrupt:
        print(f"\n[interrupted] {done}/{len(tasks)} seeds saved to {run_dir}", flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")

    if all_results:
        _save_sweep(all_results, run_dir)
        _print_summary(all_results)

    # Plots — reuse plot_calvano with DQN-specific color/label overrides
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
        from plot_dqn_calvano import make_all_plots
        make_all_plots(run_dir, n, args.seeds)
        print(f"\nPlots saved to {run_dir}/plots/")
    except Exception as e:
        print(f"[plot] {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
