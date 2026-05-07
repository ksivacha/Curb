"""
experiments/cournot_sweep.py — Cournot CURB sweep for n=2.

Conditions
----------
  baseline               : perfect monitoring, deterministic demand, no intervention
  imperfect_monitoring   : stochastic demand d∈{290,310}, state = p_{t-1} (37 levels),
                           no CURB — replicates Calvano et al. (2021) IM baseline
  full_l01               : perfect monitoring + λ=0.1 CURB (TV + belief)
  full_l02               : perfect monitoring + λ=0.2 CURB (TV + belief)
  full_l05               : perfect monitoring + λ=0.5 CURB (TV + belief)

Both Cournot variants share the K=15 quantity grid {70..105} and the same
α=0.15, δ=0.95, β=4e-6 hyperparameters. The IM env exposes state_shape=(37,)
so the Q-table is 37×15 instead of 15×15×15.

Output
------
results/cournot_n{N}/run_{K}/
  config.json
  sweep_results.csv
  <condition>/seed_<s>/{steps.csv, sessions.csv}

Usage
-----
  python cournot_sweep.py --n 2 --seeds 20 --workers 32
  python cournot_sweep.py --n 2 --seeds 2 --smoke
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

from envs.cournot           import CournotEnv
from envs.cournot_imperfect import CournotImperfectEnv
from wrappers.curb          import CURBWrapper
from agents.tabular_q       import TabularQAgent

LAM_VALUES = [0.1, 0.2, 0.5]


# ── Conditions ────────────────────────────────────────────────────────────────

def all_conditions():
    conds = [{"name": "baseline", "lam": 0.0,
              "env": "perfect", "use_tv": False, "use_belief": False}]
    conds.append({"name": "imperfect_monitoring", "lam": 0.0,
                  "env": "imperfect", "use_tv": False, "use_belief": False})
    for lam in LAM_VALUES:
        tag = str(lam).replace(".", "")
        conds.append({"name": f"full_l{tag}", "lam": lam,
                      "env": "perfect",
                      "use_tv": True, "use_belief": True})
    return conds


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(args):
    n, cond, seed, n_steps, log_every, k_belief, m_belief, beta = args
    np.random.seed(seed)

    curb_cfg  = {"lam": cond["lam"], "tv_threshold": 0.0, "W": 500}
    agent_cfg = {"alpha": 0.15, "delta": 0.95, "beta": beta,
                 "tstable": 100_000,
                 "use_belief": cond["use_belief"],
                 "K_belief": k_belief, "M_belief": m_belief,
                 "seed": seed}

    if cond["env"] == "imperfect":
        base_env = CournotImperfectEnv(n=n, seed=seed)
        env      = base_env                        # no CURB wrapper for IM baseline
    else:
        base_env = CournotEnv(n=n)
        env      = (CURBWrapper(base_env, curb_cfg)
                    if cond["lam"] > 0 or cond["use_belief"] else base_env)

    agent = TabularQAgent(env, agent_cfg)
    r     = agent.run(env, n_steps, log_every)

    print(f"  [{cond['name']:<22}] seed={seed:3d}  CI={r['final_ci']:.3f}",
          flush=True)
    return cond, seed, r


# ── Saving ────────────────────────────────────────────────────────────────────

def _save_seed(cond, seed, r, run_dir, n):
    seed_dir = os.path.join(run_dir, cond["name"], f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    q_cols  = [f"q_agent{i}" for i in range(n)]
    tv_cols = [f"tv{i}" for i in range(n)]
    with open(os.path.join(seed_dir, "steps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step"] + q_cols + ["CI"] + tv_cols)
        w.writeheader()
        prices = r["prices"]   # carries per-agent quantity for Cournot
        tv     = r["tv"]
        for t_idx, step in enumerate(r["steps"]):
            row = {"step": int(step),
                   "CI":   round(float(r["ci"][t_idx]), 6)}
            for i in range(n):
                row[f"q_agent{i}"] = round(float(prices[t_idx, i]), 4) if t_idx < len(prices) else 0.0
                row[f"tv{i}"]      = round(float(tv[t_idx, i]),     6) if t_idx < len(tv)     else 0.0
            w.writerow(row)

    with open(os.path.join(seed_dir, "sessions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "final_CI", "converged", "t_conv"])
        w.writeheader()
        w.writerow({"seed": seed,
                    "final_CI":  round(r["final_ci"], 4),
                    "converged": r["converged"],
                    "t_conv":    r["t_conv"]})


def _save_sweep(all_results, run_dir):
    with open(os.path.join(run_dir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "env", "lam",
                                          "use_tv", "use_belief",
                                          "seed", "final_CI", "converged", "t_conv"])
        w.writeheader()
        for cond, seed, r in all_results:
            w.writerow({"condition":  cond["name"],
                        "env":        cond["env"],
                        "lam":        cond["lam"],
                        "use_tv":     cond["use_tv"],
                        "use_belief": cond["use_belief"],
                        "seed":       seed,
                        "final_CI":   round(r["final_ci"], 4),
                        "converged":  r["converged"],
                        "t_conv":     r["t_conv"]})


def _print_summary(all_results):
    by_cond = defaultdict(list)
    for cond, seed, r in all_results:
        by_cond[cond["name"]].append(r["final_ci"])

    base_ci = float(np.mean(by_cond.get("baseline", [0])))
    print(f"\n{'Condition':<22} {'CI':>6} {'±':>5}  {'vs base':>8}")
    print("-" * 48)
    for name in ["baseline"] + sorted(
            [n for n in by_cond if n != "baseline"],
            key=lambda x: np.mean(by_cond[x])):
        cis   = by_cond[name]
        delta = np.mean(cis) - base_ci
        tag   = "—" if name == "baseline" else f"{'↓' if delta < 0 else '↑'}{abs(delta):.3f}"
        print(f"  {name:<20} {np.mean(cis):>6.3f} {np.std(cis):>5.3f}  {tag:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=2)
    parser.add_argument("--seeds",    type=int, default=20)
    parser.add_argument("--workers",  type=int, default=20)
    parser.add_argument("--steps",    type=int, default=None,
                        help="override step count (default 3M for n=2)")
    parser.add_argument("--log_every",type=int, default=10_000)
    parser.add_argument("--k_belief", type=int, default=100)
    parser.add_argument("--m_belief", type=int, default=5)
    parser.add_argument("--beta",     type=float, default=None,
                        help="ε-decay rate (default 4e-6 for n=2)")
    parser.add_argument("--smoke",    action="store_true")
    args = parser.parse_args()

    if args.beta is None:
        args.beta = 4e-6 if args.n == 2 else 1.2e-6

    if args.smoke:
        args.seeds     = 2
        args.workers   = 2
        args.steps     = 50_000
        args.log_every = 5_000

    n_steps = args.steps or 3_000_000

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", f"cournot_n{args.n}"
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
        "env":         "cournot",
        "n":           args.n,
        "n_steps":     n_steps,
        "seeds":       args.seeds,
        "workers":     args.workers,
        "beta":        args.beta,
        "k_belief":    args.k_belief,
        "m_belief":    args.m_belief,
        "log_every":   args.log_every,
        "alpha":       0.15,
        "delta":       0.95,
        "tstable":     100_000,
        "K_actions":   15,
        "demand_d":    300.0,
        "q_grid":      [70.0, 105.0],
        "normalize_rewards": True,
        "W_tv":        500,
        "tv_threshold": 0.0,
        "tv_lag":      1,
        "dev_idx_rule": "nash_idx + 1",
        "conditions":  [c["name"] for c in conds],
        "lam_values":  sorted({c["lam"] for c in conds}),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg_dump, f, indent=2)

    tasks = [(args.n, cond, seed, n_steps, args.log_every,
              args.k_belief, args.m_belief, args.beta)
             for cond in conds
             for seed in range(args.seeds)]

    env_ref = CournotEnv(n=args.n)
    env_ref.summary()
    print(f"Conditions: {len(conds)}  Seeds: {args.seeds}  "
          f"Total: {len(tasks)}  Workers: {args.workers}\n")

    t0 = time.time()
    with mp.Pool(min(args.workers, len(tasks))) as pool:
        all_results = pool.map(_worker, tasks)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")

    for cond, seed, r in all_results:
        _save_seed(cond, seed, r, run_dir, args.n)
    _save_sweep(all_results, run_dir)
    _print_summary(all_results)
    print(f"\nResults saved to {run_dir}")

    # Generate plots
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
        from plot_cournot import make_all_plots
        make_all_plots(run_dir, n=args.n, n_seeds=args.seeds)
        print(f"Plots saved to {run_dir}/plots/")
    except Exception as e:
        print(f"[plot] {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
