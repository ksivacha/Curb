"""
experiments/calvano_sweep.py — Calvano CURB sweep for n=2 and n=3.

Directory structure (mirrors Market_env/tv_intervention)
---------------------------------------------------------
results/calvano_n{N}/run_{K}/
  sweep_results.csv
  plots/
  <condition>/
    seed_<s>/
      steps.csv       — step, price_agent0…, CI, tv0…
      sessions.csv    — final scalars

Conditions
----------
  baseline       : no intervention (Calvano 2020 replication)
  tv_only_lXX    : TV shaping only
  belief_only    : belief injection only
  full_lXX       : TV + belief (main CURB result)

Usage
-----
  python calvano_sweep.py --n 2 --seeds 50 --workers 50
  python calvano_sweep.py --n 3 --seeds 50 --workers 50 --steps 10000000
  python calvano_sweep.py --n 2 --seeds 2 --smoke
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.calvano_bertrand import CalvanoBertrandEnv
from wrappers.curb import CURBWrapper
from agents.tabular_q import TabularQAgent

LAM_VALUES      = [0.05, 0.1, 0.2, 0.5]
TV_ONLY_LAMS    = [0.05, 0.1, 0.2, 0.5]
BELIEF_ONLY_LAM = 0.0


# ── Condition definitions ─────────────────────────────────────────────────────

def all_conditions(lams=None, tv_only_lams=None,
                   include_belief_only=True, include_tv_only=True):
    """Build condition list. Optional overrides let you slim the sweep."""
    if lams is None:
        lams = LAM_VALUES
    if tv_only_lams is None:
        tv_only_lams = TV_ONLY_LAMS

    conds = []

    conds.append({"name": "baseline", "lam": 0.0,
                  "use_tv": False, "use_belief": False})

    if include_belief_only:
        conds.append({"name": "belief_only", "lam": BELIEF_ONLY_LAM,
                      "use_tv": False, "use_belief": True})

    if include_tv_only:
        for lam in tv_only_lams:
            tag = str(lam).replace(".", "")
            conds.append({"name": f"tv_only_l{tag}", "lam": lam,
                          "use_tv": True, "use_belief": False})

    for lam in lams:
        tag = str(lam).replace(".", "")
        conds.append({"name": f"full_l{tag}", "lam": lam,
                      "use_tv": True, "use_belief": True})

    return conds


# ── Per-seed worker ───────────────────────────────────────────────────────────

def _worker(args):
    n, cond, seed, n_steps, log_every, beta, k_belief, m_belief = args
    np.random.seed(seed)

    curb_cfg = {"lam": cond["lam"], "tv_threshold": 0.0, "W": 500}
    agent_cfg = {"alpha": 0.15, "delta": 0.95, "beta": beta,
                 "tstable": 100_000,
                 "use_belief": cond["use_belief"],
                 "K_belief": k_belief, "M_belief": m_belief,
                 "seed": seed}

    base_env = CalvanoBertrandEnv(n=n)
    env      = CURBWrapper(base_env, curb_cfg) if cond["lam"] > 0 or cond["use_belief"] else base_env
    agent    = TabularQAgent(env, agent_cfg)
    r        = agent.run(env, n_steps, log_every)

    print(f"  [{cond['name']:<18}] seed={seed:3d}  CI={r['final_ci']:.3f}", flush=True)

    return cond, seed, r


# ── File saving ───────────────────────────────────────────────────────────────

def _save_seed(cond, seed, r, run_dir, n):
    seed_dir = os.path.join(run_dir, cond["name"], f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # steps.csv
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

    # sessions.csv
    with open(os.path.join(seed_dir, "sessions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "final_CI", "converged", "t_conv"])
        w.writeheader()
        w.writerow({"seed":     seed,
                    "final_CI": round(r["final_ci"], 4),
                    "converged": r["converged"],
                    "t_conv":    r["t_conv"]})


def _save_sweep(all_results, run_dir):
    with open(os.path.join(run_dir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "lam", "use_tv", "use_belief",
                                          "seed", "final_CI", "converged", "t_conv"])
        w.writeheader()
        for cond, seed, r in all_results:
            w.writerow({"condition":  cond["name"],
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

    base_ci = np.mean(by_cond.get("baseline", [0]))
    print(f"\n{'Condition':<22} {'CI':>6} {'+/-':>5}  {'vs base':>8}")
    print("-" * 46)
    for name in ["baseline"] + sorted(
            [n for n in by_cond if n != "baseline"],
            key=lambda x: np.mean(by_cond[x])):
        cis   = by_cond[name]
        delta = np.mean(cis) - base_ci
        tag   = "--" if name == "baseline" else f"{'v' if delta < 0 else '^'}{abs(delta):.3f}"
        print(f"  {name:<20} {np.mean(cis):>6.3f} {np.std(cis):>5.3f}  {tag:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=2)
    parser.add_argument("--seeds",    type=int, default=50)
    parser.add_argument("--workers",  type=int, default=50)
    parser.add_argument("--steps",    type=int, default=None,
                        help="override steps (default: 3M for n=2, 10M for n=3)")
    parser.add_argument("--log_every",type=int, default=10_000)
    parser.add_argument("--beta",     type=float, default=None,
                        help="ε-decay rate (default 4e-6 for n=2, 1.2e-6 for n=3)")
    parser.add_argument("--k_belief", type=int, default=100,
                        help="belief injection interval (steps between calls)")
    parser.add_argument("--m_belief", type=int, default=5,
                        help="injections per call per agent")
    parser.add_argument("--lams",     type=str, default=None,
                        help="comma-separated full-condition λ values, "
                             "e.g. '0.2,0.5' (default: full sweep)")
    parser.add_argument("--no_belief_only", action="store_true",
                        help="skip the belief_only condition")
    parser.add_argument("--no_tv_only",     action="store_true",
                        help="skip all tv_only_lXX conditions")
    parser.add_argument("--smoke",    action="store_true")
    args = parser.parse_args()

    if args.beta is None:
        args.beta = 4e-6 if args.n == 2 else 1.2e-6

    if args.smoke:
        args.seeds    = 2
        args.workers  = 2
        args.steps    = 50_000
        args.log_every = 5_000

    n_steps = args.steps or (3_000_000 if args.n == 2 else 10_000_000)

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", f"calvano_n{args.n}"
    )
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(d.split("_")[1]) for d in os.listdir(base_dir)
                if d.startswith("run_") and d.split("_")[1].isdigit()]
    run_num = max(existing, default=0) + 1
    run_dir = os.path.join(base_dir, f"run_{run_num}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving to: {run_dir}\n")

    if args.lams:
        lams_override     = [float(x) for x in args.lams.split(",") if x.strip()]
        tv_lams_override  = lams_override          # keep tv_only λ list aligned
    else:
        lams_override    = None
        tv_lams_override = None
    conds = all_conditions(lams=lams_override,
                           tv_only_lams=tv_lams_override,
                           include_belief_only=not args.no_belief_only,
                           include_tv_only=not args.no_tv_only)
    seeds = list(range(args.seeds))
    tasks = [(args.n, cond, seed, n_steps, args.log_every,
              args.beta, args.k_belief, args.m_belief)
             for cond in conds
             for seed in seeds]

    env_ref = CalvanoBertrandEnv(n=args.n)

    run_config = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "argv":        sys.argv,
        "cli_args":    vars(args),
        "n_steps":     n_steps,
        "seeds":       seeds,
        "conditions":  conds,
        "lam_values":  LAM_VALUES,
        "agent_cfg": {
            "alpha":     0.15, "delta": 0.95, "beta": args.beta,
            "tstable":   100_000,
            "K_belief":  args.k_belief, "M_belief": args.m_belief,
        },
        "curb_cfg": {
            "W":             500,
            "tv_threshold":  0.0,
            "tv_lag":        1,
        },
        "env": {
            "n":           args.n,
            "K":           env_ref.K,
            "beta":        env_ref.beta,
            "nash_price":  env_ref.nash_price,
            "mono_price":  env_ref.mono_price,
            "nash_profit": env_ref.nash_profit,
            "mono_profit": env_ref.mono_profit,
        },
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    print(f"Calvano n={args.n}  beta={env_ref.beta:.1e}  steps={n_steps:,}")
    print(f"Conditions: {len(conds)}  Seeds: {args.seeds}  "
          f"Total: {len(tasks)}  Workers: {args.workers}\n")

    t0 = time.time()
    all_results = []
    done = 0
    try:
        with mp.Pool(min(args.workers, len(tasks))) as pool:
            for cond, seed, r in pool.imap_unordered(_worker, tasks):
                _save_seed(cond, seed, r, run_dir, args.n)
                all_results.append((cond, seed, r))
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"  [progress] {done}/{len(tasks)} seeds saved "
                          f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
    except KeyboardInterrupt:
        print(f"\n[interrupted] {done}/{len(tasks)} seeds saved to {run_dir}", flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")

    if all_results:
        _save_sweep(all_results, run_dir)
        _print_summary(all_results)

    # Generate plots
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
        from plot_calvano import make_all_plots
        make_all_plots(run_dir, args.n, args.seeds)
        print(f"\nPlots saved to {run_dir}/plots/")
    except Exception as e:
        print(f"[plot] {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
