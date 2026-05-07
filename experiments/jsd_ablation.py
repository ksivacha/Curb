"""
experiments/jsd_ablation.py — Divergence-metric ablation for CURB on Calvano n=2.

Conditions
----------
  baseline           : no intervention
  tv_full_l{01,02,05}      : CURB with TV distance               (max penalty = λ·1.0)
  jsd_norm_full_l{01,02,05}: CURB with JSD ÷ ln(2) ∈ [0,1]      (max penalty = λ·1.0)
  jsd_raw_full_l{01,02,05} : CURB with raw JSD in nats ∈ [0,ln 2](max penalty = λ·0.693)

All shaped conditions use belief injection. The three-way comparison answers:
  TV vs jsd_norm  : does the choice of divergence matter at matched scale?
  jsd_norm vs jsd_raw : does normalisation matter (i.e. effective λ)?

Usage
-----
  python jsd_ablation.py --seeds 20 --workers 20
  python jsd_ablation.py --seeds 2 --smoke
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
from wrappers.curb_jsd import CURBJSDWrapper
from agents.tabular_q import TabularQAgent


# ── Conditions ────────────────────────────────────────────────────────────────

LAM_TAGS = [(0.1, "l01"), (0.2, "l02"), (0.5, "l05")]


def all_conditions():
    conds = [{"name":     "baseline", "lam": 0.0,
              "metric":   "tv", "jsd_normalize": True,
              "use_belief": False}]
    for lam, tag in LAM_TAGS:
        conds.append({"name":          f"tv_full_{tag}",
                      "lam":           lam,
                      "metric":        "tv",
                      "jsd_normalize": True,    # ignored when metric=tv
                      "use_belief":    True})
        conds.append({"name":          f"jsd_norm_full_{tag}",
                      "lam":           lam,
                      "metric":        "jsd",
                      "jsd_normalize": True,
                      "use_belief":    True})
        conds.append({"name":          f"jsd_raw_full_{tag}",
                      "lam":           lam,
                      "metric":        "jsd",
                      "jsd_normalize": False,
                      "use_belief":    True})
    return conds


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(args):
    cond, seed, n_steps, log_every = args
    np.random.seed(seed)

    curb_cfg = {"lam":           cond["lam"],
                "tv_threshold":  0.0,
                "W":             500,
                "metric":        cond["metric"],
                "jsd_normalize": cond["jsd_normalize"]}
    agent_cfg = {"alpha": 0.15, "delta": 0.95, "beta": 4e-6,
                 "tstable": 100_000,
                 "use_belief": cond["use_belief"],
                 "K_belief": 100, "M_belief": 5,
                 "seed": seed}

    base_env = CalvanoBertrandEnv(n=2)
    env      = CURBJSDWrapper(base_env, curb_cfg) if cond["lam"] > 0 or cond["use_belief"] else base_env
    agent    = TabularQAgent(env, agent_cfg)
    r        = agent.run(env, n_steps, log_every)

    print(f"  [{cond['name']:<16}] seed={seed:3d}  CI={r['final_ci']:.3f}", flush=True)
    return cond, seed, r


# ── Saving ────────────────────────────────────────────────────────────────────

def _save_seed(cond, seed, r, run_dir):
    seed_dir = os.path.join(run_dir, cond["name"], f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    with open(os.path.join(seed_dir, "steps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "CI", "tv0", "tv1"])
        w.writeheader()
        tv = r["tv"]
        for t_idx, step in enumerate(r["steps"]):
            w.writerow({"step": int(step),
                        "CI":  round(float(r["ci"][t_idx]), 6),
                        "tv0": round(float(tv[t_idx, 0]),   6) if t_idx < len(tv) else 0.0,
                        "tv1": round(float(tv[t_idx, 1]),   6) if t_idx < len(tv) else 0.0})

    with open(os.path.join(seed_dir, "sessions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "final_CI", "converged", "t_conv"])
        w.writeheader()
        w.writerow({"seed":     seed,
                    "final_CI": round(r["final_ci"], 4),
                    "converged": r["converged"],
                    "t_conv":    r["t_conv"]})


def _save_sweep(all_results, run_dir):
    with open(os.path.join(run_dir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "metric", "jsd_normalize",
                                          "lam", "seed", "final_CI",
                                          "converged", "t_conv"])
        w.writeheader()
        for cond, seed, r in all_results:
            w.writerow({"condition":     cond["name"],
                        "metric":        cond["metric"],
                        "jsd_normalize": cond["jsd_normalize"],
                        "lam":           cond["lam"],
                        "seed":          seed,
                        "final_CI":      round(r["final_ci"], 4),
                        "converged":     r["converged"],
                        "t_conv":        r["t_conv"]})


def _print_summary(all_results, conds):
    by_cond = defaultdict(list)
    for cond, seed, r in all_results:
        by_cond[cond["name"]].append(r["final_ci"])

    base_ci = float(np.mean(by_cond.get("baseline", [0])))
    print(f"\n{'Condition':<20}  {'CI':>6}  {'+/-':>5}  {'vs base':>8}")
    print("-" * 44)
    for cond in conds:
        vals  = by_cond[cond["name"]]
        if not vals:
            continue
        delta = np.mean(vals) - base_ci
        tag   = "--" if cond["name"] == "baseline" else f"{'v' if delta < 0 else '^'}{abs(delta):.3f}"
        print(f"  {cond['name']:<18}  {np.mean(vals):>6.3f}  {np.std(vals):>5.3f}  {tag:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",     type=int, default=20)
    parser.add_argument("--workers",   type=int, default=20)
    parser.add_argument("--steps",     type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10_000)
    parser.add_argument("--smoke",     action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds     = 2
        args.workers   = 2
        args.steps     = 50_000
        args.log_every = 5_000

    n_steps = args.steps or 3_000_000

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "jsd_ablation"
    )
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(d.split("_")[1]) for d in os.listdir(base_dir)
                if d.startswith("run_") and d.split("_")[1].isdigit()]
    run_num = max(existing, default=0) + 1
    run_dir = os.path.join(base_dir, f"run_{run_num}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving to: {run_dir}\n")

    conds = all_conditions()
    tasks = [(cond, seed, n_steps, args.log_every)
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
        _save_sweep(all_results, run_dir)
        _print_summary(all_results, conds)

        # Generate plots
        try:
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "analysis"))
            from plot_jsd_ablation import make_all_plots
            make_all_plots(run_dir, n_seeds=args.seeds)
        except Exception as e:
            print(f"[plot] {e}")

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
