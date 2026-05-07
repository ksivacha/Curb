"""
experiments/forced_defection.py — Calvano-style forced-defection experiment.

For each (condition, seed):
  1. Train a tabular Q-learning pair on Calvano n=2 (3M steps).
  2. After training, run forced_defection_test:
       - Start from joint state (mono, mono)
       - For 200 steps, both agents play greedy (ε=0)
       - At step 50, force agent 0 to play `defect_action`
       - Watch agent 1's response (punishment / recovery)

Conditions
----------
  baseline       : no intervention
  tv_only_l05    : λ=0.5 TV penalty, NO belief injection.
                   We use TV-only (rather than full CURB) because under full
                   CURB belief injection has already driven agents toward Nash,
                   so there is nothing for agent 1 to "defend" — both agents
                   start near competitive prices and the punishment-vs-no-
                   punishment contrast disappears. With TV-only, agents still
                   try to collude (so they have a collusive policy to forcibly
                   start from at (mono, mono)), but the TV penalty
                   disincentivises the punishment response specifically.

Output
------
results/forced_defection/run_{N}/
  config.json
  sweep_summary.csv                  per-seed punished / recovery_t / final_CI
  <condition>/seed_<s>/
    train_steps.csv                  training-time CI / TV / prices
    defection_prices.csv             200-step forced-defection prices per agent
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
from wrappers.curb           import CURBWrapper
from agents.tabular_q        import TabularQAgent


# ── Conditions ────────────────────────────────────────────────────────────────

CONDITIONS = [
    {"name": "baseline",    "lam": 0.0, "use_curb": False, "use_belief": False},
    {"name": "tv_only_l05", "lam": 0.5, "use_curb": True,  "use_belief": False},
]

DEFECT_STEP   = 50
TEST_STEPS    = 200
# Default defection target price. The actual `defect_action` is the action
# whose price is closest to this on the Calvano grid (resolved at runtime).
DEFAULT_DEFECT_PRICE = 1.5


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(args):
    cond, seed, n_steps, log_every, k_belief, m_belief, beta, defect_action = args
    np.random.seed(seed)

    curb_cfg  = {"lam": cond["lam"], "tv_threshold": 0.0, "W": 500}
    agent_cfg = {"alpha": 0.15, "delta": 0.95, "beta": beta,
                 "tstable": 100_000,
                 "use_belief": cond["use_belief"],
                 "K_belief": k_belief, "M_belief": m_belief,
                 "seed": seed}

    base_env = CalvanoBertrandEnv(n=2)
    env      = CURBWrapper(base_env, curb_cfg) if cond["use_curb"] else base_env
    agent    = TabularQAgent(env, agent_cfg)
    train_r  = agent.run(env, n_steps, log_every)

    # Forced-defection test on the trained Q-tables. Use a fresh base env so
    # the wrapper's TV buffer doesn't leak into the test rollout.
    test_env = CalvanoBertrandEnv(n=2)
    test_r   = agent.forced_defection_test(
        test_env,
        n_steps=TEST_STEPS,
        defect_step=DEFECT_STEP,
        defect_action=int(defect_action),
        punish_window=20,
    )

    print(f"  [{cond['name']:<14}] seed={seed:3d}  CI={train_r['final_ci']:.3f}  "
          f"punished={test_r['punished']}  recovery={test_r['recovery_t']}",
          flush=True)
    return cond, seed, train_r, test_r


# ── Saving ────────────────────────────────────────────────────────────────────

def _save_seed(cond, seed, train_r, test_r, run_dir):
    seed_dir = os.path.join(run_dir, cond["name"], f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # Training trajectory (mirrors calvano_sweep.py format)
    p_cols  = [f"price_agent{i}" for i in range(2)]
    tv_cols = [f"tv{i}" for i in range(2)]
    with open(os.path.join(seed_dir, "train_steps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step"] + p_cols + ["CI"] + tv_cols)
        w.writeheader()
        prices, tv = train_r["prices"], train_r["tv"]
        for t_idx, step in enumerate(train_r["steps"]):
            row = {"step": int(step),
                   "CI":   round(float(train_r["ci"][t_idx]), 6)}
            for i in range(2):
                row[f"price_agent{i}"] = (round(float(prices[t_idx, i]), 6)
                                          if t_idx < len(prices) else 0.0)
                row[f"tv{i}"]          = (round(float(tv[t_idx, i]),     6)
                                          if t_idx < len(tv)     else 0.0)
            w.writerow(row)

    # Forced-defection trajectory
    with open(os.path.join(seed_dir, "defection_prices.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "price_agent0", "price_agent1",
                    "action_agent0", "action_agent1"])
        prices, acts = test_r["prices"], test_r["actions"]
        for t in range(len(prices)):
            w.writerow([t,
                        round(float(prices[t, 0]), 6),
                        round(float(prices[t, 1]), 6),
                        int(acts[t, 0]),
                        int(acts[t, 1])])


def _save_summary(all_results, run_dir):
    with open(os.path.join(run_dir, "sweep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "seed", "final_CI",
                                          "punished", "recovery_t",
                                          "converged", "t_conv"])
        w.writeheader()
        for cond, seed, train_r, test_r in all_results:
            w.writerow({"condition":  cond["name"],
                        "seed":       seed,
                        "final_CI":   round(train_r["final_ci"], 4),
                        "punished":   bool(test_r["punished"]),
                        "recovery_t": int(test_r["recovery_t"]),
                        "converged":  train_r["converged"],
                        "t_conv":     train_r["t_conv"]})


def _print_summary(all_results):
    by_cond = defaultdict(lambda: {"ci": [], "punished": [], "recovery": []})
    for cond, seed, tr, te in all_results:
        by_cond[cond["name"]]["ci"].append(tr["final_ci"])
        by_cond[cond["name"]]["punished"].append(int(te["punished"]))
        rec = te["recovery_t"]
        if rec >= 0:
            by_cond[cond["name"]]["recovery"].append(rec)

    print(f"\n{'Condition':<16}  {'CI':>6}  {'%punished':>10}  "
          f"{'recovery (mean ± std)':>22}")
    print("-" * 60)
    for c in CONDITIONS:
        d = by_cond[c["name"]]
        ci_mean = np.mean(d["ci"]) if d["ci"] else float("nan")
        pun_pct = 100.0 * np.mean(d["punished"]) if d["punished"] else float("nan")
        if d["recovery"]:
            rec_str = f"{np.mean(d['recovery']):>5.1f} ± {np.std(d['recovery']):>4.1f}"
        else:
            rec_str = "    no punishments"
        print(f"  {c['name']:<14}  {ci_mean:>6.3f}  {pun_pct:>8.1f}%  {rec_str:>22}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",     type=int, default=20)
    parser.add_argument("--workers",   type=int, default=20)
    parser.add_argument("--steps",     type=int, default=3_000_000)
    parser.add_argument("--log_every", type=int, default=10_000)
    parser.add_argument("--beta",      type=float, default=4e-6)
    parser.add_argument("--k_belief",  type=int, default=100)
    parser.add_argument("--m_belief",  type=int, default=5)
    parser.add_argument("--smoke",     action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds     = 2
        args.workers   = 2
        args.steps     = 50_000
        args.log_every = 5_000

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "forced_defection")
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(d.split("_")[1]) for d in os.listdir(base_dir)
                if d.startswith("run_") and d.split("_")[1].isdigit()]
    run_num = max(existing, default=0) + 1
    run_dir = os.path.join(base_dir, f"run_{run_num}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving to: {run_dir}\n")

    run_config = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "argv":         sys.argv,
        "cli_args":     vars(args),
        "n_steps":      args.steps,
        "seeds":        list(range(args.seeds)),
        "conditions":   CONDITIONS,
        "defect_step":  DEFECT_STEP,
        "defect_action": DEFECT_ACTION,
        "test_steps":   TEST_STEPS,
        "agent_cfg": {
            "alpha":     0.15, "delta": 0.95, "beta": args.beta,
            "tstable":   100_000,
            "K_belief":  args.k_belief, "M_belief": args.m_belief,
        },
        "curb_cfg": {"W": 500, "tv_threshold": 0.0, "tv_lag": 1},
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    tasks = [(cond, seed, args.steps, args.log_every,
              args.k_belief, args.m_belief, args.beta)
             for cond in CONDITIONS
             for seed in range(args.seeds)]

    print(f"Conditions: {len(CONDITIONS)}  Seeds: {args.seeds}  "
          f"Total: {len(tasks)}  Workers: {args.workers}  "
          f"Train steps: {args.steps:,}\n")

    t0 = time.time()
    all_results = []
    done = 0
    try:
        with mp.Pool(min(args.workers, len(tasks))) as pool:
            for cond, seed, train_r, test_r in pool.imap_unordered(_worker, tasks):
                _save_seed(cond, seed, train_r, test_r, run_dir)
                all_results.append((cond, seed, train_r, test_r))
                done += 1
                if done % 5 == 0 or done == len(tasks):
                    print(f"  [progress] {done}/{len(tasks)} seeds saved "
                          f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
    except KeyboardInterrupt:
        print(f"\n[interrupted] {done}/{len(tasks)} seeds saved to {run_dir}",
              flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")

    if all_results:
        _save_summary(all_results, run_dir)
        _print_summary(all_results)

        # Generate plots
        try:
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "analysis"))
            from plot_forced_defection import make_all_plots
            make_all_plots(run_dir, n_seeds=args.seeds)
        except Exception as e:
            print(f"[plot] {e}")

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
