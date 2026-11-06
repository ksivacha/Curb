# CURB

This repository contains the code and environments needed to reproduce all
experiments in the paper. CURB is an intervention mechanism that mitigates
algorithmic collusion by combining a Total Variation based reward shaping
(which detects strategically contingent punishment patterns) with
counterfactual belief injection.

## Repository structure

```
.
├── agents/                     RL agents
│   ├── tabular_q.py            Tabular Q-learning (Calvano-style)
│   └── dqn_agent.py            Independent multi-agent DQN (PyTorch)
│
├── envs/                       Environments
│   ├── calvano_bertrand.py     Bertrand price competition (Calvano 2020)
│   ├── cournot.py              Cournot quantity competition (deterministic demand)
│   └── cournot_imperfect.py    Cournot with stochastic demand (Calvano 2021)
│
├── wrappers/                   Reward-shaping wrappers
│   ├── curb.py                 CURB: TV-based reward shaping
│   ├── curb_jsd.py             CURB with pluggable divergence (TV / JSD)
│   └── buybox.py               PDP / DPDP buy-box baselines (platform-design)
│
├── experiments/                One sweep script per experiment
│   ├── calvano_sweep.py        Calvano (Bertrand) λ-sweep, n=2 and n=3
│   ├── cournot_sweep.py        Cournot λ-sweep with imperfect-monitoring baseline
│   ├── dqn_calvano_sweep.py    DQN-agent λ-sweep on Bertrand
│   ├── baseline_comparison.py  CURB vs PDP/DPDP buy-box interventions
│   ├── jsd_ablation.py         TV vs JSD (norm./raw) divergence ablation
│   └── forced_defection.py     Calvano-style forced-defection probe
│
├── analysis/                   Plotting / table scripts
│   ├── plot_calvano.py
│   ├── plot_cournot.py
│   ├── plot_dqn_calvano.py
│   ├── plot_jsd_ablation.py
│   ├── plot_forced_defection.py
│   ├── table_baseline_comparison.py
│   └── plot_neurips_*.py
│
├── requirements.txt            Pip dependencies
└── README.md
```

## Experiments

The seven experiments below correspond to the seven Results subsections
in the paper. Each writes its outputs to `results/<experiment_name>/run_N/`,
including a `run_config.json` (or `config.json`) recording all
hyperparameters used.

| # | Script                                | Purpose                                                              |
|---|---------------------------------------|----------------------------------------------------------------------|
| 1 | `experiments/calvano_sweep.py`        | λ ablation and component (TV-only / Belief-only / CURB) on Bertrand n=2 |
| 2 | `experiments/calvano_sweep.py --n 3`  | Scaling to n=3 sellers                                               |
| 3 | `experiments/dqn_calvano_sweep.py`    | DQN agents (function approximation) on Bertrand n=2                  |
| 4 | `experiments/cournot_sweep.py`        | Cournot quantity competition with imperfect-monitoring baseline      |
| 5 | `experiments/baseline_comparison.py`  | CURB vs PDP and DPDP buy-box platform-design baselines               |
| 6 | `experiments/jsd_ablation.py`         | TV vs JSD (normalised / raw) divergence ablation                     |
| 7 | `experiments/forced_defection.py`     | Calvano-style forced-defection probe (mechanism verification)        |

## Reproduction

### Smoke tests (local)

Each experiment has a `--smoke` flag that runs a minimal version (2 seeds,
50K steps) for sanity checking:

```bash
python experiments/calvano_sweep.py --n 2 --smoke
python experiments/cournot_sweep.py --n 2 --smoke
python experiments/dqn_calvano_sweep.py --smoke
python experiments/baseline_comparison.py --smoke
python experiments/jsd_ablation.py --smoke
python experiments/forced_defection.py --smoke
```

### Full runs

The full experiments are designed to run with parallel workers
(`--workers N`). On a single multi-core machine:

```bash
# Bertrand n=2 (10 conditions × 20 seeds × 3M steps; tabular Q, CPU-only)
python experiments/calvano_sweep.py --n 2 --seeds 20 --workers 32 --steps 3000000

# Bertrand n=3 (3 conditions × 20 seeds × 20M steps)
python experiments/calvano_sweep.py --n 3 --seeds 20 --workers 32 \
    --steps 20000000 --beta 1.2e-6 --lams 0.2,0.5 --no_belief_only --no_tv_only

# DQN-Calvano (4 conditions × 20 seeds × 2M steps; PyTorch DQN)
python experiments/dqn_calvano_sweep.py --seeds 20 --workers 32 --steps 2000000

# Cournot (5 conditions × 20 seeds × 3M steps)
python experiments/cournot_sweep.py --n 2 --seeds 20 --workers 32 --steps 3000000

# CURB vs PDP/DPDP (5 conditions × 20 seeds × 3M steps)
python experiments/baseline_comparison.py --seeds 20 --workers 32

# JSD ablation (10 conditions × 20 seeds × 3M steps)
python experiments/jsd_ablation.py --seeds 20 --workers 32 --steps 3000000

# Forced defection (2 conditions × 20 seeds × 3M steps)
python experiments/forced_defection.py --seeds 20 --workers 32 --steps 3000000
```

These can also be submitted to a SLURM (or other) batch scheduler — each
script is multiprocessing-based so the only requirement is enough CPU
cores (and a single GPU for `dqn_calvano_sweep.py`). Wall-clock per job
ranges from a few minutes (forced-defection at smoke scale) to ~10 hours
(Bertrand n=3 with 20M steps).

### Plots and tables

After a job completes, default plots are auto-generated into
`results/<experiment>/run_N/plots/`. To regenerate the paper-ready (PDF)
versions:

```bash
python analysis/plot_neurips_calvano.py            --run_dir results/calvano_n2/run_1
python analysis/plot_neurips_extra.py --kind cournot --run_dir results/cournot_n2/run_1
python analysis/plot_neurips_extra.py --kind dqn     --run_dir results/dqn_calvano/run_1
python analysis/plot_neurips_jsd.py                  --run_dir results/jsd_ablation/run_1
python analysis/plot_neurips_forced_defection.py     --run_dir results/forced_defection/run_1
```

The baseline-comparison table is rendered separately:

```bash
python analysis/table_baseline_comparison.py --run_dir results/baseline_comparison/run_1
```

## Key hyperparameters

All hyperparameters used in a run are recorded in that run's
`run_config.json` for full reproducibility. Defaults are:

| Component        | Default                                              |
|------------------|------------------------------------------------------|
| Q-learning       | α=0.15, γ=0.95, β=4×10⁻⁶ (n=2) / 1.2×10⁻⁶ (n=3)      |
| Belief injection | K_belief=100, M_belief=5                             |
| TV detector      | rolling window W=500, lag-1 conditioning             |
| DQN              | lr=10⁻³, target update C=100, replay 10K, n_step=1   |
| Seeds            | 20                                                   |
| Action grid      | K=15 (Calvano 2020 defaults)                         |

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{anonymous2026curb,
  title  = {CURB: ...},
  author = {Anonymous},
  booktitle = {NeurIPS 2026},
  year   = {2026},
}
```
