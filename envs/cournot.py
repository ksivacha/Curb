"""
envs/cournot.py — Symmetric Cournot oligopoly with deterministic demand.

Adapted from Calvano, Calzolari, Denicolò, Pastorello (2021).

Demand        : p = d − Σ q_i
Reward (raw)  : π_i = p · q_i
Reward (norm) : π_i / mono_profit_raw    (returned in step())

Default parameters (n=2):
    d = 300, c = 0
    q ∈ {70, 72.5, ..., 102.5, 105}    (K=15 levels, step v=2.5)
    Nash:     q^C = 100,  π^C = 10,000      (Δ=0)
    Monopoly: q^M = 75,   π^M = 11,250      (Δ=1)
    For n=3:  q^C = 75,   q^M = 50

Rewards are normalised by mono_profit_raw so λ values in CURB are
directly comparable to the Bertrand sweep (where profits are ~0.3).
"""

import numpy as np
import gymnasium as gym


class CournotEnv(gym.Env):

    metadata = {}

    # ── Calvano (2021) defaults ──────────────────────────────────────────────
    K          = 15
    D          = 300.0
    Q_MIN      = 70.0
    Q_MAX      = 105.0
    ALPHA      = 0.15
    DELTA      = 0.95
    BETA_BASE  = 4e-6

    def __init__(self, n: int = 2, beta: float = None,
                 normalize: bool = True):
        super().__init__()
        self.n         = n
        self.beta      = beta if beta is not None else self.BETA_BASE
        self.normalize = normalize

        self.quantity_grid = np.linspace(self.Q_MIN, self.Q_MAX, self.K)
        # Alias for compatibility with code that reads price_grid
        self.price_grid    = self.quantity_grid

        self.nash_q = self.D / (self.n + 1)
        self.mono_q = self.D / (2 * self.n)
        self.nash_idx = int(np.argmin(np.abs(self.quantity_grid - self.nash_q)))
        self.mono_idx = int(np.argmin(np.abs(self.quantity_grid - self.mono_q)))

        self.nash_profit_raw = self.nash_q * (self.D - self.n * self.nash_q)
        self.mono_profit_raw = self.mono_q * (self.D - self.n * self.mono_q)

        self._scale       = self.mono_profit_raw if normalize else 1.0
        self.nash_profit  = self.nash_profit_raw / self._scale
        self.mono_profit  = self.mono_profit_raw / self._scale

        self.profit_matrix = self._build_profit_matrix()

        self.action_space      = gym.spaces.MultiDiscrete([self.K] * self.n)
        self.observation_space = gym.spaces.MultiDiscrete([self.K] * self.n)
        self.state             = np.zeros(self.n, dtype=np.int64)

    # ── Profit / equilibrium machinery ───────────────────────────────────────

    def profits(self, quantities: np.ndarray) -> np.ndarray:
        total_q = float(np.sum(quantities))
        price   = max(0.0, self.D - total_q)
        return (price * quantities) / self._scale

    def _build_profit_matrix(self) -> np.ndarray:
        PI = np.zeros([self.K] * self.n + [self.n], dtype=np.float64)
        for idxs in np.ndindex(*([self.K] * self.n)):
            quantities = self.quantity_grid[list(idxs)]
            total_q    = float(quantities.sum())
            price      = max(0.0, self.D - total_q)
            PI[tuple(idxs)] = (price * quantities) / self._scale
        return PI

    def collusion_index(self, profits) -> float:
        """Δ = (mean_π − π_C) / (π_M − π_C)."""
        avg_pi = float(np.mean(profits))
        denom  = self.mono_profit - self.nash_profit
        if denom <= 0:
            return 0.0
        return (avg_pi - self.nash_profit) / denom

    # ── gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.randint(0, self.K, size=self.n)
        return self.state.copy(), {}

    def step(self, actions):
        actions    = np.asarray(actions, dtype=np.int64)
        quantities = self.quantity_grid[actions]
        total_q    = float(quantities.sum())
        market_p   = max(0.0, self.D - total_q)
        rewards    = (market_p * quantities) / self._scale

        info = {
            "raw_rewards":  rewards.copy(),
            # "prices" key kept for compatibility with logging in tabular_q —
            # for Cournot it carries per-agent quantity (the action value).
            "prices":       quantities.copy(),
            "market_price": float(market_p),
            "total_q":      total_q,
        }
        self.state = actions.copy()
        return self.state.copy(), rewards, False, False, info

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def summary(self):
        print(f"CournotEnv n={self.n}  K={self.K}  d={self.D}  "
              f"normalize={self.normalize}")
        print(f"  Nash:     q*={self.nash_q:.2f}  π*={self.nash_profit_raw:.2f} "
              f"(norm={self.nash_profit:.4f})  idx={self.nash_idx}")
        print(f"  Monopoly: q*={self.mono_q:.2f}  π*={self.mono_profit_raw:.2f} "
              f"(norm={self.mono_profit:.4f})  idx={self.mono_idx}")
