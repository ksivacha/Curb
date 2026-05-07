"""
calvano_bertrand.py — Calvano (2020) n-agent Bertrand oligopoly as a gymnasium Env.

Observation : int array of shape (n,) — last joint price indices (= current state)
Action      : int array of shape (n,) — joint price indices (MultiDiscrete)
Reward      : float array of shape (n,) — per-agent profits

The environment is multi-agent: step() accepts joint actions and returns
joint observations and rewards. TabularQAgent and the RLlib adapter both
consume this interface.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import product as iproduct
from scipy.optimize import fsolve


class CalvanoBertrandEnv(gym.Env):
    """
    Symmetric Bertrand oligopoly with multinomial logit demand (Calvano 2020).

    Parameters
    ----------
    n     : number of competing agents
    beta  : exploration decay rate; if None, auto-scaled as β_2 × k^(2−n)
            so the total exploration budget (≈1/β) grows with the state space.
            n=2 → 4e-6, n=3 → 2.7e-7, n=4 → 1.8e-8
    """

    metadata = {}

    # ── Default Calvano (2020) parameters ────────────────────────────────────
    MU    = 0.25
    A     = 2.0
    A0    = 0.0
    C     = 1.0
    ALPHA = 0.15
    DELTA = 0.95
    K     = 15       # price grid size
    BETA_BASE = 4e-6  # Calvano (2020) n=2 baseline

    def __init__(self, n: int = 2, beta: float = None):
        super().__init__()
        self.n    = n
        # Auto-scale exploration decay with n so exploration covers the
        # larger state space proportionally (Calvano 2020 recipe).
        # n=2 → 4e-6, n=3 → 2.67e-7, n=4 → 1.78e-8
        self.beta = beta if beta is not None else self.BETA_BASE * (self.K ** (2 - n))

        self.nash_price, self.mono_price = self._solve_equilibria()
        self.price_grid  = self._build_price_grid()
        self.profit_matrix = self._build_profit_matrix()

        self.nash_profit = float(self._profit_at(self.nash_price))
        self.mono_profit = float(self._profit_at(self.mono_price))
        self.nash_idx    = int(np.argmin(np.abs(self.price_grid - self.nash_price)))
        self.mono_idx    = int(np.argmin(np.abs(self.price_grid - self.mono_price)))

        self.observation_space = spaces.MultiDiscrete([self.K] * n)
        self.action_space      = spaces.MultiDiscrete([self.K] * n)

        self._state = np.zeros(n, dtype=np.int64)

    # ── Demand and profit ─────────────────────────────────────────────────────

    def demand(self, prices: np.ndarray) -> np.ndarray:
        e = np.exp((self.A - prices) / self.MU)
        return e / (e.sum() + np.exp(self.A0 / self.MU))

    def profits(self, prices: np.ndarray) -> np.ndarray:
        return (prices - self.C) * self.demand(prices)

    # ── Equilibrium solving ───────────────────────────────────────────────────

    def _foc_nash(self, p):
        d = self.demand(p)
        return 1.0 - (p - self.C) * (1.0 - d) / self.MU

    def _foc_mono(self, p):
        d     = self.demand(p)
        cross = np.sum((p - self.C) * d) - (p - self.C) * d
        return 1.0 - (p - self.C) * (1.0 - d) / self.MU + cross / self.MU

    def _solve_equilibria(self):
        p0     = np.ones(self.n) * 3.0 * self.C
        p_nash = float(fsolve(self._foc_nash, p0)[0])
        p_mono = float(fsolve(self._foc_mono, p0)[0])
        return p_nash, p_mono

    def _profit_at(self, price: float) -> float:
        return float(self.profits(np.full(self.n, price))[0])

    # ── Price grid ────────────────────────────────────────────────────────────

    def _build_price_grid(self) -> np.ndarray:
        inner = np.linspace(self.nash_price, self.mono_price, self.K - 2)
        step  = inner[1] - inner[0]
        return np.linspace(inner[0] - step, inner[-1] + step, self.K)

    # ── Profit matrix ─────────────────────────────────────────────────────────

    def _build_profit_matrix(self) -> np.ndarray:
        shape = tuple([self.K] * self.n) + (self.n,)
        PI    = np.zeros(shape)
        for idxs in iproduct(range(self.K), repeat=self.n):
            PI[idxs] = self.profits(self.price_grid[list(idxs)])
        return PI

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = np.zeros(self.n, dtype=np.int64)
        return self._state.copy(), {}

    def step(self, actions: np.ndarray):
        """
        actions : int array of shape (n,) — joint price indices.
        Returns (obs, rewards, terminated, truncated, info).
        obs     : int array (n,) — same as actions (state = last joint action)
        rewards : float array (n,) — per-agent profits
        """
        actions   = np.asarray(actions, dtype=np.int64)
        prices    = self.price_grid[actions]
        rewards   = self.profit_matrix[tuple(actions)].copy()
        self._state = actions.copy()

        info = {
            "prices":          prices,
            "actions":         actions,
            "collusion_index": self.collusion_index(rewards),
            "cs_both":         self.consumer_surplus_both(prices),
            "cs_bbx":          self.consumer_surplus_bbx(prices),
        }
        return self._state.copy(), rewards, False, False, info

    # ── Metrics ───────────────────────────────────────────────────────────────

    def collusion_index(self, profits: np.ndarray) -> float:
        pi = float(np.mean(profits))
        ci = (pi - self.nash_profit) / (self.mono_profit - self.nash_profit)
        return float(np.clip(ci, 0.0, 1.0))

    def consumer_surplus_both(self, prices: np.ndarray) -> float:
        val = sum(np.exp((self.A - p) / self.MU) for p in prices)
        return float(self.MU * np.log(val + np.exp(self.A0 / self.MU)))

    def consumer_surplus_bbx(self, prices: np.ndarray) -> float:
        p_min = float(np.min(prices))
        return float(self.MU * np.log(
            np.exp((self.A - p_min) / self.MU) + np.exp(self.A0 / self.MU)))

    def summary(self):
        print(f"CalvanoBertrandEnv  n={self.n}  k={self.K}  beta={self.beta:.1e}")
        print(f"  Nash:     price={self.nash_price:.4f}  profit={self.nash_profit:.4f}  idx={self.nash_idx}")
        print(f"  Monopoly: price={self.mono_price:.4f}  profit={self.mono_profit:.4f}  idx={self.mono_idx}")
