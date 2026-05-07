"""
envs/cournot_imperfect.py — Cournot oligopoly with imperfect monitoring.

Adapted from Calvano, Calzolari, Denicolò, Pastorello (2021), Section 3.

Demand          : p_t = d_t − Σ q_i,    d_t ∈ {290, 310} i.i.d., equiprobable
Reward (raw)    : π_i = p_t · q_i
Reward (norm)   : π_i / mono_profit_raw  (returned in step())
State           : previous-period market price p_{t-1}  (discrete, 37 levels for n=2)
Action          : quantity q_i ∈ {70, 72.5, …, 105}    (K=15)

Imperfect monitoring: agents observe only the realised price, not opponents'
quantities. Demand is stochastic so a low price can be either a deviation
or a negative shock. Calvano et al. (2021) report CI ≈ 0.763 in this regime
vs ≈ 0.847 with perfect monitoring.

For belief injection / equilibrium reference we use the *mean* demand
d̄ = 300, giving Nash q* = 100, mono q = 75 — same as the perfect-monitoring env.
"""

import numpy as np
import gymnasium as gym


class CournotImperfectEnv(gym.Env):

    metadata = {}

    K          = 15
    Q_MIN      = 70.0
    Q_MAX      = 105.0
    Q_STEP     = 2.5
    D_LEVELS   = (290.0, 310.0)
    BETA_BASE  = 4e-6

    def __init__(self, n: int = 2, beta: float = None,
                 normalize: bool = True, seed: int = None):
        super().__init__()
        self.n         = n
        self.beta      = beta if beta is not None else self.BETA_BASE
        self.normalize = normalize

        self.quantity_grid = np.linspace(self.Q_MIN, self.Q_MAX, self.K)
        self.price_grid    = self.quantity_grid          # alias for compatibility

        # Use mean demand for equilibrium references (matches paper)
        self.D_mean   = float(np.mean(self.D_LEVELS))
        self.nash_q   = self.D_mean / (self.n + 1)
        self.mono_q   = self.D_mean / (2 * self.n)
        self.nash_idx = int(np.argmin(np.abs(self.quantity_grid - self.nash_q)))
        self.mono_idx = int(np.argmin(np.abs(self.quantity_grid - self.mono_q)))

        self.nash_profit_raw = self.nash_q * (self.D_mean - self.n * self.nash_q)
        self.mono_profit_raw = self.mono_q * (self.D_mean - self.n * self.mono_q)

        self._scale       = self.mono_profit_raw if normalize else 1.0
        self.nash_profit  = self.nash_profit_raw / self._scale
        self.mono_profit  = self.mono_profit_raw / self._scale

        # Build the discrete price state space
        self.price_levels = self._build_price_levels()
        self.num_states   = len(self.price_levels)
        self.state_shape  = (self.num_states,)
        self._price_to_idx = {round(p, 6): i for i, p in enumerate(self.price_levels)}

        # Profit matrix at *mean* demand (used for Q-table init / belief injection)
        self.profit_matrix = self._build_profit_matrix()

        self.action_space      = gym.spaces.MultiDiscrete([self.K] * self.n)
        self.observation_space = gym.spaces.Discrete(self.num_states)

        self._rng = np.random.default_rng(seed)
        self.state = 0

    # ── State / profit machinery ─────────────────────────────────────────────

    def _build_price_levels(self):
        """All possible market prices given quantity grid × demand support."""
        levels = set()
        # Total quantity grid: n agents each pick from K levels
        # Coarse-grained: every q is multiple of Q_STEP, so total_q is too
        total_q_min = self.n * self.Q_MIN
        total_q_max = self.n * self.Q_MAX
        for d in self.D_LEVELS:
            tq = total_q_min
            while tq <= total_q_max + 1e-9:
                p = max(0.0, d - tq)
                levels.add(round(p, 6))
                tq += self.Q_STEP
        return sorted(levels)

    def _price_idx(self, p: float) -> int:
        return self._price_to_idx[round(float(p), 6)]

    def _build_profit_matrix(self) -> np.ndarray:
        """Profits at deterministic mean demand — for Q-init and belief injection."""
        PI = np.zeros([self.K] * self.n + [self.n], dtype=np.float64)
        for idxs in np.ndindex(*([self.K] * self.n)):
            quantities = self.quantity_grid[list(idxs)]
            total_q    = float(quantities.sum())
            p          = max(0.0, self.D_mean - total_q)
            PI[tuple(idxs)] = (p * quantities) / self._scale
        return PI

    def collusion_index(self, profits) -> float:
        avg_pi = float(np.mean(profits))
        denom  = self.mono_profit - self.nash_profit
        if denom <= 0:
            return 0.0
        return (avg_pi - self.nash_profit) / denom

    # ── gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Random initial price state
        self.state = int(self._rng.integers(0, self.num_states))
        return self.state, {}

    def step(self, actions):
        actions    = np.asarray(actions, dtype=np.int64)
        quantities = self.quantity_grid[actions]
        d          = float(self._rng.choice(self.D_LEVELS))
        total_q    = float(quantities.sum())
        market_p   = max(0.0, d - total_q)
        rewards    = (market_p * quantities) / self._scale

        next_state = self._price_idx(market_p)

        info = {
            "raw_rewards":  rewards.copy(),
            # 'prices' carries per-agent quantity (for logging compatibility)
            "prices":       quantities.copy(),
            "market_price": market_p / self._scale,
            "demand":       d,
        }
        self.state = next_state
        return next_state, rewards, False, False, info

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def summary(self):
        print(f"CournotImperfectEnv  n={self.n}  K={self.K}  states={self.num_states}")
        print(f"  Demand levels: {self.D_LEVELS}  (mean={self.D_mean})")
        print(f"  Nash:     q*={self.nash_q:.2f}  π*={self.nash_profit_raw:.2f} "
              f"(norm={self.nash_profit:.4f})  idx={self.nash_idx}")
        print(f"  Monopoly: q*={self.mono_q:.2f}  π*={self.mono_profit_raw:.2f} "
              f"(norm={self.mono_profit:.4f})  idx={self.mono_idx}")
