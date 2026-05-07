"""
wrappers/curb.py — CURB reward shaping as a gymnasium Wrapper.

Applies TV-based reward shaping to the multi-agent Calvano environment:
    r̃_j = r_j − λ · max(0, TV_j − ε)
    TV_j = max_{i≠j} TV(i→j)
    TV(i→j) = 0.5 · Σ |P(a_j^t | a_i^{t-1} < θ) − P(a_j^t | a_i^{t-1} ≥ θ)|

LAG-1 conditioning: agent j's action at t is conditioned on opponent i's
action at t-1, capturing the temporal causality of punishment strategies.

cfg keys
--------
  lam          : float  — penalty weight λ
  tv_threshold : float  — dead-zone ε (default 0.0)
  W            : int    — rolling window length (default 500)
"""

import numpy as np
import gymnasium as gym


class _RollingBuffer:
    """Circular buffer of joint actions for n agents."""

    def __init__(self, n: int, W: int):
        self.n     = n
        self.W     = W
        self._buf  = np.zeros((W, n), dtype=np.int32)
        self._ptr  = 0
        self._full = False

    def push(self, actions: np.ndarray):
        self._buf[self._ptr] = actions
        self._ptr = (self._ptr + 1) % self.W
        if self._ptr == 0:
            self._full = True

    def get(self) -> np.ndarray:
        n = self.W if self._full else self._ptr
        return self._buf[:n]


def _tv_calvano(buf: np.ndarray, agent_j: int, k: int) -> float:
    """
    TV penalty for agent j in Calvano (paper-faithful, lag-1 conditioning):
      max_{i≠j} TV(i→j)
    TV(i→j) = 0.5 * |P(a_j^t | a_i^{t-1} < θ) - P(a_j^t | a_i^{t-1} >= θ)|
    The lag-1 partition matches the paper's definition of π_j conditioned on
    agent i's previous action (since state = last joint action).
    Only counts when j plays lower when i was low (punishment direction).
    """
    n      = buf.shape[1]
    max_tv = 0.0
    for i in range(n):
        if i == agent_j:
            continue
        theta      = int(np.median(buf[:, i]))
        # Lag-1: for row r, condition on agent i's action at row r-1.
        # Row 0 has no predecessor, so we pad with theta (becomes "above").
        prev_i     = np.concatenate([[theta], buf[:-1, i]])
        below_mask = prev_i < theta
        n_below    = below_mask.sum()
        n_above    = len(buf) - n_below
        if n_below < 2 or n_above < 2:
            continue
        acts_below = buf[below_mask,  agent_j].astype(float)
        acts_above = buf[~below_mask, agent_j].astype(float)
        if acts_below.mean() >= acts_above.mean():
            continue
        p_low  = np.bincount(buf[below_mask,  agent_j], minlength=k).astype(float)
        p_high = np.bincount(buf[~below_mask, agent_j], minlength=k).astype(float)
        p_low  /= p_low.sum()
        p_high /= p_high.sum()
        max_tv = max(max_tv, 0.5 * float(np.sum(np.abs(p_low - p_high))))
    return max_tv


class CURBWrapper(gym.Wrapper):
    """
    Applies TV-based reward shaping to a multi-agent gymnasium Env.

    The wrapped env must:
      - Accept joint actions (array of shape (n,))
      - Return rewards as an array of shape (n,)
      - Have env.n set (number of agents)
    """

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.lam        = cfg.get("lam",          0.0)
        self.tv_eps     = cfg.get("tv_threshold",  0.0)
        W               = cfg.get("W",            500)
        self.n          = env.n
        _k = getattr(env, "K", None)
        if _k is not None:
            self.K = _k                              # forwarded for TabularQAgent
        k               = getattr(env, "K", 2)
        self.k          = k
        self._buf       = _RollingBuffer(self.n, W)
        # Forward profit_matrix and equilibrium attrs for belief injection
        for attr in ("nash_idx", "mono_idx", "nash_profit", "mono_profit",
                     "price_grid", "profit_matrix"):
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self, **kwargs):
        self._buf = _RollingBuffer(self.env.n, self._buf.W)
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        info["raw_rewards"] = rewards.copy()   # unshaped profits for CI calc

        self._buf.push(np.asarray(actions, dtype=np.int32))
        buf = self._buf.get()

        tv_vals = np.zeros(self.env.n)
        if len(buf) >= 10:
            for j in range(self.env.n):
                tv_vals[j] = _tv_calvano(buf, j, self.k)
        info["tv"] = tv_vals

        if self.lam > 0.0 and len(buf) >= 10:
            shaped = rewards.copy()
            for j in range(self.env.n):
                shaped[j] -= self.lam * max(0.0, tv_vals[j] - self.tv_eps)
            return obs, shaped, term, trunc, info

        return obs, rewards, term, trunc, info
