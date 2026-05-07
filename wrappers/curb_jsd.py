"""
wrappers/curb_jsd.py — CURB wrapper with pluggable divergence metric.

Identical to curb.py except the step penalty uses either TV or JSD,
selected via cfg["metric"] = "tv" (default) | "jsd".

JSD scaling is controlled by cfg["jsd_normalize"]:
  True  (default): JSD ÷ ln(2) → [0, 1]  → λ directly comparable to TV
  False:           JSD raw, in nats → [0, ln 2 ≈ 0.693] (~31 % weaker
                   penalty than TV at matched λ)

Used only for the JSD ablation experiment — the main calvano_sweep.py
still uses the original curb.py.
"""

import numpy as np
import gymnasium as gym


# ── Rolling buffer ────────────────────────────────────────────────────────────

class _RollingBuffer:
    def __init__(self, n: int, W: int):
        self.n    = n
        self.W    = W
        self._buf = np.zeros((W, n), dtype=np.int32)
        self._ptr = 0
        self._full = False

    def push(self, actions: np.ndarray):
        self._buf[self._ptr] = actions
        self._ptr = (self._ptr + 1) % self.W
        if self._ptr == 0:
            self._full = True

    def get(self) -> np.ndarray:
        n = self.W if self._full else self._ptr
        return self._buf[:n]


# ── TV (original) ─────────────────────────────────────────────────────────────

def _tv_calvano(buf: np.ndarray, agent_j: int, k: int) -> float:
    """LAG-1 TV: P(a_j^t | a_i^{t-1} < θ) vs P(a_j^t | a_i^{t-1} ≥ θ)."""
    n      = buf.shape[1]
    max_tv = 0.0
    for i in range(n):
        if i == agent_j:
            continue
        theta      = int(np.median(buf[:, i]))
        # Lag-1: pad row 0's predecessor with theta (treats first row as "above")
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


# ── JSD (ablation) ────────────────────────────────────────────────────────────

def _jsd_calvano(buf: np.ndarray, agent_j: int, k: int,
                 eps: float = 1e-8) -> float:
    """
    LAG-1 JSD penalty, raw in nats (range [0, ln 2] ≈ [0, 0.693]).
    Same lag-1 conditioning, median split, and punishment-direction filter as TV.
    Laplace smoothing prevents log(0) on sparse price bins.

    Caller is responsible for any scaling (e.g. divide by ln 2 to put on [0,1]
    so λ is directly comparable to TV).
    """
    n       = buf.shape[1]
    max_jsd = 0.0
    for i in range(n):
        if i == agent_j:
            continue
        theta      = int(np.median(buf[:, i]))
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
        p_low  = (p_low  + eps) / (p_low  + eps).sum()
        p_high = (p_high + eps) / (p_high + eps).sum()
        m      = 0.5 * (p_low + p_high)
        jsd    = 0.5 * (np.sum(p_low  * np.log(p_low  / m)) +
                        np.sum(p_high * np.log(p_high / m)))
        max_jsd = max(max_jsd, float(jsd))
    return max_jsd


# ── Wrapper ───────────────────────────────────────────────────────────────────

class CURBJSDWrapper(gym.Wrapper):
    """
    Drop-in replacement for CURBWrapper with selectable divergence metric.

    cfg keys (same as curb.py, plus):
      metric         : "tv" (default) | "jsd"
      jsd_normalize  : bool — divide JSD by ln(2) so it lives in [0,1].
                       Default True so λ is directly comparable to TV.
                       Only meaningful when metric == "jsd".
    """

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.lam            = cfg.get("lam",            0.0)
        self.tv_eps         = cfg.get("tv_threshold",    0.0)
        self.metric         = cfg.get("metric",          "tv")
        self.jsd_normalize  = cfg.get("jsd_normalize",   True)
        W                   = cfg.get("W",               500)
        self.n        = env.n
        k             = getattr(env, "K", 2)
        self.k        = k
        if hasattr(env, "K"):
            self.K = env.K
        self._buf     = _RollingBuffer(self.n, W)
        for attr in ("nash_idx", "mono_idx", "nash_profit", "mono_profit",
                     "price_grid", "profit_matrix"):
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self, **kwargs):
        self._buf = _RollingBuffer(self.env.n, self._buf.W)
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        info["raw_rewards"] = rewards.copy()

        self._buf.push(np.asarray(actions, dtype=np.int32))
        buf = self._buf.get()

        tv_vals = np.zeros(self.env.n)
        if len(buf) >= 10:
            ln2 = float(np.log(2))
            for j in range(self.env.n):
                if self.metric == "jsd":
                    val = _jsd_calvano(buf, j, self.k)
                    if self.jsd_normalize:
                        val = val / ln2          # → [0, 1] like TV
                    tv_vals[j] = val
                else:
                    tv_vals[j] = _tv_calvano(buf, j, self.k)
        info["tv"] = tv_vals

        if self.lam > 0.0 and len(buf) >= 10:
            shaped = rewards.copy()
            for j in range(self.env.n):
                shaped[j] -= self.lam * max(0.0, tv_vals[j] - self.tv_eps)
            return obs, shaped, term, trunc, info

        return obs, rewards, term, trunc, info
