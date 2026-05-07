"""
wrappers/buybox.py — Buy-box demand steering wrappers for PDP and DPDP.

Implements Johnson, Rhodes & Wildenbeest (2023) platform design rules
as gymnasium wrappers over CalvanoBertrandEnv.

PDP  : Each period, the lowest-price agent wins the buy-box (γ fraction
       of consumers see only that agent).
DPDP : Current occupant keeps the buy-box if they did not raise price
       AND were not undercut by more than ADV; otherwise PDP applies.

With γ=1 (default), all consumers are steered to the buy-box winner.

cfg keys
--------
  dp_type : str   — "pdp" or "dpdp"
  gamma   : float — buy-box intensity (default 1.0)
  adv     : float — DPDP pricing advantage threshold (default 0.3)
"""

import numpy as np
import gymnasium as gym


class BuyBoxWrapper(gym.Wrapper):
    """
    Buy-box demand steering wrapper for n=2 Calvano environment.

    During training, demand is steered toward the buy-box winner.
    The observation (last joint price indices) is unchanged, so
    Q-tables transfer directly to a normal Calvano test environment.
    """

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.dp_type = cfg.get("dp_type", "pdp")
        self.gamma   = cfg.get("gamma",   1.0)
        self.adv     = cfg.get("adv",     0.3)
        self.n       = env.n

        # Forward Calvano attrs so TabularQAgent can initialise correctly
        for attr in ("K", "beta", "nash_idx", "mono_idx",
                     "nash_profit", "mono_profit",
                     "price_grid", "profit_matrix"):
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

        self._bbx_occupant = 0
        self._prev_prices  = None   # None until first step has occurred

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, **kwargs):
        self._bbx_occupant = 0
        self._prev_prices  = None
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)

        base   = self.env.unwrapped
        prices = base.price_grid[np.asarray(actions, dtype=np.int64)]

        winner = self._bbx_winner(prices)

        # Buy-box demand for winner (competing only with outside good)
        p_w   = float(prices[winner])
        e_w   = np.exp((base.A - p_w) / base.MU)
        e_0   = np.exp(base.A0 / base.MU)
        d_bbx = e_w / (e_w + e_0)

        # Normal (no buy-box) demands
        normal_d = base.demand(prices)

        # Shaped rewards under buy-box
        shaped = np.empty(self.n)
        for j in range(self.n):
            if j == winner:
                d_j = self.gamma * d_bbx + (1.0 - self.gamma) * normal_d[j]
            else:
                d_j = (1.0 - self.gamma) * normal_d[j]
            shaped[j] = (prices[j] - base.C) * d_j

        # Preserve unshaped profits so CI can be computed correctly
        info["raw_rewards"] = rewards.copy()
        info["tv"]          = np.zeros(self.n)
        info["bbx_winner"]  = winner

        # Update DPDP internal state
        self._prev_prices  = prices.copy()
        self._bbx_occupant = winner

        return obs, shaped, term, trunc, info

    # ── Buy-box rule ──────────────────────────────────────────────────────────

    def _bbx_winner(self, prices: np.ndarray) -> int:
        if self.dp_type == "pdp":
            return int(np.argmin(prices))

        # DPDP
        if self._prev_prices is None:
            # No history on first step of an episode → default to PDP
            return int(np.argmin(prices))

        occupant     = self._bbx_occupant
        non_occupant = 1 - occupant          # valid only for n=2

        price_not_raised = prices[occupant] <= self._prev_prices[occupant]
        undercut_diff    = prices[occupant] - prices[non_occupant]
        # Occupant keeps slot unless rival undercuts by > adv
        if price_not_raised and undercut_diff < self.adv:
            return occupant
        else:
            return int(np.argmin(prices))
