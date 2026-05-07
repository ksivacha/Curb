"""
agents/dqn_agent.py — Independent multi-agent DQN for the Calvano environment.

Architecture mirrors Hettich (2021) / mesjou/price_simulator:
  - Custom PyTorch DQN (no SB3) — one network per agent
  - 2×32 hidden layers, ReLU, linear output, MSE loss, Adam
  - Exponential epsilon decay: eps = exp(−β·t)  [same as tabular Q]
  - Hard target-network copy every C steps
  - Replay buffer 10K, batch 32, γ=0.95, lr=0.001
  - State: normalized competitor prices in [0, 1]^n

Belief injection targets the cooperative (mono) state directly.

Usage
-----
    env   = CalvanoBertrandEnv(n=2)
    agent = MultiAgentDQN(env, cfg={})
    r     = agent.run(env, n_steps=1_000_000, log_every=10_000)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ── Q-network ─────────────────────────────────────────────────────────────────

class _QNet(nn.Module):
    def __init__(self, n_obs: int, n_actions: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class _ReplayBuffer:
    def __init__(self, maxlen: int, rng: np.random.Generator):
        self._buf = deque(maxlen=maxlen)
        self._rng = rng

    def add(self, s, a, r, s2):
        self._buf.append((s.copy(), int(a), float(r), s2.copy()))

    def sample(self, batch: int):
        idx   = self._rng.choice(len(self._buf), size=batch, replace=False)
        items = [self._buf[i] for i in idx]
        s  = np.array([x[0] for x in items], dtype=np.float32)
        a  = np.array([x[1] for x in items], dtype=np.int64)
        r  = np.array([x[2] for x in items], dtype=np.float32)
        s2 = np.array([x[3] for x in items], dtype=np.float32)
        return s, a, r, s2

    def __len__(self):
        return len(self._buf)


# ── Agent ─────────────────────────────────────────────────────────────────────

class MultiAgentDQN:
    """Independent DQN — one custom PyTorch network per agent."""

    def __init__(self, env, cfg: dict):
        self.n = env.n
        self.k = env.K
        pg = getattr(env, "price_grid",
                     getattr(getattr(env, "unwrapped", env), "price_grid", None))
        self.price_grid = np.asarray(pg)
        self.p_min = float(self.price_grid.min())
        self.p_max = float(self.price_grid.max())

        hidden   = cfg.get("hidden",   32)
        lr       = cfg.get("lr",       0.001)
        buf_size = cfg.get("buf_size", 10_000)
        self.batch  = cfg.get("batch",  32)
        self.gamma  = cfg.get("gamma",  0.95)
        self.C      = cfg.get("C",      100)    # hard target copy interval (steps)
        self.beta   = cfg.get("beta",   4e-6)   # epsilon decay: eps = exp(-beta*t)

        seed = cfg.get("seed", None)
        self._rng = np.random.default_rng(seed)

        self.nets    = [_QNet(self.n, self.k, hidden) for _ in range(self.n)]
        self.targets = [_QNet(self.n, self.k, hidden) for _ in range(self.n)]
        for i in range(self.n):
            self.targets[i].load_state_dict(self.nets[i].state_dict())
            self.targets[i].eval()
        self.optims  = [optim.Adam(self.nets[i].parameters(), lr=lr)
                        for i in range(self.n)]
        self.buffers = [
            _ReplayBuffer(buf_size, np.random.default_rng(
                seed + i if seed is not None else None))
            for i in range(self.n)
        ]

        self.use_belief = cfg.get("use_belief", False)
        self.K_belief   = cfg.get("K_belief", 100)
        self.M_belief   = cfg.get("M_belief", 5)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_state(self, obs) -> np.ndarray:
        prices = self.price_grid[np.asarray(obs, dtype=np.int64)]
        return ((prices - self.p_min) /
                (self.p_max - self.p_min + 1e-8)).astype(np.float32)

    # ── Q-update ──────────────────────────────────────────────────────────────

    def _update(self, i: int):
        if len(self.buffers[i]) < self.batch:
            return
        s, a, r, s2 = self.buffers[i].sample(self.batch)
        s_t  = torch.FloatTensor(s)
        a_t  = torch.LongTensor(a)
        r_t  = torch.FloatTensor(r)
        s2_t = torch.FloatTensor(s2)

        with torch.no_grad():
            next_q = self.targets[i](s2_t).max(1).values
            target = r_t + self.gamma * next_q

        current_q = self.nets[i](s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(current_q, target)

        self.optims[i].zero_grad()
        loss.backward()
        self.optims[i].step()

    # ── Belief injection ──────────────────────────────────────────────────────

    def _inject_beliefs(self, base):
        """Inject M_belief no-retaliation transitions per agent into the replay
        buffer. Source states are sampled uniformly at random across the joint
        action space so the belief propagates across the Q-network, not just
        the all-monopoly state. Counterfactual next-state: agent j at Nash,
        opponents at monopoly (no retaliation)."""
        mono_idx = getattr(base, "mono_idx", self.k - 1)
        nash_idx = getattr(base, "nash_idx", 1)
        dev_idx  = nash_idx

        for j in range(self.n):
            no_ret_obs = np.array([mono_idx if i != j else dev_idx
                                   for i in range(self.n)], dtype=np.int64)
            r_dev      = float(base.profit_matrix[tuple(no_ret_obs)][j])
            next_state = self._to_state(no_ret_obs)

            src_obs_arr = self._rng.integers(0, self.k, size=(self.M_belief, self.n))
            for m in range(self.M_belief):
                src_state = self._to_state(src_obs_arr[m].astype(np.int64))
                self.buffers[j].add(src_state, dev_idx, r_dev, next_state)

    # ── Main training loop ────────────────────────────────────────────────────

    def run(self, env, n_steps: int, log_every: int = 10_000) -> dict:
        torch.set_num_threads(1)
        obs, _ = env.reset()
        s      = self._to_state(obs)
        base   = getattr(env, "unwrapped", env)

        log_steps, log_ci, log_prices, log_tv = [], [], [], []

        win_profits = np.zeros((log_every, self.n))
        win_prices  = np.zeros((log_every, self.n))
        win_tv      = np.zeros((log_every, self.n))
        win_ptr     = 0

        for t in range(1, n_steps + 1):
            eps = float(np.exp(-self.beta * t))

            actions = []
            for i in range(self.n):
                if self._rng.random() < eps:
                    actions.append(int(self._rng.integers(0, self.k)))
                else:
                    with torch.no_grad():
                        q = self.nets[i](torch.FloatTensor(s).unsqueeze(0))
                    actions.append(int(q.argmax().item()))
            actions = np.array(actions, dtype=np.int64)

            obs2, rewards, _, _, info = env.step(actions)
            s2  = self._to_state(obs2)
            raw = info.get("raw_rewards", rewards)

            for i in range(self.n):
                self.buffers[i].add(s, actions[i], float(rewards[i]), s2)
                self._update(i)

            # Hard target copy
            if t % self.C == 0:
                for i in range(self.n):
                    self.targets[i].load_state_dict(self.nets[i].state_dict())

            if self.use_belief and t % self.K_belief == 0:
                self._inject_beliefs(base)

            win_profits[win_ptr] = raw
            win_prices [win_ptr] = info.get("prices", base.price_grid[actions])
            win_tv     [win_ptr] = info.get("tv",      np.zeros(self.n))
            win_ptr = (win_ptr + 1) % log_every

            if t % log_every == 0:
                avg_pi = win_profits.mean(0)
                log_ci.append(base.collusion_index(avg_pi))
                log_steps.append(t)
                log_prices.append(win_prices.mean(0).copy())
                log_tv.append(win_tv.mean(0).copy())

            s = s2

        ci_arr = np.array(log_ci)
        p_arr  = np.array(log_prices) if log_prices else np.zeros((0, self.n))
        tv_arr = np.array(log_tv)     if log_tv     else np.zeros((0, self.n))
        warmup = int(0.9 * len(log_steps))

        return {
            "steps":     np.array(log_steps),
            "ci":        ci_arr,
            "prices":    p_arr,
            "tv":        tv_arr,
            "final_ci":  float(ci_arr[warmup:].mean()) if len(ci_arr) > warmup else 0.0,
            "converged": False,
            "t_conv":    n_steps,
        }
