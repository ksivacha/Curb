"""
agents/tabular_q.py — Tabular Q-learning for n symmetric agents.

Mirrors the RLSupervisorQPricingWrapper pattern from the supplementary material:
  - In-house Q-tables (numpy arrays indexed by state tuple)
  - Exponential epsilon decay: ε = exp(−β · t)
  - Optional belief injection at interval K_belief
  - Convergence detection: tstable consecutive steps of stable joint argmax

Usage
-----
    env   = CURBWrapper(CalvanoBertrandEnv(n=2), cfg)
    agent = TabularQAgent(env, cfg)
    results = agent.run(n_steps=3_000_000, log_every=10_000)
"""

import numpy as np


class TabularQAgent:
    """Tabular Q-learning driver for a multi-agent gymnasium Env."""

    def __init__(self, env, cfg: dict):
        self.n     = env.n
        self.alpha = cfg.get("alpha",     0.15)
        self.delta = cfg.get("delta",     0.95)
        # Fall back to env.beta so CalvanoBertrandEnv's auto-scaled value is used
        self.beta  = cfg.get("beta", getattr(env, "beta", 4e-6))
        self.tstable = cfg.get("tstable", 100_000)

        self.use_belief = cfg.get("use_belief", False)
        self.K_belief   = cfg.get("K_belief",  100)
        self.M_belief   = cfg.get("M_belief",  5)

        # Action space size and Q-table shape
        self.k = env.K
        # Envs may expose `state_shape` if their state space is not the default
        # joint-action grid (e.g. Cournot with imperfect monitoring uses a single
        # price-index dim instead of (K,)*n).
        self.state_shape = tuple(getattr(env, "state_shape",
                                         tuple([env.K] * env.n)))

        self.Q = self._init_q(env)
        self._rng = np.random.default_rng(cfg.get("seed", None))

    # ── Q-table initialisation ────────────────────────────────────────────────

    def _init_q(self, env) -> list:
        """
        Calvano (2020) analytical initialisation:
            Q[i, s, a] = E_{opp}[π_i(a)] / (1 − δ)
        averaged over all opponent action combinations, independent of state s.
        """
        Q  = []
        pm = env.profit_matrix
        for i in range(self.n):
            q = np.zeros(self.state_shape + (self.k,), dtype=np.float64)
            opp_axes = tuple(j for j in range(self.n) if j != i)
            pi_avg   = np.mean(pm[..., i], axis=opp_axes)
            q[...]   = pi_avg / (1.0 - self.delta)
            Q.append(q)
        return Q

    # ── Action selection ──────────────────────────────────────────────────────

    def _state_key(self, obs) -> tuple:
        if hasattr(obs, "__len__"):
            return tuple(int(x) for x in obs)
        return (int(obs),)

    def _greedy(self, sk: tuple) -> np.ndarray:
        return np.array([int(np.argmax(self.Q[j][sk])) for j in range(self.n)],
                        dtype=np.int64)

    def _eps_greedy(self, sk: tuple, eps: float) -> np.ndarray:
        acts = []
        for j in range(self.n):
            if self._rng.random() < eps:
                acts.append(int(self._rng.integers(0, self.k)))
            else:
                acts.append(int(np.argmax(self.Q[j][sk])))
        return np.array(acts, dtype=np.int64)

    # ── Q-table update ────────────────────────────────────────────────────────

    def _update(self, sk: tuple, actions: np.ndarray,
                rewards: np.ndarray, nsk: tuple):
        for j in range(self.n):
            a      = int(actions[j])
            target = rewards[j] + self.delta * float(np.max(self.Q[j][nsk]))
            self.Q[j][sk + (a,)] += self.alpha * (target - self.Q[j][sk + (a,)])

    # ── Belief injection ──────────────────────────────────────────────────────

    def _inject_beliefs(self, env):
        """
        Inject synthetic TD experiences modelling a non-retaliating opponent
        (Algorithm 1 line 14: "no retaliation" experiences).

        Inject (rand_state, dev_idx, r_dev, coop_state) where dev_idx is Nash-
        adjacent and the opponent stays at monopoly price (no retaliation),
        giving agent j the maximum short-term gain with no punishment continuation.
        """
        base_env = getattr(env, "unwrapped", env)
        mono_idx = getattr(base_env, "mono_idx", self.k - 1)
        nash_idx = getattr(base_env, "nash_idx", 1)
        dev_idx  = nash_idx

        for j in range(self.n):
            # Counterfactual next state: j at Nash, opponents at monopoly (no retaliation).
            nsk        = tuple(mono_idx if i != j else dev_idx
                               for i in range(self.n))
            r_dev      = float(base_env.profit_matrix[nsk][j])
            max_q_next = float(np.max(self.Q[j][nsk]))
            target     = r_dev + self.delta * max_q_next

            # Broadcast the belief across M_belief random source states so the
            # "defection pays" signal propagates across the whole Q-table.
            s_arr = self._rng.integers(0, self.k, size=(self.M_belief, self.n))
            for m in range(self.M_belief):
                s_key = tuple(int(x) for x in s_arr[m])
                self.Q[j][s_key + (dev_idx,)] += self.alpha * (
                    target - self.Q[j][s_key + (dev_idx,)])

    # ── Convergence check ─────────────────────────────────────────────────────

    def _converged(self, stable_count: int) -> bool:
        return stable_count >= self.tstable

    # ── Main training loop ────────────────────────────────────────────────────

    def run(self, env, n_steps: int, log_every: int = 10_000) -> dict:
        """
        Drive training loop.

        Parameters
        ----------
        env        : gymnasium Env (multi-agent, possibly wrapped with CURBWrapper)
        n_steps    : total environment steps
        log_every  : how often to record metrics

        Returns
        -------
        dict with keys: steps, ci (or coop_rate), q_diff, final_metric, converged_at
        """
        obs, _  = env.reset()
        sk      = self._state_key(obs)

        log_steps, log_ci, log_tv = [], [], []
        log_prices = []
        prev_actions = None
        stable_cnt   = 0
        converged_at = n_steps

        # Windowed accumulators (matches Calvano 2020 CI calculation)
        win_profits = np.zeros((log_every, self.n))
        win_prices  = np.zeros((log_every, self.n))
        win_tv      = np.zeros((log_every, self.n))
        win_ptr     = 0

        for t in range(1, n_steps + 1):
            eps     = np.exp(-self.beta * t)
            actions = self._eps_greedy(sk, eps)

            obs2, rewards, _, _, info = env.step(actions)
            nsk = self._state_key(obs2)

            # shaped rewards drive Q-update; raw profits used only for CI
            raw = info.get("raw_rewards", rewards)
            self._update(sk, actions, rewards, nsk)

            # Belief injection
            if self.use_belief and t % self.K_belief == 0:
                self._inject_beliefs(env)

            # Convergence tracking
            greedy = self._greedy(sk)
            if prev_actions is not None and np.array_equal(greedy, prev_actions):
                stable_cnt += 1
            else:
                stable_cnt = 0
            prev_actions = greedy

            if stable_cnt == self.tstable:
                converged_at = t

            # Windowed accumulation
            win_profits[win_ptr] = raw
            win_prices [win_ptr] = info.get("prices", np.zeros(self.n))
            win_tv     [win_ptr] = info.get("tv", np.zeros(self.n))
            win_ptr = (win_ptr + 1) % log_every

            if t % log_every == 0:
                log_steps.append(t)
                avg_pi   = win_profits.mean(0)
                base_env = getattr(env, "unwrapped", env)
                log_ci.append(base_env.collusion_index(avg_pi))
                log_tv.append(win_tv.mean(0).copy())
                log_prices.append(win_prices.mean(0).copy())

            sk = nsk

        # Final Q-value differences (cooperation vs defection preference)
        q_diff = self._q_diff()

        tv_arr  = np.array(log_tv)     if log_tv     else np.zeros((0, self.n))
        p_arr   = np.array(log_prices) if log_prices else np.zeros((0, self.n))
        warmup  = int(0.9 * len(log_steps))
        ci_arr   = np.array(log_ci)
        final_ci = float(ci_arr[warmup:].mean()) if len(ci_arr) > 0 else 0.0
        return {
            "steps":     np.array(log_steps),
            "ci":        ci_arr,
            "prices":    p_arr,
            "tv":        tv_arr,
            "q_diff":    q_diff,
            "Q":         self.Q,
            "final_ci":  final_ci,
            "converged": converged_at < n_steps,
            "t_conv":    converged_at,
        }

    # ── Forced-defection test ─────────────────────────────────────────────────

    def forced_defection_test(self, env, n_steps=200, defect_step=50,
                              defect_action=2, punish_window=20) -> dict:
        """
        Greedy episode where agent 0 is forced to defect at defect_step.
        Returns punished (bool) and recovery_t (steps to return to Nash).
        """
        base    = getattr(env, "unwrapped", env)
        n       = self.n
        nash_idx = getattr(base, "nash_idx", 1)
        mono_idx = getattr(base, "mono_idx", self.k - 1)

        s          = np.full(n, mono_idx, dtype=np.int64)
        prices_log = np.zeros((n_steps, n))
        acts_log   = np.zeros((n_steps, n), dtype=np.int64)

        for t in range(n_steps):
            a = self._greedy(tuple(s))
            if t == defect_step:
                a = a.copy()
                a[0] = defect_action
            prices_log[t] = base.price_grid[a]
            acts_log[t]   = a
            s = a.copy()

        post = acts_log[defect_step + 1: defect_step + 1 + punish_window, 1:]
        punished = bool(np.any(post < nash_idx))

        recovery_t = -1
        for t in range(defect_step + 1, n_steps):
            if np.all(acts_log[t] >= nash_idx):
                recovery_t = t - defect_step
                break

        return {
            "prices":     prices_log,
            "actions":    acts_log,
            "punished":   punished,
            "recovery_t": recovery_t,
        }

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _q_diff(self) -> np.ndarray:
        """
        Per-state Q(low-price) − Q(high-price) averaged across agents.
        Positive = agents prefer the competitive (low-price) action.
        """
        diffs = np.zeros(np.prod(self.state_shape))
        for j in range(self.n):
            q = self.Q[j].reshape(-1, self.k)
            diffs += q[:, 0] - q[:, -1]
        return diffs / self.n
