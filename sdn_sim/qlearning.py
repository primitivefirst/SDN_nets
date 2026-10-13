"""Tabular Q-learning baseline.

State = (src_id, dst_id, congestion_bucket), actions = k candidate paths.
Small and fast; a starting baseline before DQN/GNN agents.
"""
import numpy as np


class QLearningAgent:
    def __init__(self, k_paths, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05):
        self.k = k_paths
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.Q = {}  # state -> np.array(k)

    def _q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.k)
        return self.Q[state]

    @staticmethod
    def _state_of(env):
        s, t, _d = env.cur_flow
        return (int(s), int(t), env.global_congestion_bucket(),
                env.sp_path_congestion_bucket(s, t))

    def act(self, env, eps, rng):
        s0, t0, _d0 = env.cur_flow
        cands = len(env.paths[(s0, t0)])
        if cands == 0:
            return 0  # disconnected: env drops the flow
        if rng.random() < eps:
            return int(rng.integers(min(self.k, cands)))
        q = self._q(self._state_of(env))
        return int(np.argmax(q[:cands]))

    def train(self, env_fn, episodes, seed=0, log_every=0):
        rng = np.random.default_rng(seed)
        for ep in range(episodes):
            frac = ep / max(1, episodes - 1)
            eps = self.eps_start + frac * (self.eps_end - self.eps_start)
            env = env_fn(seed=seed + ep)
            obs, _ = env.reset()
            done = False
            while not done:
                st = self._state_of(env)
                a = self.act(env, eps, rng)
                obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                q = self._q(st)
                if done:
                    target = r
                else:
                    st2 = self._state_of(env)
                    target = r + self.gamma * float(np.max(self._q(st2)))
                q[a] += self.alpha * (target - q[a])
            if log_every and (ep + 1) % log_every == 0:
                print(f"    [qlearn] episode {ep + 1}/{episodes}, eps={eps:.3f}",
                      flush=True)
        return self

    def evaluate(self, env_fn, episodes, base_seed):
        rng = np.random.default_rng(base_seed)
        rows = []
        for ep in range(episodes):
            env = env_fn(seed=base_seed + ep)
            env.reset()
            done = False
            while not done:
                a = self.act(env, 0.0, rng)  # greedy
                obs, _, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
            m = env.compute_metrics()
            m["episode"] = ep
            rows.append(m)
        return rows
