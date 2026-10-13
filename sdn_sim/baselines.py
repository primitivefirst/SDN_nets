"""Classical baselines: Shortest Path (SP) and ECMP."""
import numpy as np


def run_episode(env, method, rng):
    """Route one episode with a classical baseline; return metrics dict."""
    obs, _ = env.reset()
    done = False
    while not done:
        s0, t0, _d0 = env.cur_flow
        if not env.paths[(s0, t0)]:
            action = 0  # disconnected: env drops the flow
        elif method == "sp":
            action = 0  # first candidate = min-hop shortest path
        elif method == "ecmp":
            # True ECMP: uniform choice among EQUAL-cost (min-hop) paths only.
            s, t, _d = env.cur_flow
            cands = env.paths[(s, t)]
            mlen = env.min_len[(s, t)]
            valid = [i for i, p in enumerate(cands) if len(p) == mlen]
            action = int(rng.choice(valid))
        else:
            raise ValueError(f"Unknown baseline: {method}")
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.compute_metrics()


def evaluate(env_fn, method, episodes, base_seed):
    rows = []
    for ep in range(episodes):
        env = env_fn(seed=base_seed + ep)
        rng = np.random.default_rng(base_seed + ep)
        m = run_episode(env, method, rng)
        m["episode"] = ep
        rows.append(m)
    return rows
