"""Experiment E7: control-plane cost (real-time suitability).

Protocol: for each topology, train QL/DQN once with modest budgets
(inference latency does not depend on training length), then measure:
  * decision latency: wall time per routing decision (policy call only),
    N decisions x REPS repetitions -> ms/decision, decisions/s;
  * episode wall time: full 400-step episode (env + policy), 3 reps;
  * DQN policy size (parameter count).
SP/ECMP need no training. No failures, default loads. Everything runs
on a regular CPU; report the machine in the paper.

Outputs (in --outdir, default ./results_e7):
  config.json, results_e7.csv, results_e7_summary.csv,
  e7_latency.png, e7_episode_wall.png

Example:  python run_e7.py
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sdn_sim.topologies import get_topology
from sdn_sim.env import RoutingEnv
from sdn_sim.qlearning import QLearningAgent

LOAD = {"nsfnet": 80, "fattree": 120}
TRAIN_LOADS = {"nsfnet": [20, 40, 60, 80], "fattree": [30, 60, 90, 120]}
METHODS = ["sp", "ecmp", "qlearn", "dqn"]


def train_agents(topo, args):
    G, hosts = get_topology(topo)

    def qenv(seed):
        return RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[topo],
                          ep_len=args.ep_len, warmup=args.warmup,
                          fail_mode="train_random",
                          fail_duration=args.fail_duration, seed=seed)
    print(f"  training qlearn ({args.qlearn_episodes} eps)...", flush=True)
    ql = QLearningAgent(k_paths=3).train(
        qenv, args.qlearn_episodes, seed=1000, log_every=0)
    from stable_baselines3 import DQN
    print(f"  training dqn ({args.dqn_steps} steps)...", flush=True)
    denv = RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[topo],
                      ep_len=args.ep_len, warmup=args.warmup,
                      fail_mode="train_random",
                      fail_duration=args.fail_duration, seed=0)
    dqn = DQN("MlpPolicy", denv, verbose=0, seed=0,
              policy_kwargs=dict(net_arch=[64, 64]),
              learning_starts=2000, buffer_size=30000)
    dqn.learn(total_timesteps=args.dqn_steps)
    n_params = sum(int(p.numel()) for p in dqn.policy.parameters())
    print(f"  dqn params: {n_params}", flush=True)
    return {"qlearn": ql, "dqn": dqn, "dqn_params": n_params}


def make_pol(method, agents):
    if method == "sp":
        return lambda env, obs: 0
    if method == "ecmp":
        rng = np.random.default_rng(0)
        def pol(env, obs):
            s, t, _d = env.cur_flow
            cands = env.paths[(s, t)]
            if not cands:
                return 0
            mlen = env.min_len[(s, t)]
            valid = [i for i, p in enumerate(cands) if len(p) == mlen]
            return int(rng.choice(valid))
        return pol
    if method == "qlearn":
        agent = agents["qlearn"]
        rng = np.random.default_rng(1)
        return lambda env, obs: agent.act(env, 0.0, rng)
    agent = agents["dqn"]
    return lambda env, obs: int(agent.predict(obs, deterministic=True)[0])


def measure_latency(topo, method, agents, args):
    """Pure policy-call time over N decisions (env.step excluded)."""
    G, hosts = get_topology(topo)
    pol = make_pol(method, agents)
    ms = []
    for rep in range(args.reps):
        env = RoutingEnv(G, hosts, hold_mean=LOAD[topo],
                         ep_len=args.ep_len, warmup=0, seed=rep)
        obs, _ = env.reset()
        t_pol = 0.0
        for _ in range(args.decisions):
            t0 = time.perf_counter()
            a = pol(env, obs)
            t_pol += time.perf_counter() - t0
            obs, _, term, trunc, _ = env.step(int(a))
            if term or trunc:
                obs, _ = env.reset()
        ms.append(t_pol / args.decisions * 1000.0)
    return ms


def measure_episode_wall(topo, method, agents, args):
    """Full-episode wall time (env + policy)."""
    G, hosts = get_topology(topo)
    pol = make_pol(method, agents)
    secs = []
    for rep in range(3):
        env = RoutingEnv(G, hosts, hold_mean=LOAD[topo],
                         ep_len=args.ep_len, warmup=args.warmup, seed=rep)
        obs, _ = env.reset()
        t0 = time.perf_counter()
        done = False
        while not done:
            a = pol(env, obs)
            obs, _, term, trunc, _ = env.step(int(a))
            done = term or trunc
        secs.append(time.perf_counter() - t0)
    return secs


def run(args):
    topos = ["nsfnet", "fattree"]
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    rows = []
    for topo in topos:
        print(f"\n===== Topology {topo} =====", flush=True)
        agents = train_agents(topo, args)
        for method in METHODS:
            ms = measure_latency(topo, method, agents, args)
            ep = measure_episode_wall(topo, method, agents, args)
            for v in ms:
                rows.append({"topology": topo, "method": method,
                             "metric": "ms_per_decision", "value": v})
            for v in ep:
                rows.append({"topology": topo, "method": method,
                             "metric": "episode_wall_s", "value": v})
            rows.append({"topology": topo, "method": method,
                         "metric": "dqn_params",
                         "value": agents["dqn_params"] if method == "dqn"
                         else 0})
            print(f"  {method}: {np.mean(ms):.4f} ms/decision, "
                  f"episode {np.mean(ep):.2f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "results_e7.csv"), index=False)
    g = df[df["metric"].isin(["ms_per_decision", "episode_wall_s"])].groupby(
        ["topology", "method", "metric"])["value"].agg(
        ["mean", "std", "size"]).reset_index()
    g.to_csv(os.path.join(args.outdir, "results_e7_summary.csv"), index=False)
    print("\n=== E7 SUMMARY ===", flush=True)
    print(g.to_string(index=False), flush=True)

    for metric, ylabel, fname, log in [
            ("ms_per_decision", "Decision latency (ms)", "e7_latency.png", True),
            ("episode_wall_s", "Episode wall time (s)", "e7_episode_wall.png", False)]:
        sub = g[g["metric"] == metric]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, topo in zip(axes, topos):
            s = sub[sub["topology"] == topo].set_index("method").loc[METHODS]
            ax.bar(METHODS, s["mean"], yerr=s["std"], capsize=4)
            ax.set_title(topo)
            ax.set_ylabel(ylabel)
            if log:
                ax.set_yscale("log")
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"E7 {ylabel.lower()} per method (CPU)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, fname), dpi=150)
        plt.close(fig)
    print("Saved plots: e7_latency.png, e7_episode_wall.png", flush=True)
    print("DONE.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="E7: control-plane cost")
    ap.add_argument("--decisions", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ep-len", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--fail-duration", type=int, default=100)
    ap.add_argument("--qlearn-episodes", type=int, default=500)
    ap.add_argument("--dqn-steps", type=int, default=10000)
    ap.add_argument("--outdir", default="./results_e7")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        args.decisions = 300
        args.reps = 2
        args.qlearn_episodes = 50
        args.dqn_steps = 1000
        args.outdir = "./results_e7_fast"
    run(args)


if __name__ == "__main__":
    main()
