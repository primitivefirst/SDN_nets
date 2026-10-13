"""Experiment E8: learning convergence and stability (reviewer hygiene).

Protocol: train DQN and QL on each topology (same training setup as
E1-E4: mixed loads + random link failures) with periodic evaluation on
a FIXED eval set (3 episodes, seeds 0-2, same distribution) under a
greedy/deterministic policy. Records mean episode reward and loss
ratio vs training progress, 5 training seeds per agent/topology.

Outputs (in --outdir, default ./results_e8):
  config.json, results_e8.csv, results_e8_summary.csv,
  e8_dqn_nsfnet.png, e8_dqn_fattree.png,
  e8_ql_nsfnet.png, e8_ql_fattree.png

Example (full):  python run_e8.py
Example (fast):  python run_e8.py --fast
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
EVAL_SEEDS = [0, 1, 2]


def eval_policy(topo, args, policy_fn, tag=""):
    """Mean episode reward + loss on the fixed eval set."""
    G, hosts = get_topology(topo)
    rews, losses = [], []
    for es in EVAL_SEEDS:
        env = RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[topo],
                         ep_len=args.ep_len, warmup=args.warmup,
                         fail_mode="train_random",
                         fail_duration=args.fail_duration, seed=es)
        obs, _ = env.reset()
        done, tot = False, 0.0
        while not done:
            a = policy_fn(env, obs)
            obs, r, term, trunc, _ = env.step(int(a))
            tot += r
            done = term or trunc
        rews.append(tot)
        losses.append(env.compute_metrics()["loss_ratio"])
    return float(np.mean(rews)), float(np.mean(losses))


def train_dqn_curve(topo, args, seed, rows):
    from stable_baselines3 import DQN
    G, hosts = get_topology(topo)
    env = RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[topo],
                     ep_len=args.ep_len, warmup=args.warmup,
                     fail_mode="train_random",
                     fail_duration=args.fail_duration, seed=seed)
    model = DQN("MlpPolicy", env, verbose=0, seed=seed,
                policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=2000, buffer_size=30000)
    pol = lambda env, obs: int(model.predict(obs, deterministic=True)[0])
    r, l = eval_policy(topo, args, pol)
    rows.append({"topology": topo, "agent": "dqn", "seed": seed, "x": 0,
                 "reward": r, "loss": l})
    done = 0
    while done < args.dqn_steps:
        model.learn(total_timesteps=args.dqn_eval_every)
        done += args.dqn_eval_every
        r, l = eval_policy(topo, args, pol)
        rows.append({"topology": topo, "agent": "dqn", "seed": seed,
                     "x": done, "reward": r, "loss": l})
    print(f"    dqn/{topo}/seed{seed}: final reward {r:.1f}, loss {l:.3f}",
          flush=True)


def train_ql_curve(topo, args, seed, rows):
    G, hosts = get_topology(topo)

    def _fn(s):
        return RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[topo],
                          ep_len=args.ep_len, warmup=args.warmup,
                          fail_mode="train_random",
                          fail_duration=args.fail_duration, seed=s)
    agent = QLearningAgent(k_paths=3)
    rng = np.random.default_rng(9000 + seed)
    pol = lambda env, obs: agent.act(env, 0.0, rng)
    r, l = eval_policy(topo, args, pol)
    rows.append({"topology": topo, "agent": "qlearn", "seed": seed, "x": 0,
                 "reward": r, "loss": l})
    # manual episode loop with periodic eval (mirrors QLearningAgent.train)
    for ep in range(args.qlearn_episodes):
        frac = ep / max(1, args.qlearn_episodes - 1)
        eps = agent.eps_start + frac * (agent.eps_end - agent.eps_start)
        env = _fn(1000 + seed + ep)
        env.reset()
        done = False
        while not done:
            st = agent._state_of(env)
            a = agent.act(env, eps, rng)
            obs, rew, term, trunc, _ = env.step(a)
            done = term or trunc
            q = agent._q(st)
            if done:
                target = rew
            else:
                st2 = agent._state_of(env)
                target = rew + agent.gamma * float(np.max(agent._q(st2)))
            q[a] += agent.alpha * (target - q[a])
        if (ep + 1) % args.ql_eval_every == 0:
            r, l = eval_policy(topo, args, pol)
            rows.append({"topology": topo, "agent": "qlearn", "seed": seed,
                         "x": ep + 1, "reward": r, "loss": l})
    print(f"    qlearn/{topo}/seed{seed}: final reward {r:.1f}, loss {l:.3f}",
          flush=True)


def run(args):
    topos = ["nsfnet", "fattree"]
    seeds = list(range(args.seeds))
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    t0 = time.time()
    rows = []
    for topo in topos:
        print(f"\n===== Topology {topo} =====", flush=True)
        for seed in seeds:
            print(f"  seed {seed}...", flush=True)
            train_dqn_curve(topo, args, seed, rows)
            train_ql_curve(topo, args, seed, rows)
    dt = time.time() - t0
    print(f"\nTotal wall time: {dt / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "results_e8.csv"), index=False)
    g = df.groupby(["topology", "agent", "x"]).agg(
        reward_mean=("reward", "mean"), reward_std=("reward", "std"),
        loss_mean=("loss", "mean"), loss_std=("loss", "std"),
        n=("reward", "size")).reset_index()
    g.to_csv(os.path.join(args.outdir, "results_e8_summary.csv"), index=False)
    print(f"Saved results ({len(df)} rows)", flush=True)

    for topo in topos:
        for agent, xlabel in [("dqn", "Timesteps"), ("qlearn", "Episodes")]:
            sub = df[(df["topology"] == topo) & (df["agent"] == agent)]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, col, yl in zip(axes, ["reward", "loss"],
                                   ["Eval episode reward", "Eval loss ratio"]):
                for seed in seeds:
                    s = sub[sub["seed"] == seed].sort_values("x")
                    ax.plot(s["x"], s[col], alpha=0.35, label=f"seed {seed}")
                gs = g[(g["topology"] == topo) & (g["agent"] == agent)]
                gs = gs.sort_values("x")
                ax.plot(gs["x"], gs[f"{col}_mean"], "k-", lw=2, label="mean")
                ax.fill_between(gs["x"], gs[f"{col}_mean"] - gs[f"{col}_std"],
                                gs[f"{col}_mean"] + gs[f"{col}_std"],
                                color="k", alpha=0.12)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(yl)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            fig.suptitle(f"E8 {agent} on {topo}: convergence (5 seeds)")
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, f"e8_{agent}_{topo}.png"),
                        dpi=150)
            plt.close(fig)
    print("Saved plots: e8_dqn_nsfnet.png, e8_dqn_fattree.png, "
          "e8_ql_nsfnet.png, e8_ql_fattree.png", flush=True)
    print("DONE.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="E8: learning curves")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ep-len", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--fail-duration", type=int, default=100)
    ap.add_argument("--qlearn-episodes", type=int, default=1500)
    ap.add_argument("--ql-eval-every", type=int, default=150)
    ap.add_argument("--dqn-steps", type=int, default=50000)
    ap.add_argument("--dqn-eval-every", type=int, default=2500)
    ap.add_argument("--outdir", default="./results_e8")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        args.seeds = 1
        args.qlearn_episodes = 300
        args.ql_eval_every = 100
        args.dqn_steps = 4000
        args.dqn_eval_every = 1000
        args.outdir = "./results_e8_fast"
    run(args)


if __name__ == "__main__":
    main()
