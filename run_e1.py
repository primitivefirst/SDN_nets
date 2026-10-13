"""Experiment E1: baseline comparison under normal load (no failures).

Dynamic online model: flows arrive each step, hold for a geometrically
distributed time, then depart. Load level = mean holding time.
Methods: sp, ecmp, qlearn (tabular), dqn (Stable-Baselines3).
Topology: nsfnet (default) or fattree.

Outputs (in --outdir, default ./results_e1):
  config.json, results_e1.csv, results_e1_summary.csv,
  e1_throughput.png, e1_delay.png, e1_loss.png

Example (full):  python run_e1.py
Example (fast):  python run_e1.py --fast
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
from sdn_sim.baselines import evaluate as eval_classical
from sdn_sim.qlearning import QLearningAgent

LOADS = {"nsfnet": [20, 40, 60, 80], "fattree": [30, 60, 90, 120]}
METHODS = ["sp", "ecmp", "qlearn", "dqn"]


def make_env_fn(G, hosts, load, k_paths=3, ep_len=300, warmup=50):
    def _fn(seed):
        return RoutingEnv(G, hosts, k_paths=k_paths, hold_mean=load,
                          ep_len=ep_len, warmup=warmup, seed=seed)
    return _fn


def train_dqn(G, hosts, train_load, timesteps, seed):
    from stable_baselines3 import DQN
    env = RoutingEnv(G, hosts, hold_mean=train_load, seed=seed)
    model = DQN("MlpPolicy", env, verbose=0, seed=seed,
                policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=2000, buffer_size=30000)
    model.learn(total_timesteps=timesteps)
    return model


def eval_dqn(model, env_fn, episodes, base_seed):
    rows = []
    for ep in range(episodes):
        env = env_fn(seed=base_seed + ep)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        m = env.compute_metrics()
        m["episode"] = ep
        rows.append(m)
    return rows


def run(args):
    G, hosts = get_topology(args.topology)
    print(f"Topology: {args.topology} "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} links, "
          f"{len(hosts)} endpoints)", flush=True)
    loads = [int(x) for x in args.loads.split(",")] if args.loads else LOADS[args.topology]
    methods = [m for m in METHODS if m in args.methods]
    seeds = list(range(args.seeds))
    print(f"Loads (mean holding): {loads} | Methods: {methods} | "
          f"Seeds: {seeds} | Eval episodes: {args.eval_episodes} "
          f"({args.ep_len} steps, warmup {args.warmup})", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    train_load = loads  # mixed-load training: random load per episode
    print(f"RL training loads (mixed): {train_load} "
          f"(qlearn: {args.qlearn_episodes} eps, dqn: {args.dqn_steps} steps)",
          flush=True)

    t0 = time.time()
    all_rows = []
    for si, seed in enumerate(seeds):
        print(f"\n=== Seed {seed} ({si + 1}/{len(seeds)}) ===", flush=True)
        q_agent = None
        dqn_model = None
        if "qlearn" in methods:
            print("  training qlearn...", flush=True)
            q_agent = QLearningAgent(k_paths=3).train(
                make_env_fn(G, hosts, train_load, ep_len=args.ep_len,
                            warmup=args.warmup),
                args.qlearn_episodes, seed=1000 + seed, log_every=0)
        if "dqn" in methods:
            print("  training dqn...", flush=True)
            dqn_model = train_dqn(G, hosts, train_load, args.dqn_steps, seed)
        for load in loads:
            env_fn = make_env_fn(G, hosts, load, ep_len=args.ep_len,
                                 warmup=args.warmup)
            base = 5000 + seed * 100000 + load * 1000
            if "sp" in methods:
                for r in eval_classical(env_fn, "sp", args.eval_episodes, base):
                    all_rows.append(("sp", load, seed, r))
            if "ecmp" in methods:
                for r in eval_classical(env_fn, "ecmp", args.eval_episodes, base):
                    all_rows.append(("ecmp", load, seed, r))
            if q_agent is not None:
                for r in q_agent.evaluate(env_fn, args.eval_episodes, base):
                    all_rows.append(("qlearn", load, seed, r))
            if dqn_model is not None:
                for r in eval_dqn(dqn_model, env_fn, args.eval_episodes, base):
                    all_rows.append(("dqn", load, seed, r))
            print(f"  load={load}: done", flush=True)
    dt = time.time() - t0
    print(f"\nTotal wall time: {dt / 60:.1f} min", flush=True)

    recs = []
    for (method, load, seed, r) in all_rows:
        recs.append({"method": method, "load": load, "seed": seed,
                     "episode": r["episode"], "throughput": r["throughput"],
                     "avg_delay": r["avg_delay"], "loss_ratio": r["loss_ratio"]})
    df = pd.DataFrame(recs)
    raw_path = os.path.join(args.outdir, "results_e1.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path} ({len(df)} rows)", flush=True)

    g = df.groupby(["method", "load"]).agg(
        throughput_mean=("throughput", "mean"), throughput_std=("throughput", "std"),
        delay_mean=("avg_delay", "mean"), delay_std=("avg_delay", "std"),
        loss_mean=("loss_ratio", "mean"), loss_std=("loss_ratio", "std"),
        n=("throughput", "size")).reset_index()
    sum_path = os.path.join(args.outdir, "results_e1_summary.csv")
    g.to_csv(sum_path, index=False)
    print(f"Saved {sum_path}", flush=True)
    print("\n=== E1 SUMMARY (mean +/- std) ===", flush=True)
    print(g.to_string(index=False), flush=True)

    for metric, col_m, col_s, ylabel, fname in [
            ("throughput", "throughput_mean", "throughput_std",
             "Throughput (delivered units/step)", "e1_throughput.png"),
            ("delay", "delay_mean", "delay_std",
             "Avg flow delay", "e1_delay.png"),
            ("loss", "loss_mean", "loss_std",
             "Loss ratio", "e1_loss.png")]:
        plt.figure()
        for method in methods:
            sub = g[g["method"] == method].sort_values("load")
            plt.errorbar(sub["load"], sub[col_m], yerr=sub[col_s],
                         marker="o", capsize=3, label=method)
        plt.xlabel("Load (mean holding time, steps)")
        plt.ylabel(ylabel)
        plt.title(f"E1 ({args.topology}): {metric} vs load")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=150)
        plt.close()
    print("Saved plots: e1_throughput.png, e1_delay.png, e1_loss.png", flush=True)
    print("DONE.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="E1: baselines vs load (dynamic)")
    ap.add_argument("--topology", default="nsfnet", choices=["nsfnet", "fattree"])
    ap.add_argument("--loads", default="",
                    help="e.g. '20,40,60,80' (default per topology)")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="subset of sp,ecmp,qlearn,dqn")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--ep-len", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--qlearn-episodes", type=int, default=1500)
    ap.add_argument("--dqn-steps", type=int, default=50000)
    ap.add_argument("--outdir", default="./results_e1")
    ap.add_argument("--fast", action="store_true",
                    help="smoke test: 1 seed, 2 loads, tiny training")
    args = ap.parse_args()
    if args.fast:
        args.seeds = 1
        args.loads = "20,60" if args.topology == "nsfnet" else "30,90"
        args.eval_episodes = 3
        args.ep_len = 120
        args.warmup = 20
        args.qlearn_episodes = 100
        args.dqn_steps = 2000
        args.outdir = f"./results_e1_fast_{args.topology}"
    run(args)


def _selftest():
    """Quick import + tiny rollout sanity check."""
    G, hosts = get_topology("nsfnet")
    env = RoutingEnv(G, hosts, hold_mean=20, ep_len=60, warmup=10, seed=1)
    obs, _ = env.reset()
    assert obs.shape == (3 + env.L + env.k + env.L,)
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(0)
        done = term or trunc
    m = env.compute_metrics()
    assert set(m) == {"throughput", "avg_delay", "loss_ratio"}
    print("selftest OK:", m)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
