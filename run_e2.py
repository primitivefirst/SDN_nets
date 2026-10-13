"""Experiment E2: single link failure (fail + recover within episode).

Protocol: episode of --ep-len steps; one critical link fails at
--fail-start for --fail-duration steps, then recovers.
  * Data plane (instant): affected active flows are rerouted onto backup
    candidates (method-specific policy) or dropped if none exists.
  * Control plane (delayed by react_delay=20): candidate paths are
    recomputed without the failed link; arrivals sent into it before
    that are dropped. RL agents see the failure instantly (mask).
Failed links = 5 most utilized ('critical') links measured with SP.
RL baselines are trained WITH random failures (failure-aware training).

Methods: sp, ecmp, qlearn (tabular), dqn (Stable-Baselines3).
Load: single representative high load (nsfnet 80 / fattree 120).

Outputs (in --outdir, default ./results_e2):
  config.json, results_e2.csv, results_e2_summary.csv,
  e2_loss_time.png, e2_thr_time.png, e2_recovery.png, e2_window_loss.png,
  e2_disturbance.png

Example (full):  python run_e2.py
Example (fast):  python run_e2.py --fast
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
REROUTE = {"sp": "first", "ecmp": "random", "qlearn": "ql", "dqn": "dqn"}


def pick_critical_links(G, hosts, args, n_links=5):
    """Measure mean link utilization under SP; return top-n link indices."""
    L = len(list(G.edges()))
    acc = np.zeros(L)
    cnt = 0
    for ep in range(5):
        env = RoutingEnv(G, hosts, hold_mean=args.load,
                         ep_len=args.ep_len, warmup=args.warmup,
                         seed=9000 + ep)
        env.reset()
        done = False
        while not done:
            _o, _r, term, trunc, _ = env.step(0)
            done = term or trunc
            if env.t >= args.warmup:
                acc += env.loads / env.caps
                cnt += 1
    mean_u = acc / max(1, cnt)
    # Only infrastructure links (exclude host links: degree-1 endpoints).
    edge_list = list(G.edges())
    infra = [i for i, (a, b) in enumerate(edge_list)
             if G.degree(a) > 1 and G.degree(b) > 1] or list(range(L))
    ranked = sorted(infra, key=lambda i: mean_u[i])
    top = sorted(ranked[-min(n_links, len(ranked)):])
    print(f"  mean link utilization: {np.round(mean_u, 2).tolist()}",
          flush=True)
    return top


def make_train_env_fn(G, hosts, args):
    def _fn(seed):
        return RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[args.topology],
                          ep_len=args.ep_len, warmup=args.warmup,
                          fail_mode="train_random",
                          fail_duration=args.fail_duration, seed=seed)
    return _fn


def make_eval_env_fn(G, hosts, args, link, method, agent=None):
    def _fn(seed):
        env = RoutingEnv(G, hosts, hold_mean=args.load,
                         ep_len=args.ep_len, warmup=args.warmup,
                         fail_duration=args.fail_duration, seed=seed,
                         reroute_policy=REROUTE[method],
                         reroute_agent=agent)
        env.set_failure(link, args.fail_start, args.fail_duration)
        return env
    return _fn


def train_dqn(G, hosts, args, seed):
    from stable_baselines3 import DQN
    env = RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[args.topology],
                     ep_len=args.ep_len, warmup=args.warmup,
                     fail_mode="train_random",
                     fail_duration=args.fail_duration, seed=seed)
    model = DQN("MlpPolicy", env, verbose=0, seed=seed,
                policy_kwargs=dict(net_arch=[64, 64]),
                learning_starts=2000, buffer_size=30000)
    model.learn(total_timesteps=args.dqn_steps)
    return model


def run_episode_collect(env_fn, seed, policy_fn):
    """Run one episode with policy_fn(env)->action; return metrics+series."""
    env = env_fn(seed=seed)
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy_fn(env, obs)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    m = env.compute_metrics()
    m.update(env.failure_metrics())
    return m, env.get_series()


def run(args):
    G, hosts = get_topology(args.topology)
    print(f"Topology: {args.topology} "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} links)",
          flush=True)
    methods = [m for m in METHODS if m in args.methods]
    seeds = list(range(args.seeds))
    print("Measuring critical links (quick SP probe)...", flush=True)
    links = pick_critical_links(G, hosts, args, args.links)
    edge_list = list(G.edges())  # same order as env (do NOT sort!)
    print(f"Load: {args.load} | Failure: link fails at t={args.fail_start} "
          f"for {args.fail_duration} steps (react delay 20)", flush=True)
    print(f"Critical-link set: {links} "
          f"(edges: {[edge_list[i] for i in links]})", flush=True)
    print(f"Methods: {methods} | Seeds: {seeds} | "
          f"Episodes per (method, link): {args.eval_episodes}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        cfg = vars(args)
        cfg["failed_links"] = links
        json.dump(cfg, f, indent=2)

    t0 = time.time()
    all_rows, all_series = [], []
    for si, seed in enumerate(seeds):
        print(f"\n=== Seed {seed} ({si + 1}/{len(seeds)}) ===", flush=True)
        agents = {}
        if "qlearn" in methods:
            print("  training qlearn (with failures)...", flush=True)
            agents["qlearn"] = QLearningAgent(k_paths=3).train(
                make_train_env_fn(G, hosts, args), args.qlearn_episodes,
                seed=1000 + seed, log_every=300)
        if "dqn" in methods:
            print("  training dqn (with failures)...", flush=True)
            agents["dqn"] = train_dqn(G, hosts, args, seed)
        rng = np.random.default_rng(7000 + seed)
        for link in links:
            for method in methods:
                agent = agents.get(method)
                env_fn = make_eval_env_fn(G, hosts, args, link, method,
                                          agent)
                base = 5000 + seed * 100000 + link * 1000
                if method == "sp":
                    pol = lambda env, obs: 0
                elif method == "ecmp":
                    def pol(env, obs, _rng=rng):
                        s, t, _d = env.cur_flow
                        cands = env.paths[(s, t)]
                        if not cands:
                            return 0
                        mlen = env.min_len[(s, t)]
                        valid = [i for i, p in enumerate(cands)
                                 if len(p) == mlen]
                        return int(_rng.choice(valid))
                elif method == "qlearn":
                    pol = lambda env, obs: agent.act(env, 0.0, rng)
                else:
                    pol = lambda env, obs: int(
                        agent.predict(obs, deterministic=True)[0])
                for ep in range(args.eval_episodes):
                    m, series = run_episode_collect(env_fn, base + ep, pol)
                    m.update({"method": method, "seed": seed, "link": link,
                              "episode": ep})
                    all_rows.append(m)
                    all_series.append((method, series))
            print(f"  link={link}: done", flush=True)
    dt = time.time() - t0
    print(f"\nTotal wall time: {dt / 60:.1f} min", flush=True)

    df = pd.DataFrame(all_rows)
    raw_path = os.path.join(args.outdir, "results_e2.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path} ({len(df)} rows)", flush=True)

    g = df.groupby("method").agg(
        throughput_mean=("throughput", "mean"), throughput_std=("throughput", "std"),
        delay_mean=("avg_delay", "mean"), delay_std=("avg_delay", "std"),
        loss_mean=("loss_ratio", "mean"), loss_std=("loss_ratio", "std"),
        base_loss_mean=("base_loss", "mean"),
        window_loss_mean=("window_loss", "mean"),
        window_loss_std=("window_loss", "std"),
        recovery_mean=("recovery_time", "mean"),
        recovery_std=("recovery_time", "std"),
        affected_mean=("affected_n", "mean"),
        rerouted_mean=("rerouted_n", "mean"),
        rerouted_std=("rerouted_n", "std"), n=("throughput", "size")
    ).reset_index()
    sum_path = os.path.join(args.outdir, "results_e2_summary.csv")
    g.to_csv(sum_path, index=False)
    print(f"Saved {sum_path}", flush=True)
    print("\n=== E2 SUMMARY (mean +/- std) ===", flush=True)
    print(g.to_string(index=False), flush=True)

    t_axis = all_series[0][1]["t"]
    for key, ylabel, fname in [("loss", "Loss ratio", "e2_loss_time.png"),
                               ("throughput", "Throughput (delivered/step)",
                                "e2_thr_time.png")]:
        plt.figure()
        for method in methods:
            arr = np.stack([s[key] for (m, s) in all_series if m == method])
            mu, sd = arr.mean(axis=0), arr.std(axis=0)
            plt.plot(t_axis, mu, label=method)
            plt.fill_between(t_axis, mu - sd, mu + sd, alpha=0.15)
        plt.axvspan(args.fail_start, args.fail_start + args.fail_duration,
                    color="red", alpha=0.12, label="link failure")
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(f"E2 ({args.topology}, load {args.load}): {key} over time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=150)
        plt.close()
    for col, sd, ylabel, fname in [
            ("recovery_mean", "recovery_std", "Recovery time (steps)",
             "e2_recovery.png"),
            ("window_loss_mean", "window_loss_std", "Loss during failure",
             "e2_window_loss.png"),
            ("rerouted_mean", "rerouted_std", "Rerouted flows (disturbance)",
             "e2_disturbance.png")]:
        plt.figure()
        order = [m for m in methods if m in set(g["method"])]
        vals = [float(g[g["method"] == m][col].iloc[0]) for m in order]
        errs = [float(g[g["method"] == m][sd].iloc[0]) for m in order]
        plt.bar(order, vals, yerr=errs, capsize=4)
        plt.ylabel(ylabel)
        plt.title(f"E2 ({args.topology}, load {args.load})")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=150)
        plt.close()
    print("Saved plots: e2_loss_time.png, e2_thr_time.png, e2_recovery.png, "
          "e2_window_loss.png, e2_disturbance.png", flush=True)
    print("DONE.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="E2: single link failure")
    ap.add_argument("--topology", default="nsfnet", choices=["nsfnet", "fattree"])
    ap.add_argument("--load", type=int, default=0,
                    help="0 = default per topology (nsfnet 80 / fattree 120)")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--links", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--ep-len", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--fail-start", type=int, default=250)
    ap.add_argument("--fail-duration", type=int, default=100)
    ap.add_argument("--qlearn-episodes", type=int, default=1500)
    ap.add_argument("--dqn-steps", type=int, default=50000)
    ap.add_argument("--outdir", default="./results_e2")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.load == 0:
        args.load = LOAD[args.topology]
    if args.fast:
        args.seeds = 1
        args.links = 2
        args.eval_episodes = 2
        args.ep_len = 120
        args.warmup = 50
        args.fail_start = 70
        args.fail_duration = 25
        args.qlearn_episodes = 100
        args.dqn_steps = 2000
        args.outdir = f"./results_e2_fast_{args.topology}"
    run(args)


if __name__ == "__main__":
    main()
