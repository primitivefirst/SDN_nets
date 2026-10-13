"""Experiment E3: multiple simultaneous link failures (severity sweep).

Protocol: like E2, but K links fail at once, K in --levels (default
1,2,3,4). Failed sets are nested from the critical-link ordering
(K=1: {l1}, K=2: {l1,l2}, ...), so severity grows monotonically.
Protection reroute + delayed recompute (react delay 20) + recovery,
same as E2. RL baselines trained WITH random 1-3 link failures.

Methods: sp, ecmp, qlearn (tabular), dqn (Stable-Baselines3).
Load: single representative high load (nsfnet 80 / fattree 120).

Outputs (in --outdir, default ./results_e3):
  config.json, results_e3.csv, results_e3_summary.csv,
  e3_loss_vs_k.png, e3_thr_vs_k.png, e3_disturbance_vs_k.png,
  e3_recovery_vs_k.png, e3_loss_time_kmax.png

Example (full):  python run_e3.py
Example (fast):  python run_e3.py --fast
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


def pick_critical_links(G, hosts, args, n_links=4):
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
    edge_list = list(G.edges())
    infra = [i for i, (a, b) in enumerate(edge_list)
             if G.degree(a) > 1 and G.degree(b) > 1] or list(range(L))
    ranked = sorted(infra, key=lambda i: mean_u[i], reverse=True)
    top = ranked[:min(n_links, len(ranked))]
    print(f"  critical ordering: {top} "
          f"(edges: {[edge_list[i] for i in top]})", flush=True)
    return top


def make_train_env_fn(G, hosts, args):
    def _fn(seed):
        return RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[args.topology],
                          ep_len=args.ep_len, warmup=args.warmup,
                          fail_mode="train_random",
                          fail_duration=args.fail_duration, seed=seed)
    return _fn


def make_eval_env_fn(G, hosts, args, links, method, agent=None):
    def _fn(seed):
        env = RoutingEnv(G, hosts, hold_mean=args.load,
                         ep_len=args.ep_len, warmup=args.warmup,
                         fail_duration=args.fail_duration, seed=seed,
                         reroute_policy=REROUTE[method],
                         reroute_agent=agent)
        env.set_failures(links, args.fail_start, args.fail_duration)
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
    levels = [int(x) for x in args.levels.split(",")]
    print("Measuring critical links (quick SP probe)...", flush=True)
    crit = pick_critical_links(G, hosts, args, max(levels))
    print(f"Load: {args.load} | Failure: top-K links fail at "
          f"t={args.fail_start} for {args.fail_duration} steps "
          f"(react delay 20)", flush=True)
    print(f"Severity levels K: {levels} | Methods: {methods} | "
          f"Seeds: {seeds} | Episodes per (method, K): "
          f"{args.eval_episodes}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        cfg = vars(args)
        cfg["critical_order"] = crit
        json.dump(cfg, f, indent=2)

    t0 = time.time()
    all_rows, kmax_series = [], []
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
        for K in levels:
            flinks = crit[:K]
            for method in methods:
                agent = agents.get(method)
                env_fn = make_eval_env_fn(G, hosts, args, flinks, method,
                                          agent)
                base = 5000 + seed * 100000 + K * 1000
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
                    m.update({"method": method, "seed": seed, "K": K,
                              "episode": ep})
                    all_rows.append(m)
                    if K == max(levels):
                        kmax_series.append((method, series))
            print(f"  K={K}: done", flush=True)
    dt = time.time() - t0
    print(f"\nTotal wall time: {dt / 60:.1f} min", flush=True)

    df = pd.DataFrame(all_rows)
    raw_path = os.path.join(args.outdir, "results_e3.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path} ({len(df)} rows)", flush=True)

    g = df.groupby(["method", "K"]).agg(
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
    sum_path = os.path.join(args.outdir, "results_e3_summary.csv")
    g.to_csv(sum_path, index=False)
    print(f"Saved {sum_path}", flush=True)
    print("\n=== E3 SUMMARY (mean +/- std) ===", flush=True)
    print(g.to_string(index=False), flush=True)

    for col_m, col_s, ylabel, fname in [
            ("window_loss_mean", "window_loss_std", "Loss during failure",
             "e3_loss_vs_k.png"),
            ("throughput_mean", "throughput_std",
             "Throughput (delivered/step)", "e3_thr_vs_k.png"),
            ("rerouted_mean", "rerouted_std", "Rerouted flows (disturbance)",
             "e3_disturbance_vs_k.png"),
            ("recovery_mean", "recovery_std", "Recovery time (steps)",
             "e3_recovery_vs_k.png")]:
        plt.figure()
        for method in methods:
            sub = g[g["method"] == method].sort_values("K")
            plt.errorbar(sub["K"], sub[col_m], yerr=sub[col_s],
                         marker="o", capsize=3, label=method)
        plt.xlabel("Failed links K")
        plt.ylabel(ylabel)
        plt.title(f"E3 ({args.topology}, load {args.load})")
        plt.xticks(levels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=150)
        plt.close()
    # Time course at max severity.
    t_axis = kmax_series[0][1]["t"]
    plt.figure()
    for method in methods:
        arr = np.stack([s["loss"] for (m, s) in kmax_series if m == method])
        mu, sd = arr.mean(axis=0), arr.std(axis=0)
        plt.plot(t_axis, mu, label=method)
        plt.fill_between(t_axis, mu - sd, mu + sd, alpha=0.15)
    plt.axvspan(args.fail_start, args.fail_start + args.fail_duration,
                color="red", alpha=0.12, label="link failures")
    plt.xlabel("Step")
    plt.ylabel("Loss ratio")
    plt.title(f"E3 ({args.topology}, load {args.load}, "
              f"K={max(levels)}): loss over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "e3_loss_time_kmax.png"), dpi=150)
    plt.close()
    print("Saved plots: e3_loss_vs_k.png, e3_thr_vs_k.png, "
          "e3_disturbance_vs_k.png, e3_recovery_vs_k.png, "
          "e3_loss_time_kmax.png", flush=True)
    print("DONE.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="E3: multiple link failures")
    ap.add_argument("--topology", default="nsfnet", choices=["nsfnet", "fattree"])
    ap.add_argument("--load", type=int, default=0,
                    help="0 = default per topology (nsfnet 80 / fattree 120)")
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--ep-len", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--fail-start", type=int, default=250)
    ap.add_argument("--fail-duration", type=int, default=100)
    ap.add_argument("--qlearn-episodes", type=int, default=1500)
    ap.add_argument("--dqn-steps", type=int, default=50000)
    ap.add_argument("--outdir", default="./results_e3")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.load == 0:
        args.load = LOAD[args.topology]
    if args.fast:
        args.seeds = 1
        args.levels = "1,2"
        args.eval_episodes = 2
        args.ep_len = 120
        args.warmup = 50
        args.fail_start = 70
        args.fail_duration = 25
        args.qlearn_episodes = 100
        args.dqn_steps = 2000
        args.outdir = f"./results_e3_fast_{args.topology}"
    run(args)


if __name__ == "__main__":
    main()
