"""Experiment E6: ablation of the DQN rerouting method's components.

Protocol: fixed double-link failure (top-2 critical links, nested set
as in E3), same timing as E3 (fail at t=250 for 100 steps, react
delay 20). Four DQN variants, each trained in its own environment
configuration (random 1-3 link failures, mixed loads, as in E3):

  full       - complete method (failure mask in obs, overload penalty
               in reward, instant protection at failure);
  nofailmask - failed-link mask zeroed in the observation;
  nopenalty  - reward without the overload term (delay-like cost only);
  noprotect  - restoration-only: no instant backup at failure moment.

Load: single representative high load (nsfnet 80 / fattree 120).

Outputs (in --outdir, default ./results_e6):
  config.json, results_e6.csv, results_e6_summary.csv,
  e6_loss_bars.png, e6_thr_bars.png, e6_disturbance_bars.png,
  e6_recovery_bars.png, e6_loss_time.png

Example (full):  python run_e6.py
Example (fast):  python run_e6.py --fast
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

LOAD = {"nsfnet": 80, "fattree": 120}
TRAIN_LOADS = {"nsfnet": [20, 40, 60, 80], "fattree": [30, 60, 90, 120]}
VARIANTS = {
    "full": {},
    "nofailmask": {"obs_failmask": False},
    "nopenalty": {"overload_w": 0.0},
    "noprotect": {"protection": False},
}


def pick_critical_links(G, hosts, args, n_links=2):
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


def make_eval_env_fn(G, hosts, args, links, vkw, agent):
    def _fn(seed):
        env = RoutingEnv(G, hosts, hold_mean=args.load,
                         ep_len=args.ep_len, warmup=args.warmup,
                         fail_duration=args.fail_duration, seed=seed,
                         reroute_policy="dqn", reroute_agent=agent, **vkw)
        env.set_failures(links, args.fail_start, args.fail_duration)
        return env
    return _fn


def train_dqn(G, hosts, args, seed, vkw):
    from stable_baselines3 import DQN
    env = RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[args.topology],
                     ep_len=args.ep_len, warmup=args.warmup,
                     fail_mode="train_random",
                     fail_duration=args.fail_duration, seed=seed, **vkw)
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
    variants = [v for v in VARIANTS if v in args.variants]
    seeds = list(range(args.seeds))
    print("Measuring critical links (quick SP probe)...", flush=True)
    crit = pick_critical_links(G, hosts, args, args.K)
    print(f"Load: {args.load} | Failure: top-{args.K} links {crit} fail at "
          f"t={args.fail_start} for {args.fail_duration} steps "
          f"(react delay 20)", flush=True)
    print(f"Variants: {variants} | Seeds: {seeds} | "
          f"Episodes per (variant, seed): {args.eval_episodes}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        cfg = vars(args)
        cfg["critical_order"] = crit
        json.dump(cfg, f, indent=2)

    t0 = time.time()
    all_rows, series = [], []
    for si, seed in enumerate(seeds):
        print(f"\n=== Seed {seed} ({si + 1}/{len(seeds)}) ===", flush=True)
        for variant in variants:
            vkw = VARIANTS[variant]
            print(f"  training dqn [{variant}] (with failures)...",
                  flush=True)
            agent = train_dqn(G, hosts, args, 100 + seed, vkw)
            env_fn = make_eval_env_fn(G, hosts, args, crit, vkw, agent)
            pol = lambda env, obs: int(
                agent.predict(obs, deterministic=True)[0])
            base = 5000 + seed * 100000
            for ep in range(args.eval_episodes):
                m, s = run_episode_collect(env_fn, base + ep, pol)
                m.update({"variant": variant, "seed": seed,
                          "episode": ep})
                all_rows.append(m)
                series.append((variant, s))
            print(f"  [{variant}]: done", flush=True)
    dt = time.time() - t0
    print(f"\nTotal wall time: {dt / 60:.1f} min", flush=True)

    df = pd.DataFrame(all_rows)
    raw_path = os.path.join(args.outdir, "results_e6.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path} ({len(df)} rows)", flush=True)

    g = df.groupby(["variant"]).agg(
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
    sum_path = os.path.join(args.outdir, "results_e6_summary.csv")
    g.to_csv(sum_path, index=False)
    print(f"Saved {sum_path}", flush=True)
    print("\n=== E6 SUMMARY (mean +/- std) ===", flush=True)
    print(g.to_string(index=False), flush=True)

    order = [v for v in VARIANTS if v in set(g["variant"])]
    g = g.set_index("variant").loc[order].reset_index()
    xs = np.arange(len(order))
    for col_m, col_s, ylabel, fname in [
            ("window_loss_mean", "window_loss_std", "Loss during failure",
             "e6_loss_bars.png"),
            ("throughput_mean", "throughput_std",
             "Throughput (delivered/step)", "e6_thr_bars.png"),
            ("rerouted_mean", "rerouted_std", "Rerouted flows (disturbance)",
             "e6_disturbance_bars.png"),
            ("recovery_mean", "recovery_std", "Recovery time (steps)",
             "e6_recovery_bars.png")]:
        plt.figure()
        plt.bar(xs, g[col_m], yerr=g[col_s], capsize=4,
                color=["C0", "C1", "C2", "C3"][:len(order)])
        plt.xticks(xs, order)
        plt.ylabel(ylabel)
        plt.title(f"E6 ({args.topology}, load {args.load}, K={args.K})")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=150)
        plt.close()
    # Time course.
    t_axis = series[0][1]["t"]
    plt.figure()
    for variant in order:
        arr = np.stack([s["loss"] for (v, s) in series if v == variant])
        mu, sd = arr.mean(axis=0), arr.std(axis=0)
        plt.plot(t_axis, mu, label=variant)
        plt.fill_between(t_axis, mu - sd, mu + sd, alpha=0.15)
    plt.axvspan(args.fail_start, args.fail_start + args.fail_duration,
                color="red", alpha=0.12, label="link failures")
    plt.xlabel("Step")
    plt.ylabel("Loss ratio")
    plt.title(f"E6 ({args.topology}, load {args.load}, "
              f"K={args.K}): loss over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "e6_loss_time.png"), dpi=150)
    plt.close()
    print("Saved plots: e6_loss_bars.png, e6_thr_bars.png, "
          "e6_disturbance_bars.png, e6_recovery_bars.png, "
          "e6_loss_time.png", flush=True)
    print("DONE.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="E6: DQN ablation")
    ap.add_argument("--topology", default="nsfnet", choices=["nsfnet", "fattree"])
    ap.add_argument("--load", type=int, default=0,
                    help="0 = default per topology (nsfnet 80 / fattree 120)")
    ap.add_argument("--K", type=int, default=2,
                    help="number of failed top-critical links")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--ep-len", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--fail-start", type=int, default=250)
    ap.add_argument("--fail-duration", type=int, default=100)
    ap.add_argument("--dqn-steps", type=int, default=50000)
    ap.add_argument("--outdir", default="./results_e6")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.load == 0:
        args.load = LOAD[args.topology]
    if args.fast:
        args.seeds = 1
        args.eval_episodes = 2
        args.ep_len = 120
        args.warmup = 50
        args.fail_start = 70
        args.fail_duration = 25
        args.dqn_steps = 2000
        args.outdir = f"./results_e6_fast_{args.topology}"
    run(args)


if __name__ == "__main__":
    main()
