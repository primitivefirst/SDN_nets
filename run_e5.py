"""Experiment E5: generalization to an unseen topology (no retraining).

Protocol: QL and DQN agents are trained on a SOURCE topology (mixed
loads + random link failures, as in E3), then FROZEN and evaluated on
a TARGET topology under a single top-1 critical-link failure (E2-style
timing: fail at t=250 for 100 steps, react delay 20). SP/ECMP run
natively on the target as anchors. Native (same-topology) agent
numbers already exist in E2/E3 archives for the gap analysis.

Directions: n2f (nsfnet -> fattree) and f2n (fattree -> nsfnet).
DQN transfer is enabled by a padded observation (both topologies use
width P = max link count, unused slots are zeros). QL transfers
natively (bucket state); unseen (s,t) pairs fall back to candidate 0.

Methods on target: sp, ecmp, qlearn_t, dqn_t (_t = transferred).

Outputs (in --outdir, default ./results_e5):
  config.json, results_e5.csv, results_e5_summary.csv,
  e5_loss_bars.png, e5_thr_bars.png,
  e5_loss_time_n2f.png, e5_loss_time_f2n.png

Example (full):  python run_e5.py
Example (fast):  python run_e5.py --fast
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
METHODS = ["sp", "ecmp", "qlearn_t", "dqn_t"]
REROUTE = {"sp": "first", "ecmp": "random",
           "qlearn_t": "ql", "dqn_t": "dqn"}
DIRS = {"n2f": ("nsfnet", "fattree"), "f2n": ("fattree", "nsfnet")}
PAD = 48  # max link count over topologies (fattree k=4)


def pick_top_link(G, hosts, args, load):
    """Top-1 critical link on the target (SP utilization probe)."""
    L = len(list(G.edges()))
    acc = np.zeros(L)
    cnt = 0
    for ep in range(5):
        env = RoutingEnv(G, hosts, hold_mean=load,
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
    top = sorted(infra, key=lambda i: mean_u[i], reverse=True)[0]
    print(f"  target top-1 link: {top} (edge {edge_list[top]})", flush=True)
    return [top]


def make_train_env_fn(G, hosts, args, topo, padded):
    def _fn(seed):
        kw = {"obs_pad_L": PAD} if padded else {}
        return RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[topo],
                          ep_len=args.ep_len, warmup=args.warmup,
                          fail_mode="train_random",
                          fail_duration=args.fail_duration, seed=seed, **kw)
    return _fn


def make_eval_env_fn(G, hosts, args, load, links, method, agent=None):
    def _fn(seed):
        kw = {"obs_pad_L": PAD} if method == "dqn_t" else {}
        env = RoutingEnv(G, hosts, hold_mean=load,
                         ep_len=args.ep_len, warmup=args.warmup,
                         fail_duration=args.fail_duration, seed=seed,
                         reroute_policy=REROUTE[method],
                         reroute_agent=agent, **kw)
        env.set_failures(links, args.fail_start, args.fail_duration)
        return env
    return _fn


def train_dqn(G, hosts, args, topo, seed):
    from stable_baselines3 import DQN
    env = RoutingEnv(G, hosts, hold_mean=TRAIN_LOADS[topo],
                     ep_len=args.ep_len, warmup=args.warmup,
                     fail_mode="train_random",
                     fail_duration=args.fail_duration, seed=seed,
                     obs_pad_L=PAD)
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
    dirs = [d for d in DIRS if args.direction in ("both", d)]
    methods = [m for m in METHODS if m in args.methods]
    seeds = list(range(args.seeds))
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    t0 = time.time()
    all_rows, series = [], []
    for direc in dirs:
        src, tgt = DIRS[direc]
        Gs, hs = get_topology(src)
        Gt, ht = get_topology(tgt)
        print(f"\n===== Direction {direc}: {src} -> {tgt} =====", flush=True)
        print("Measuring target top-1 link (quick SP probe)...", flush=True)
        flink = pick_top_link(Gt, ht, args, LOAD[tgt])
        for si, seed in enumerate(seeds):
            print(f"--- Seed {seed} ({si + 1}/{len(seeds)}) ---", flush=True)
            agents = {}
            if "qlearn_t" in methods:
                print("  training qlearn on source...", flush=True)
                agents["qlearn_t"] = QLearningAgent(k_paths=3).train(
                    make_train_env_fn(Gs, hs, args, src, False),
                    args.qlearn_episodes, seed=1000 + seed, log_every=300)
            if "dqn_t" in methods:
                print("  training dqn on source (padded obs)...", flush=True)
                agents["dqn_t"] = train_dqn(Gs, hs, args, src, seed)
            rng = np.random.default_rng(7000 + seed)
            for method in methods:
                agent = agents.get(method)
                env_fn = make_eval_env_fn(Gt, ht, args, LOAD[tgt], flink,
                                          method, agent)
                base = 5000 + seed * 100000
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
                elif method == "qlearn_t":
                    pol = lambda env, obs: agent.act(env, 0.0, rng)
                else:
                    pol = lambda env, obs: int(
                        agent.predict(obs, deterministic=True)[0])
                for ep in range(args.eval_episodes):
                    m, s = run_episode_collect(env_fn, base + ep, pol)
                    m.update({"direction": direc, "method": method,
                              "seed": seed, "episode": ep})
                    all_rows.append(m)
                    series.append((direc, method, s))
            print(f"  seed {seed}: done", flush=True)
    dt = time.time() - t0
    print(f"\nTotal wall time: {dt / 60:.1f} min", flush=True)

    df = pd.DataFrame(all_rows)
    raw_path = os.path.join(args.outdir, "results_e5.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path} ({len(df)} rows)", flush=True)

    g = df.groupby(["direction", "method"]).agg(
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
    sum_path = os.path.join(args.outdir, "results_e5_summary.csv")
    g.to_csv(sum_path, index=False)
    print(f"Saved {sum_path}", flush=True)
    print("\n=== E5 SUMMARY (mean +/- std) ===", flush=True)
    print(g.to_string(index=False), flush=True)

    order = [m for m in METHODS if m in set(g["method"])]
    xs = np.arange(len(order))
    for col_m, col_s, ylabel, fname in [
            ("window_loss_mean", "window_loss_std", "Loss during failure",
             "e5_loss_bars.png"),
            ("throughput_mean", "throughput_std",
             "Throughput (delivered/step)", "e5_thr_bars.png")]:
        plt.figure()
        w = 0.35
        for ji, direc in enumerate(dirs):
            sub = g[g["direction"] == direc].set_index("method").loc[order]
            plt.bar(xs + (ji - len(dirs) / 2 + 0.5) * w, sub[col_m],
                    yerr=sub[col_s], width=w, capsize=3, label=direc)
        plt.xticks(xs, order)
        plt.ylabel(ylabel)
        plt.title("E5: transferred agents on unseen topology (K=1 failure)")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=150)
        plt.close()
    for direc in dirs:
        t_axis = [s["t"] for (d, m, s) in series if d == direc][0]
        plt.figure()
        for method in order:
            arr = np.stack([s["loss"] for (d, m, s) in series
                            if d == direc and m == method])
            mu, sd = arr.mean(axis=0), arr.std(axis=0)
            plt.plot(t_axis, mu, label=method)
            plt.fill_between(t_axis, mu - sd, mu + sd, alpha=0.15)
        plt.axvspan(args.fail_start, args.fail_start + args.fail_duration,
                    color="red", alpha=0.12, label="link failures")
        plt.xlabel("Step")
        plt.ylabel("Loss ratio")
        src, tgt = DIRS[direc]
        plt.title(f"E5 ({src} -> {tgt}, load {LOAD[tgt]}): loss over time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"e5_loss_time_{direc}.png"),
                    dpi=150)
        plt.close()
    print("Saved plots: e5_loss_bars.png, e5_thr_bars.png, " +
          ", ".join(f"e5_loss_time_{d}.png" for d in dirs), flush=True)
    print("DONE.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="E5: unseen-topology transfer")
    ap.add_argument("--direction", default="both",
                    choices=["both", "n2f", "f2n"])
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--ep-len", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--fail-start", type=int, default=250)
    ap.add_argument("--fail-duration", type=int, default=100)
    ap.add_argument("--qlearn-episodes", type=int, default=1500)
    ap.add_argument("--dqn-steps", type=int, default=50000)
    ap.add_argument("--outdir", default="./results_e5")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        args.seeds = 1
        args.eval_episodes = 2
        args.ep_len = 120
        args.warmup = 50
        args.fail_start = 70
        args.fail_duration = 25
        args.qlearn_episodes = 100
        args.dqn_steps = 2000
        args.outdir = "./results_e5_fast"
    run(args)


if __name__ == "__main__":
    main()
