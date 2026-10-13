"""Dynamic flow-level SDN routing environment (Gymnasium-compatible).

Online model: every step, previous flows may depart (geometric holding
time) and exactly one new flow arrives and must be routed immediately
from k precomputed candidate paths. Load level = mean holding time
(more concurrent flows). Metrics = time averages over steps.

`hold_mean` can be a scalar or a list: with a list, each episode draws
its holding time at random (mixed-load training for RL agents).

Link failures (E2+): a SET of links can fail at `fail_start` for
`fail_duration` steps (E2: 1 link, E3: 1-4 links, E4: many/nodes).
  * Data plane (instant, protection-like): active flows traversing any
    failed link are immediately rerouted onto alternative precomputed
    candidates (policy per method, see `reroute_policy`); flows with no
    alternative are dropped and counted as undelivered. Each reroute
    counts as one unit of rerouting disturbance.
  * Control plane (delayed): candidate paths are recomputed WITHOUT the
    failed links only after `react_delay` steps; arrivals routed onto a
    failed link in between are dropped. RL agents observe failures
    immediately via a failed-link mask.
  * Recovery restores full paths instantly (optimistic assumption).
Set explicitly via `set_failure()` / `set_failures()` or randomly per
episode with `fail_mode='train_random'` (1-3 random infra links).

Traffic realism (simple, all open):
  * mice (0.5-1.5) + 10% elephant flows (5-10);
  * hotspot skew: 40% of flows go to 3 hot endpoints (servers);
  * heterogeneous link capacities: half 8.0, half 20.0 (seeded).
Link delay = 1 + 5*u^2, u = load/capacity (M/M/1-style).
Loss: a flow delivers d / max(1, max_link_util_on_path).
"""
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces


class RoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, G, endpoints, k_paths=3, hold_mean=40,
                 ep_len=300, warmup=50, demand_hi=10.0,
                 elephant_frac=0.1, hotspot_frac=0.4, seed=0,
                 fail_mode=None, fail_duration=70, react_delay=20,
                 reroute_policy="first", reroute_agent=None,
                 obs_failmask=True, overload_w=20.0, protection=True, obs_pad_L=None):
        super().__init__()
        self.G = G
        self.nodes = list(G.nodes())
        self.n = len(self.nodes)
        self.node_pos = {v: i for i, v in enumerate(self.nodes)}
        self.edges = list(G.edges())
        self.L = len(self.edges)
        self.P = int(obs_pad_L) if obs_pad_L else self.L  # padded obs width (E5)
        self.eindex = {}
        for i, (u, v) in enumerate(self.edges):
            self.eindex[(u, v)] = i
            self.eindex[(v, u)] = i
        self.k = int(k_paths)
        self.hold_mean = hold_mean  # scalar or list (mixed-load training)
        self.cur_hold = hold_mean if np.isscalar(hold_mean) else hold_mean[0]
        self.ep_len = int(ep_len)
        self.warmup = int(warmup)
        self.demand_hi = float(demand_hi)
        self.elephant_frac = float(elephant_frac)
        self.hotspot_frac = float(hotspot_frac)
        self.fail_mode = fail_mode  # None | 'train_random' | 'train_mixed'
        self.fail_duration = int(fail_duration)
        self.react_delay = int(react_delay)
        # Reroute policy for affected flows: 'first' | 'random' | 'ql' | 'dqn'.
        self.reroute_policy = reroute_policy
        self.reroute_agent = reroute_agent
        # E6 ablation switches (defaults = full method).
        self.obs_failmask = bool(obs_failmask)  # failed-link mask in obs
        self.overload_w = float(overload_w)     # overload term in reward
        self.protection = bool(protection)      # instant backup at failure
        self.fixed_failure = None  # (links, start, dur) via set_failures()
        self.fixed_nodes = None    # (nodes, start, dur) via set_node_failures()
        self.endpoints = list(endpoints)
        self.rng = np.random.default_rng(seed)
        # Heterogeneous capacities: half thin (8), half fat (20), seeded.
        perm = self.rng.permutation(self.L)
        self.caps = np.full(self.L, 20.0)
        self.caps[perm[:self.L // 2]] = 8.0
        # Links eligible for random failures: exclude links incident to
        # degree-1 nodes (failing them isolates a host = uninformative).
        self._fail_candidates = [
            i for i, (a, b) in enumerate(self.edges)
            if self.G.degree(a) > 1 and self.G.degree(b) > 1]
        if not self._fail_candidates:  # fallback: all links
            self._fail_candidates = list(range(self.L))
        # Nodes eligible for random failures (E4): infra nodes; if every
        # node is an endpoint (nsfnet), all nodes.
        non_ep = [v for v in self.nodes if v not in self.endpoints]
        self._fail_node_candidates = non_ep or list(self.nodes)
        # Hot endpoints = 3 highest-degree nodes among endpoints.
        deg = sorted(self.endpoints, key=lambda v: self.G.degree(v),
                     reverse=True)
        self.hot = deg[:3]
        self.paths = {}
        self.min_len = {}
        self._compute_paths()
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3 + self.P + self.k + self.P,), dtype=np.float32)
        self.loads = np.zeros(self.L)
        self.active = []      # list of [remaining_hold, demand, links, s, t]
        self.cur_flow = None  # (s, t, d): arrival to route now
        self.t = 0
        self.step_stats = []  # per-step (t, demanded, delivered, delay, n)
        self.failed_links = set()
        self.failed_nodes = set()
        self.connectivity_at_fail = 1.0
        self.failed_mask = np.zeros(self.L, dtype=np.float32)
        self._recompute_at = None
        self.step_drop = 0.0
        self._pending_fail_drop = 0.0
        self.ep_fail_links = []
        self.ep_fail_nodes = []
        self.ep_fail_start = None
        self.ep_fail_end = None
        self.affected_n = 0
        self.affected_dem = 0.0
        self.rerouted_n = 0

    # -- candidate paths --------------------------------------------------
    def _compute_paths(self, exclude_edges=None, exclude_nodes=None):
        """(Re)compute k candidates per pair, optionally w/o failed edges.

        NOTE: edges are removed from a graph COPY first. Filtering Yen's
        generator instead would hang: when no valid path exists, the
        generator would enumerate an exponential number of simple paths
        that all get filtered out.
        """
        self.paths = {}
        self.min_len = {}
        G = self.G
        if exclude_edges or exclude_nodes:
            G = G.copy()
            for ei in (exclude_edges or []):
                a, b = self.edges[ei]
                if G.has_edge(a, b):
                    G.remove_edge(a, b)
            for v in (exclude_nodes or []):
                if G.has_node(v):
                    G.remove_node(v)
        for s in self.endpoints:
            for t in self.endpoints:
                if s == t:
                    continue
                cands = []
                try:
                    for p in nx.shortest_simple_paths(G, s, t):
                        cands.append(p)
                        if len(cands) >= self.k + 2:
                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
                cands = sorted(cands, key=len)[:self.k]
                self.paths[(s, t)] = cands  # may be [] if disconnected
                if cands:
                    self.min_len[(s, t)] = min(len(p) for p in cands)

    def _uses_failed(self, links):
        return any(li in self.failed_links for li in links)

    def _path_avoids(self, path, failed):
        for i in range(len(path) - 1):
            if self.eindex[(path[i], path[i + 1])] in failed:
                return False
        return True

    def set_failures(self, links, start, duration=None):
        """Fix a failure schedule used by every subsequent episode."""
        dur = self.fail_duration if duration is None else int(duration)
        self.fixed_failure = ([int(x) for x in links], int(start), int(dur))
        self.fixed_nodes = None

    def set_failure(self, link, start, duration=None):
        """Single-link variant (kept for E2 compatibility)."""
        self.set_failures([link], start, duration)

    def set_node_failures(self, nodes, start, duration=None):
        """Fix a NODE failure schedule (E4); clears link schedule."""
        dur = self.fail_duration if duration is None else int(duration)
        self.fixed_nodes = ([int(v) for v in nodes], int(start), int(dur))
        self.fixed_failure = None

    # -- internal helpers -------------------------------------------------
    def _draw_flow(self):
        m = len(self.endpoints)
        s = self.endpoints[int(self.rng.integers(m))]
        if self.rng.random() < self.hotspot_frac:
            t = self.hot[int(self.rng.integers(len(self.hot)))]
            if t == s:
                t = self.endpoints[int(self.rng.integers(m))]
                while t == s:
                    t = self.endpoints[int(self.rng.integers(m))]
        else:
            t = s
            while t == s:
                t = self.endpoints[int(self.rng.integers(m))]
        if self.rng.random() < self.elephant_frac:
            d = float(self.rng.uniform(5.0, self.demand_hi))
        else:
            d = float(self.rng.uniform(0.5, 1.5))
        return (s, t, d)

    def _path_links(self, path):
        return [self.eindex[(path[i], path[i + 1])] for i in range(len(path) - 1)]

    def _utils(self):
        return self.loads / self.caps

    def _obs_for(self, flow):
        s, t, d = flow
        o = np.zeros(3 + self.P + self.k + self.P, dtype=np.float32)
        o[0] = self.node_pos[s] / max(1, self.n - 1)
        o[1] = self.node_pos[t] / max(1, self.n - 1)
        o[2] = d / self.demand_hi
        u = self._utils()
        m = min(self.L, self.P)
        o[3:3 + m] = np.clip(u[:m], 0.0, 2.0) / 2.0
        for i, p in enumerate(self.paths[(s, t)]):
            links = self._path_links(p)
            o[3 + self.P + i] = float(np.max(u[links])) / 2.0 if links else 0.0
        mask = self.failed_mask if self.obs_failmask else np.zeros(self.L)
        o[3 + self.P + self.k:3 + self.P + self.k + m] = mask[:m]
        return o

    def _step_snapshot(self):
        u = self._utils()
        demanded = self.step_drop + self._pending_fail_drop
        self._pending_fail_drop = 0.0
        delivered, delay_sum = 0.0, 0.0
        for (_rem, d, links, _s, _t) in self.active:
            umax = float(np.max(u[links])) if links else 0.0
            fdelay = float(np.sum(1.0 + 5.0 * u[links] ** 2)) if links else 0.0
            demanded += d
            delivered += d / max(1.0, umax)
            delay_sum += fdelay
        return (demanded, delivered, delay_sum, len(self.active))

    def _reroute_choice(self, s, t, d, avail):
        """Choose backup candidate index for an affected (s,t,d) flow."""
        pol = self.reroute_policy
        if pol == "random":
            return int(self.rng.choice(avail))
        if pol == "ql" and self.reroute_agent is not None:
            st = (int(s), int(t), self.global_congestion_bucket(),
                  self.sp_path_congestion_bucket(s, t))
            q = self.reroute_agent._q(st)
            best = max(avail, key=lambda i: q[i])
            return int(best)
        if pol == "dqn" and self.reroute_agent is not None:
            a, _ = self.reroute_agent.predict(
                self._obs_for((s, t, d)), deterministic=True)
            a = int(a)
            if a in avail:
                return a
            return int(avail[0])  # fallback if net picks a dead path
        return int(avail[0])  # 'first' (SP-style) and default

    def _apply_failures(self, links):
        self.failed_links = set(int(x) for x in links)
        self.failed_mask = np.zeros(self.L, dtype=np.float32)
        for li in self.failed_links:
            self.failed_mask[li] = 1.0
        # Instant data-plane protection: reroute affected actives.
        still = []
        for f in self.active:
            rem, d, old_links, s, t = f
            if not self._uses_failed(old_links):
                still.append(f)
                continue
            cands = self.paths[(s, t)]
            avail = [i for i, p in enumerate(cands)
                     if self._path_avoids(p, self.failed_links)]
            if not self.protection:
                avail = []  # E6 'noprotect': restoration-only, drop all
            if not avail:
                for li in old_links:
                    self.loads[li] -= d
                self.affected_n += 1
                self.affected_dem += d
                self._pending_fail_drop += d
            else:
                j = self._reroute_choice(s, t, d, avail)
                new_links = self._path_links(cands[j])
                for li in old_links:
                    self.loads[li] -= d
                for li in new_links:
                    self.loads[li] += d
                self.rerouted_n += 1
                still.append([rem, d, new_links, s, t])
        self.active = still
        self.loads = np.maximum(self.loads, 0.0)
        # Control-plane recomputation happens later (stale window).
        self._recompute_at = self.t + self.react_delay
        self.connectivity_at_fail = self._connectivity()

    def _incident_links(self, nodes):
        """Edge indices incident to any of `nodes`."""
        out = set()
        for v in nodes:
            for nb in self.G.neighbors(v):
                out.add(self.eindex[(v, nb)])
        return sorted(out)

    def _connectivity(self):
        """Fraction of endpoint pairs with a failure-free candidate."""
        ok, tot = 0, 0
        for s in self.endpoints:
            for t in self.endpoints:
                if s == t:
                    continue
                tot += 1
                if s in self.failed_nodes or t in self.failed_nodes:
                    continue
                for c in self.paths[(s, t)]:
                    if self._path_avoids(c, self.failed_links):
                        ok += 1
                        break
        return ok / tot if tot else 1.0

    def _apply_node_failures(self, nodes):
        """Fail NODES (E4): dead endpoints drop, incident links fail."""
        self.failed_nodes = set(int(v) for v in nodes)
        still = []
        for f in self.active:
            rem, d, old_links, s, t = f
            if s in self.failed_nodes or t in self.failed_nodes:
                for li in old_links:
                    self.loads[li] -= d
                self.affected_n += 1
                self.affected_dem += d
                self._pending_fail_drop += d
            else:
                still.append(f)
        self.active = still
        self._apply_failures(self._incident_links(self.failed_nodes))

    def _clear_failures(self):
        self.failed_links = set()
        self.failed_nodes = set()
        self.failed_mask = np.zeros(self.L, dtype=np.float32)
        self._recompute_at = None
        self._compute_paths()

    # -- Gym API ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if isinstance(self.hold_mean, (list, tuple)):
            self.cur_hold = float(self.rng.choice(self.hold_mean))
        else:
            self.cur_hold = float(self.hold_mean)
        # Failure schedule for this episode.
        self._clear_failures()
        self.ep_fail_links = []
        self.ep_fail_nodes = []
        self.ep_fail_start = self.ep_fail_end = None
        if self.fail_mode == "train_random":
            k = int(self.rng.integers(1, 4))  # 1-3 random links
            links = self.rng.choice(self._fail_candidates, size=k,
                                    replace=False).tolist()
            start = int(self.rng.integers(100, 181))
            self.ep_fail_links = [int(x) for x in links]
            self.ep_fail_start = start
            self.ep_fail_end = start + self.fail_duration
        elif self.fail_mode == "train_mixed":
            start = int(self.rng.integers(100, 181))
            self.ep_fail_start = start
            self.ep_fail_end = start + self.fail_duration
            if self.rng.random() < 0.5:
                k = int(self.rng.integers(1, 4))  # 1-3 links
                links = self.rng.choice(self._fail_candidates, size=k,
                                        replace=False).tolist()
                self.ep_fail_links = [int(x) for x in links]
            else:
                k = int(self.rng.integers(1, 3))  # 1-2 nodes
                nodes = self.rng.choice(self._fail_node_candidates, size=k,
                                        replace=False).tolist()
                self.ep_fail_nodes = [int(v) for v in nodes]
        elif self.fixed_failure is not None:
            links, start, dur = self.fixed_failure
            self.ep_fail_links = list(links)
            self.ep_fail_start = start
            self.ep_fail_end = start + dur
        elif self.fixed_nodes is not None:
            nodes, start, dur = self.fixed_nodes
            self.ep_fail_nodes = list(nodes)
            self.ep_fail_start = start
            self.ep_fail_end = start + dur
        self.loads = np.zeros(self.L)
        self.active = []
        self.t = 0
        self.step_stats = []
        self.affected_n = 0
        self.affected_dem = 0.0
        self.rerouted_n = 0
        self.step_drop = 0.0
        self._pending_fail_drop = 0.0
        self.cur_flow = self._draw_flow()
        return self._obs_for(self.cur_flow), {}

    def step(self, action):
        # 1) Age active flows, depart expired ones.
        for f in self.active:
            f[0] -= 1
        still = []
        for f in self.active:
            if f[0] <= 0:
                self.loads[f[2]] -= f[1]
            else:
                still.append(f)
        self.active = still
        self.loads = np.maximum(self.loads, 0.0)
        # 2) Failure trigger / delayed path recomputation / recovery.
        if self.ep_fail_links or self.ep_fail_nodes:
            if self.t == self.ep_fail_start:
                if self.ep_fail_nodes:
                    self._apply_node_failures(self.ep_fail_nodes)
                else:
                    self._apply_failures(self.ep_fail_links)
            elif self.t == self.ep_fail_end:
                self._clear_failures()
        if self._recompute_at is not None and self.t >= self._recompute_at \
                and (self.failed_links or self.failed_nodes):
            self._compute_paths(exclude_edges=sorted(self.failed_links),
                                exclude_nodes=sorted(self.failed_nodes))
            self._recompute_at = None
        # 3) Route the arriving flow (drop if no path or path uses dead link).
        self.step_drop = 0.0
        s, t, d = self.cur_flow
        if s in self.failed_nodes or t in self.failed_nodes:
            cands = []
        else:
            cands = self.paths[(s, t)]
        if not cands:
            self.step_drop = d
            reward = -(5.0 + d) / 10.0
        else:
            path = cands[int(action) % len(cands)]
            links = self._path_links(path)
            if self._uses_failed(links):
                self.step_drop = d  # sent into a failed link: lost
                reward = -(5.0 + d) / 10.0
            else:
                hold = int(self.rng.geometric(1.0 / self.cur_hold))
                new_loads = self.loads.copy()
                new_loads[links] += d
                u = new_loads / self.caps
                ul = u[links]
                cost = float(np.sum(1.0 + 5.0 * ul ** 2)
                             + self.overload_w * np.sum(np.maximum(0.0, ul - 1.0)))
                reward = -cost / 10.0
                self.loads = new_loads
                self.active.append([hold, d, links, s, t])
        # 4) Snapshot metrics (after warmup).
        if self.t >= self.warmup:
            dd, dl, ds, nn = self._step_snapshot()
            self.step_stats.append((self.t, dd, dl, ds, nn))
        self.t += 1
        terminated = self.t >= self.ep_len
        # 5) Draw next arrival and observe it.
        if not terminated:
            self.cur_flow = self._draw_flow()
            obs = self._obs_for(self.cur_flow)
        else:
            obs = np.zeros(3 + self.P + self.k + self.P, dtype=np.float32)
        return obs, reward, terminated, False, {}

    # -- paper metrics (time averages) ------------------------------------
    def compute_metrics(self):
        if not self.step_stats:
            return {"throughput": 0.0, "avg_delay": 0.0, "loss_ratio": 0.0}
        thr, dly, loss, cnt = 0.0, 0.0, 0.0, 0
        for (_t, demanded, delivered, delay_sum, n) in self.step_stats:
            thr += delivered
            if n > 0:
                dly += delay_sum / n
                loss += 1.0 - delivered / max(1e-9, demanded)
                cnt += 1
        n_steps = max(1, len(self.step_stats))
        cnt = max(1, cnt)
        return {"throughput": float(thr / n_steps),
                "avg_delay": float(dly / cnt),
                "loss_ratio": float(max(0.0, loss / cnt))}

    def get_series(self):
        """Per-step time series: dict(t, throughput, delay, loss)."""
        t, thr, dly, loss = [], [], [], []
        for (ti, demanded, delivered, delay_sum, n) in self.step_stats:
            t.append(ti)
            thr.append(delivered)
            if n > 0:
                dly.append(delay_sum / n)
                loss.append(max(0.0, 1.0 - delivered / max(1e-9, demanded)))
            else:
                dly.append(0.0)
                loss.append(0.0)
        return {"t": np.array(t), "throughput": np.array(thr),
                "delay": np.array(dly), "loss": np.array(loss)}

    def failure_metrics(self, tol=0.02, roll=10, base_win=30):
        """Recovery stats for the scheduled failure (None if no failure)."""
        if not self.ep_fail_links and not self.ep_fail_nodes:
            return None
        s = self.get_series()
        t, loss = s["t"], s["loss"]
        pre = loss[t < self.ep_fail_start][-base_win:]
        baseline = float(np.mean(pre)) if len(pre) else 0.0
        in_win = (t >= self.ep_fail_start) & (t < self.ep_fail_end)
        win_loss = float(np.mean(loss[in_win])) if np.any(in_win) else 0.0
        rec = self.ep_len - self.ep_fail_start  # default: never recovered
        for i in range(len(t)):
            if t[i] < self.ep_fail_start:
                continue
            seg = loss[i:i + roll]
            if len(seg) == roll and float(np.mean(seg)) <= baseline + tol:
                rec = int(t[i] - self.ep_fail_start)
                break
        return {"base_loss": baseline, "window_loss": win_loss,
                "connectivity": float(self.connectivity_at_fail),
                "n_nodes": len(self.ep_fail_nodes),
                "recovery_time": rec, "affected_n": self.affected_n,
                "affected_dem": float(self.affected_dem),
                "rerouted_n": self.rerouted_n}

    def global_congestion_bucket(self):
        mean_u = float(np.mean(self._utils()))
        if mean_u < 0.5:
            return 0
        if mean_u < 0.85:
            return 1
        return 2

    def sp_path_congestion_bucket(self, s, t):
        cands = self.paths[(s, t)]
        if not cands:
            return 2
        links = self._path_links(cands[0])
        umax = float(np.max(self._utils()[links])) if links else 0.0
        if umax < 0.6:
            return 0
        if umax < 1.0:
            return 1
        return 2
