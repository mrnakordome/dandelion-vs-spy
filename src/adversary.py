"""The adversary: spy placement and source-estimation algorithms.

Knowledge model (assignment, part 1.3):
  * the adversary knows the topology and the **base** delay of every link;
  * it has no global view - it only sees the packets that reach one of its
    bribed (spy) nodes, and for each such reception it learns the packet id,
    the Dandelion state (S/F), the neighbour that delivered it and the local
    arrival time;
  * it never drops or blocks a packet; in phase 5 it may delay its own relays.

Estimators implemented
  * ``first_spy``            - the mandatory baseline ("first spy to observe").
  * ``timing_ml``            - phase 2: maximum-likelihood localisation from the
                               first-arrival times at all spies (Gaussian model
                               of the jitter) + a hard consistency filter on the
                               observed last hop.
  * ``dandelion_ml``         - phase 4: combines (a) a reverse random-walk
                               likelihood for the stem phase and (b) the timing
                               likelihood of the fluff phase mapped back through
                               the stem kernel.
Both ML estimators accept a ``spy_extra_delay`` model so that in phase 5 the
adversary can compensate for the artificial delay it injects itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import graphutil as G
from .topology import Topology

JITTER_VAR_COEF = (0.2 ** 2) / 3.0     # Var[U(-0.2d, 0.2d)] = (0.2 d)^2 / 3


# --------------------------------------------------------------------------
# network model shared by the estimators (precomputed once per experiment)
# --------------------------------------------------------------------------
class NetworkModel:
    def __init__(self, topo: Topology, spies: Sequence[int] = (),
                 spy_delay_factor: float = 1.0):
        """``spy_delay_factor`` > 1 tells the adversary that its own spies hold
        packets before relaying them (phase 5): the cost of every link leaving
        a spy is multiplied by that factor (2.0 for the maximum legal delay,
        which equals one base delay)."""
        self.topo = topo
        self.n = topo.n
        self.spies = list(spies)
        self.spy_delay_factor = spy_delay_factor
        self.adj = [sorted(a) for a in topo.adjacency()]
        self.deg = [len(a) for a in self.adj]
        self.base = topo.delay_map()
        sset = set(self.spies)
        # directed cost, as seen by the adversary's model of the network
        self.dmap = {(u, v): d * (spy_delay_factor if u in sset else 1.0)
                     for (u, v), d in self.base.items()}
        wadj = G.weighted_adjacency(self.n, topo.edges, topo.base_delay_ms)
        wadj = [[(v, self.dmap[(u, v)]) for v, _ in row] for u, row in enumerate(wadj)]
        self.wadj = wadj

        self.dist = np.zeros((self.n, self.n))
        self.sq = np.zeros((self.n, self.n))       # sum of d_i^2 along the path
        self.prev = np.full((self.n, self.n), -1, dtype=int)
        for s in range(self.n):
            d, prev = G.dijkstra(self.n, wadj, s)
            self.dist[s] = d
            self.prev[s] = prev
            for t in range(self.n):
                acc, cur = 0.0, t
                while cur != s and self.prev[s][cur] >= 0:
                    par = int(self.prev[s][cur])
                    acc += self.base[(par, cur)] ** 2   # jitter scales with base
                    cur = par
                self.sq[s][t] = acc
        self.var = JITTER_VAR_COEF * self.sq + 1e-9   # ms^2

        self._kernels: Dict[float, Tuple[np.ndarray, np.ndarray, Dict]] = {}

    # ------------------------------------------------------- stem kernels
    def kernels(self, p: float):
        """Return (stem_kernel, fluff_kernel, edge_index) for a given p.

        The stem walk is modelled on *directed edges* (so that the
        non-backtracking rule is exact):

            T[(a->b), (b->c)] = p / (deg(b) - 1)      c != a

        ``M = sum_k T^k = (I - T)^-1`` then gives, for every directed edge e,
        the expected number of stem walks that pass through it before reaching
        another edge.

        * ``stem[e][v]`` = P(source = v and the walk performed edge e in stem
          state) - used when a spy observes a *stem* packet arriving over e.
        * ``fluff[v][f]`` = P(the packet started at v fluffs at node f).
        """
        if p in self._kernels:
            return self._kernels[p]
        dedges: List[Tuple[int, int]] = []
        for u in range(self.n):
            for v in self.adj[u]:
                dedges.append((u, v))
        eidx = {e: i for i, e in enumerate(dedges)}
        m = len(dedges)
        T = np.zeros((m, m))
        for (a, b), i in eidx.items():
            others = [c for c in self.adj[b] if c != a]
            if not others:
                continue
            w = p / len(others)
            for c in others:
                T[i, eidx[(b, c)]] = w
        M = np.linalg.inv(np.eye(m) - T)

        # stem[v, e] : source v -> ... -> edge e traversed in stem state
        stem = np.zeros((self.n, m))
        for v in range(self.n):
            for w in self.adj[v]:
                stem[v] += M[eidx[(v, w)]] / self.deg[v]

        # fluff[v, f] : source v, walk stops (fluffs) at node f
        col_f = np.zeros((m, self.n))
        for f in range(self.n):
            for a in self.adj[f]:
                col_f[:, f] += M[:, eidx[(a, f)]]
        fluff = np.zeros((self.n, self.n))
        for v in range(self.n):
            acc = np.zeros(self.n)
            for w in self.adj[v]:
                acc += col_f[eidx[(v, w)]]
            fluff[v] = (1.0 - p) * acc / self.deg[v]
        # a walk that comes back to an already-visited node fluffs there as
        # well; the residual mass is attributed to the last visited node.
        rs = fluff.sum(axis=1, keepdims=True)
        fluff = fluff / np.maximum(rs, 1e-12)
        self._kernels[p] = (stem, fluff, eidx)
        return self._kernels[p]


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------
@dataclass
class Obs:
    """One reception seen by the adversary."""
    spy: int
    frm: int
    state: str
    t: float


def first_arrivals(obs: Sequence[Obs], state: Optional[str] = None
                   ) -> Dict[int, Obs]:
    out: Dict[int, Obs] = {}
    for o in sorted(obs, key=lambda x: x.t):
        if state is not None and o.state != state:
            continue
        if o.spy not in out:
            out[o.spy] = o
    return out


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def first_spy(obs: Sequence[Obs], spies: Sequence[int], n: int) -> int:
    """Baseline: guess the neighbour that delivered the packet to the first spy.

    (Guessing the spy itself would be pointless: sources are honest nodes.)
    """
    if not obs:
        return -1
    ordered = sorted(obs, key=lambda x: x.t)
    sset = set(spies)
    for o in ordered:
        if o.frm not in sset:
            return o.frm
    return ordered[0].frm


def _chi2(model: NetworkModel, arrivals: Dict[int, Obs], cand: Sequence[int]
          ) -> np.ndarray:
    """Weighted least squares residual of the free-origin-time delay model.

    Arrival at spy s of a packet born at v at the unknown instant t0 is
    ``t_s = t0 + d(v,s) + noise``, with ``Var[noise] = (0.2^2/3) * sum d_i^2``
    along the path.  Profiling t0 out gives the weighted residual below.
    """
    spies = np.array(list(arrivals), dtype=int)
    t = np.array([arrivals[int(s)].t for s in spies]) * 1000.0     # ms
    C = np.asarray(cand, dtype=int)
    d = model.dist[np.ix_(C, spies)]
    var = model.var[np.ix_(C, spies)]
    w = 1.0 / var
    r = t[None, :] - d
    t0 = (w * r).sum(axis=1) / w.sum(axis=1)
    chi = (w * (r - t0[:, None]) ** 2).sum(axis=1)
    return np.where(np.isfinite(chi), chi, np.inf)


def _predecessor_violations(model: NetworkModel, arrivals: Dict[int, Obs],
                            cand: Sequence[int], tol_rel: float = 0.25,
                            tol_abs: float = 2.0) -> np.ndarray:
    """How many observed last hops are incompatible with candidate v.

    Under flooding the first copy reaching a spy travels a (near) shortest
    path, so the observed previous hop u must satisfy
    ``d(v,u) + delay(u,s) ~= d(v,s)``.
    """
    C = np.asarray(cand, dtype=int)
    viol = np.zeros(len(C))
    for s, o in arrivals.items():
        link = model.dmap.get((o.frm, s))
        if link is None:
            continue
        got = model.dist[C, o.frm] + link
        exp = model.dist[C, s]
        viol += (got - exp > tol_rel * exp + tol_abs).astype(float)
    return viol


def timing_ml(obs: Sequence[Obs], model: NetworkModel,
              candidates: Sequence[int]) -> int:
    """Phase-2 estimator: ML localisation from first-arrival times."""
    if not obs:
        return -1
    arrivals = first_arrivals(obs)
    cand = list(candidates)
    if not cand:
        return -1
    if len(arrivals) == 1:
        o = next(iter(arrivals.values()))
        return o.frm if o.frm in set(cand) else _nearest(model, o.frm, cand)
    chi = _chi2(model, arrivals, cand)
    viol = _predecessor_violations(model, arrivals, cand)
    score = chi + 1e7 * viol
    return int(cand[int(np.argmin(score))])


def _nearest(model: NetworkModel, u: int, cand: Sequence[int]) -> int:
    return int(min(cand, key=lambda v: model.dist[v][u]))


def _posterior_time(model: NetworkModel, arrivals: Dict[int, Obs],
                    cand: Sequence[int], pred_penalty: float = 3.0) -> np.ndarray:
    """P(origin of the observed diffusion = v), from arrival times only."""
    if len(arrivals) < 2:
        return np.ones(len(cand)) / max(len(cand), 1)
    chi = _chi2(model, arrivals, cand)
    chi = np.where(np.isfinite(chi), chi, 1e12)
    ll = -0.5 * chi
    if pred_penalty:
        ll = ll - pred_penalty * _predecessor_violations(model, arrivals, cand)
    ll -= ll.max()
    w = np.exp(ll)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(cand)) / len(cand)


def dandelion_ml(obs: Sequence[Obs], model: NetworkModel, p: float,
                 candidates: Sequence[int], fluff_with_stem: bool = False,
                 fluff_weight: float = 1.0) -> int:
    """Phase-4 estimator against Dandelion.

    Evidence 1 (strong): a spy received the packet while it was still in the
    *stem*.  The edge it arrived on identifies one point of the stem path; the
    reverse random-walk kernel then gives P(source = v).

    Evidence 2: the fluff wave.  Its origin is localised with the phase-2
    timing ML, and the stem kernel maps that distribution back onto candidate
    sources.
    """
    if not obs:
        return -1
    cand = list(candidates)
    if not cand:
        return -1
    stem_k, fluff_k, eidx = model.kernels(p)
    logp = np.zeros(len(cand))
    used = False

    stem_obs = [o for o in obs if o.state == "S"]
    if stem_obs:
        o = min(stem_obs, key=lambda x: x.t)
        e = eidx.get((o.frm, o.spy))
        if e is not None:
            col = stem_k[:, e][cand]
            logp = logp + np.log(np.maximum(col, 1e-300))
            used = True

    if not used or fluff_with_stem:
        fluff_arr = first_arrivals([o for o in obs if o.state == "F"])
        if len(fluff_arr) >= 2:
            allv = list(range(model.n))
            post_f = _posterior_time(model, fluff_arr, allv)
            mix = fluff_k[cand, :] @ post_f      # P(v) marginalised over fluff origin
            logp = logp + fluff_weight * np.log(np.maximum(mix, 1e-300))
            used = True
        elif len(fluff_arr) == 1 and not used:
            o = next(iter(fluff_arr.values()))
            mix = fluff_k[cand, :][:, o.frm] + 1e-12
            logp = logp + np.log(mix)
            used = True

    if not used:
        return first_spy(obs, model.spies, model.n)
    return int(cand[int(np.argmax(logp))])


# --------------------------------------------------------------------------
# spy placement
# --------------------------------------------------------------------------
def betweenness(model: NetworkModel) -> np.ndarray:
    """Delay-weighted shortest-path betweenness (small graphs => brute force)."""
    n = model.n
    bc = np.zeros(n)
    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            cur = t
            while cur != s and model.prev[s][cur] >= 0:
                cur = int(model.prev[s][cur])
                if cur != s:
                    bc[cur] += 1
    return bc


def select_spies(topo: Topology, k: int, strategy: str = "kcenter",
                 seed: int = 0) -> List[int]:
    """Choose the k bribed nodes.

    * ``kcenter``     - greedy max-min (delay) spread: spies far apart give the
                        best triangulation geometry.
    * ``betweenness`` - the nodes most traffic flows through: they observe the
                        earliest.
    * ``hybrid``      - greedy: maximise betweenness while keeping a minimum
                        separation (used by default, see the report).
    * ``degree`` / ``random``.
    """
    model = NetworkModel(topo)
    n = topo.n
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)
    if strategy == "random":
        return sorted(rng.choice(n, size=min(k, n), replace=False).tolist())
    if strategy == "degree":
        order = sorted(range(n), key=lambda v: (-model.deg[v], v))
        return sorted(order[:k])
    if strategy == "betweenness":
        bc = betweenness(model)
        order = sorted(range(n), key=lambda v: (-bc[v], v))
        return sorted(order[:k])
    if strategy == "kcenter":
        start = int(np.argmax(model.dist.sum(axis=1)))
        chosen = [start]
        while len(chosen) < min(k, n):
            best, bestd = None, -1.0
            for v in range(n):
                if v in chosen:
                    continue
                d = min(model.dist[v][c] for c in chosen)
                if d > bestd:
                    best, bestd = v, d
            chosen.append(int(best))
        return sorted(chosen)
    if strategy == "hybrid":
        bc = betweenness(model)
        bc = bc / (bc.max() + 1e-9)
        chosen: List[int] = []
        while len(chosen) < min(k, n):
            best, bests = None, -1e18
            for v in range(n):
                if v in chosen:
                    continue
                sep = (min(model.dist[v][c] for c in chosen) / model.dist.max()
                       if chosen else 1.0)
                s = bc[v] + 1.2 * sep
                if s > bests:
                    best, bests = v, s
            chosen.append(int(best))
        return sorted(chosen)
    raise ValueError("unknown strategy %r" % strategy)
