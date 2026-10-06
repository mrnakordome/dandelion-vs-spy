"""Spatial topology generation for the Dandelion-vs-Spy simulator.

Requirements implemented here (part 1.2 of the assignment):

* nodes are dropped on a 1000 x 1000 plane, 20..30 of them;
* the plane contains 4..6 *clusters* where node density is clearly higher,
  the remaining nodes are scattered;
* the final graph is connected, every node has degree in [2, 4], and every
  cluster reaches the rest of the network through at least two independent
  links (we enforce the stronger property "the graph has no bridge", which
  implies at least two edge-disjoint paths between any two nodes);
* the base delay of a link is 1 ms per unit of Euclidean distance; the jitter
  U(-20%, +20%) is applied per transmission, at run time, inside the node.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from . import graphutil as G

PLANE = 1000.0
DELAY_MS_PER_UNIT = 1.0
MIN_DEG = 2
MAX_DEG = 4


@dataclass
class TopoParams:
    n_nodes_range: Tuple[int, int] = (20, 30)
    n_clusters_range: Tuple[int, int] = (4, 6)
    cluster_share: float = 0.8          # fraction of nodes living inside clusters
    cluster_sigma: float = 55.0         # std-dev of the Gaussian blob (units)
    center_margin: float = 150.0        # keep cluster centres away from the border
    center_min_sep: float = 260.0       # minimum distance between two centres
    scatter_min_sep: float = 90.0       # scattered nodes stay away from centres
    extra_cross_links: int = 2          # random inter-cluster links beyond the ring
    max_attempts: int = 200


@dataclass
class Topology:
    n: int
    seed: int
    pos: List[Tuple[float, float]]
    cluster_of: List[int]                # -1 => scattered node
    edges: List[Tuple[int, int]]
    base_delay_ms: List[float]
    params: Dict = field(default_factory=dict)
    validation: Dict = field(default_factory=dict)

    # ---------------------------------------------------------------- helpers
    @property
    def n_clusters(self) -> int:
        return len({c for c in self.cluster_of if c >= 0})

    def adjacency(self) -> List[Set[int]]:
        return G.adjacency(self.n, self.edges)

    def degree(self) -> List[int]:
        return [len(a) for a in self.adjacency()]

    def delay_of(self, u: int, v: int) -> float:
        for (a, b), d in zip(self.edges, self.base_delay_ms):
            if (a, b) == (u, v) or (a, b) == (v, u):
                return d
        raise KeyError((u, v))

    def delay_map(self) -> Dict[Tuple[int, int], float]:
        m: Dict[Tuple[int, int], float] = {}
        for (a, b), d in zip(self.edges, self.base_delay_ms):
            m[(a, b)] = d
            m[(b, a)] = d
        return m

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=1)

    @staticmethod
    def from_json(path: str) -> "Topology":
        with open(path) as fh:
            raw = json.load(fh)
        raw["pos"] = [tuple(p) for p in raw["pos"]]
        raw["edges"] = [tuple(e) for e in raw["edges"]]
        return Topology(**raw)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _sample_centers(rng: np.random.Generator, k: int, pr: TopoParams) -> np.ndarray:
    lo, hi = pr.center_margin, PLANE - pr.center_margin
    centers: List[np.ndarray] = []
    guard = 0
    while len(centers) < k:
        guard += 1
        if guard > 20000:
            raise RuntimeError("cannot place cluster centres")
        c = rng.uniform(lo, hi, size=2)
        if all(np.linalg.norm(c - o) >= pr.center_min_sep for o in centers):
            centers.append(c)
    return np.array(centers)


def _sample_positions(rng: np.random.Generator, n: int, k: int, pr: TopoParams):
    centers = _sample_centers(rng, k, pr)
    n_in = int(round(pr.cluster_share * n))
    # spread the clustered nodes over the k clusters, at least 3 each
    base = [3] * k
    left = n_in - 3 * k
    for i in range(max(left, 0)):
        base[i % k] += 1
    n_in = sum(base)

    pos: List[np.ndarray] = []
    cluster_of: List[int] = []
    for ci, cnt in enumerate(base):
        placed = 0
        while placed < cnt:
            p = centers[ci] + rng.normal(0.0, pr.cluster_sigma, size=2)
            if not (0.0 <= p[0] <= PLANE and 0.0 <= p[1] <= PLANE):
                continue
            # keep the blob compact: a node must be closest to its own centre
            d = np.linalg.norm(centers - p, axis=1)
            if int(np.argmin(d)) != ci:
                continue
            pos.append(p)
            cluster_of.append(ci)
            placed += 1

    for _ in range(n - n_in):
        while True:
            p = rng.uniform(30.0, PLANE - 30.0, size=2)
            if np.min(np.linalg.norm(centers - p, axis=1)) >= pr.scatter_min_sep:
                pos.append(p)
                cluster_of.append(-1)
                break
    return np.array(pos), cluster_of, centers


def _dist_matrix(pos: np.ndarray) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _mst_edges(idx: Sequence[int], D: np.ndarray) -> List[Tuple[int, int]]:
    """Prim over the sub-set ``idx`` of nodes."""
    idx = list(idx)
    if len(idx) < 2:
        return []
    inside = {idx[0]}
    out = set(idx[1:])
    edges: List[Tuple[int, int]] = []
    while out:
        best = None
        for a in inside:
            for b in out:
                d = D[a, b]
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best  # type: ignore[misc]
        edges.append((min(a, b), max(a, b)))
        inside.add(b)
        out.discard(b)
    return edges


def _add(edges: Set[Tuple[int, int]], u: int, v: int) -> None:
    if u != v:
        edges.add((min(u, v), max(u, v)))


def _build_edges(rng: np.random.Generator, pos: np.ndarray, cluster_of: List[int],
                 pr: TopoParams) -> List[Tuple[int, int]]:
    n = len(pos)
    D = _dist_matrix(pos)
    edges: Set[Tuple[int, int]] = set()

    groups: Dict[int, List[int]] = {}
    for i, c in enumerate(cluster_of):
        groups.setdefault(c, []).append(i)
    clusters = {c: v for c, v in groups.items() if c >= 0}
    scattered = groups.get(-1, [])

    # (1) intra-cluster backbone: Euclidean MST, then raise every member to
    #     degree >= 2 with its nearest not-yet-connected cluster mate.
    for members in clusters.values():
        for e in _mst_edges(members, D):
            _add(edges, *e)
        for u in members:
            deg = sum(1 for e in edges if u in e)
            if deg >= 2:
                continue
            cand = sorted((D[u, v], v) for v in members
                          if v != u and (min(u, v), max(u, v)) not in edges)
            for _, v in cand:
                _add(edges, u, v)
                break

    # (2) scattered nodes hook onto their two nearest neighbours anywhere.
    for u in scattered:
        cand = sorted((D[u, v], v) for v in range(n) if v != u)
        for _, v in cand[:2]:
            _add(edges, u, v)

    # (3) inter-cluster ring: order the cluster centroids by polar angle around
    #     the global centroid and connect consecutive clusters through their
    #     closest pair of nodes.  A ring gives every cluster exactly two
    #     independent links to the rest of the network by construction.
    cids = sorted(clusters)
    if len(cids) >= 2:
        cent = {c: pos[members].mean(axis=0) for c, members in clusters.items()}
        gc = np.mean([cent[c] for c in cids], axis=0)
        order = sorted(cids, key=lambda c: math.atan2(cent[c][1] - gc[1], cent[c][0] - gc[0]))
        used: Dict[int, Set[int]] = {c: set() for c in cids}
        for i in range(len(order)):
            a, b = order[i], order[(i + 1) % len(order)]
            if len(order) == 2 and i == 1:
                pass  # second link between the same two clusters is still wanted
            pairs = sorted((D[u, v], u, v) for u in clusters[a] for v in clusters[b])
            for _, u, v in pairs:
                # prefer endpoints that are not already carrying a ring link,
                # so the two links of a cluster are node-disjoint => independent
                if u in used[a] or v in used[b]:
                    continue
                _add(edges, u, v)
                used[a].add(u)
                used[b].add(v)
                break
            else:
                _, u, v = pairs[0]
                _add(edges, u, v)
                used[a].add(u)
                used[b].add(v)

        # (4) a few extra random cross links (shortest cross pairs) for realism
        cross_pairs = []
        for i, a in enumerate(cids):
            for b in cids[i + 1:]:
                pairs = sorted((D[u, v], u, v) for u in clusters[a] for v in clusters[b])
                cross_pairs.extend(pairs[:2])
        cross_pairs.sort()
        picked = 0
        for _, u, v in cross_pairs:
            if picked >= pr.extra_cross_links:
                break
            if (min(u, v), max(u, v)) in edges:
                continue
            _add(edges, u, v)
            picked += 1

    return sorted(edges)


def _repair(pos: np.ndarray, edges: List[Tuple[int, int]], cluster_of: List[int]
            ) -> List[Tuple[int, int]]:
    """Enforce degree in [2,4], connectivity and bridge-freeness."""
    n = len(pos)
    D = _dist_matrix(pos)
    E: Set[Tuple[int, int]] = set(edges)

    def deg() -> List[int]:
        d = [0] * n
        for u, v in E:
            d[u] += 1
            d[v] += 1
        return d

    # --- trim over-connected nodes (drop the longest removable edge) --------
    for _ in range(4 * n):
        d = deg()
        over = [i for i in range(n) if d[i] > MAX_DEG]
        if not over:
            break
        u = max(over, key=lambda i: d[i])
        cands = sorted(((D[u, v], v) for v in range(n)
                        if (min(u, v), max(u, v)) in E and d[v] > MIN_DEG),
                       reverse=True)
        for _, v in cands:
            trial = set(E)
            trial.discard((min(u, v), max(u, v)))
            if G.is_connected(n, trial) and not G.bridges(n, sorted(trial)):
                E = trial
                break
        else:
            break

    # --- raise under-connected nodes ---------------------------------------
    for _ in range(4 * n):
        d = deg()
        under = [i for i in range(n) if d[i] < MIN_DEG]
        if not under:
            break
        u = under[0]
        cands = sorted((D[u, v], v) for v in range(n)
                       if v != u and (min(u, v), max(u, v)) not in E and d[v] < MAX_DEG)
        if not cands:
            cands = sorted((D[u, v], v) for v in range(n)
                           if v != u and (min(u, v), max(u, v)) not in E)
        _add(E, u, cands[0][1])

    # --- connect components -------------------------------------------------
    for _ in range(n):
        comps = G.components(n, E)
        if len(comps) == 1:
            break
        a, b = comps[0], comps[1]
        _, u, v = min((D[u, v], u, v) for u in a for v in b)
        _add(E, u, v)

    # --- kill bridges by adding the cheapest chord across each one ----------
    for _ in range(3 * n):
        br = G.bridges(n, sorted(E))
        if not br:
            break
        u, v = sorted(br)[0]
        trial = set(E)
        trial.discard((u, v))
        comps = G.components(n, trial)
        side = {}
        for ci, comp in enumerate(comps):
            for x in comp:
                side[x] = ci
        A = [x for x in range(n) if side[x] == side[u]]
        B = [x for x in range(n) if side[x] == side[v]]
        d = deg()
        cand = sorted((D[a, b], a, b) for a in A for b in B
                      if (min(a, b), max(a, b)) not in E)
        placed = False
        for _, a, b in cand:
            if d[a] < MAX_DEG and d[b] < MAX_DEG:
                _add(E, a, b)
                placed = True
                break
        if not placed:
            if not cand:
                break
            _, a, b = cand[0]
            _add(E, a, b)
    return sorted(E)


def validate(topo: "Topology") -> Dict:
    n = topo.n
    edges = topo.edges
    deg = topo.degree()
    br = G.bridges(n, edges)
    clusters: Dict[int, List[int]] = {}
    for i, c in enumerate(topo.cluster_of):
        if c >= 0:
            clusters.setdefault(c, []).append(i)

    cluster_links: Dict[int, int] = {}
    cluster_indep: Dict[int, bool] = {}
    for c, members in clusters.items():
        ms = set(members)
        cross = [(u, v) for (u, v) in edges if (u in ms) != (v in ms)]
        cluster_links[c] = len(cross)
        inner_ends = {u if u in ms else v for (u, v) in cross}
        outer_ends = {v if u in ms else u for (u, v) in cross}
        cluster_indep[c] = (len(cross) >= 2 and len(inner_ends) >= 2
                            and len(outer_ends) >= 2
                            and not any((min(u, v), max(u, v)) in br for (u, v) in cross))

    return {
        "connected": G.is_connected(n, edges),
        "n_bridges": len(br),
        "bridge_free": len(br) == 0,
        "degree_min": min(deg),
        "degree_max": max(deg),
        "degree_ok": min(deg) >= MIN_DEG and max(deg) <= MAX_DEG,
        "degree_mean": round(sum(deg) / n, 3),
        "n_clusters": len(clusters),
        "cluster_sizes": {str(c): len(m) for c, m in sorted(clusters.items())},
        "cluster_cross_links": {str(c): cluster_links[c] for c in sorted(cluster_links)},
        "cluster_two_independent_links": {str(c): bool(cluster_indep[c])
                                          for c in sorted(cluster_indep)},
        "all_clusters_ok": all(cluster_indep.values()) if cluster_indep else False,
        "n_edges": len(edges),
    }


def generate(seed: int, params: TopoParams | None = None,
             n_nodes: int | None = None, n_clusters: int | None = None) -> Topology:
    """Generate a topology that satisfies every structural requirement.

    Regeneration loop: if a draw cannot be repaired into a valid graph we draw
    again with a derived seed (the assignment asks explicitly for this).
    """
    pr = params or TopoParams()
    for attempt in range(pr.max_attempts):
        rng = np.random.default_rng(seed * 10_000 + attempt)
        n = n_nodes if n_nodes is not None else int(rng.integers(pr.n_nodes_range[0],
                                                                 pr.n_nodes_range[1] + 1))
        k = n_clusters if n_clusters is not None else int(rng.integers(pr.n_clusters_range[0],
                                                                       pr.n_clusters_range[1] + 1))
        pos, cluster_of, _ = _sample_positions(rng, n, k, pr)
        edges = _build_edges(rng, pos, cluster_of, pr)
        edges = _repair(pos, edges, cluster_of)

        topo = Topology(
            n=n, seed=seed,
            pos=[(float(p[0]), float(p[1])) for p in pos],
            cluster_of=list(cluster_of),
            edges=[(int(u), int(v)) for u, v in edges],
            base_delay_ms=[float(math.dist(pos[u], pos[v]) * DELAY_MS_PER_UNIT)
                           for u, v in edges],
            params={**asdict(pr), "attempt": attempt, "n_nodes": n, "n_clusters": k},
        )
        val = validate(topo)
        if val["connected"] and val["degree_ok"] and val["all_clusters_ok"] and val["bridge_free"]:
            topo.validation = val
            return topo
    raise RuntimeError("topology generation failed after %d attempts" % pr.max_attempts)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys
    t = generate(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    print(json.dumps(t.validation, indent=2))
