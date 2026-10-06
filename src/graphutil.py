"""Small self-contained graph utilities (no third-party graph library).

Everything the project needs from graph theory lives here: connectivity,
bridges, Dijkstra over link delays and shortest-path matrices.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Edge = Tuple[int, int]


def adjacency(n: int, edges: Iterable[Edge]) -> List[Set[int]]:
    adj: List[Set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def is_connected(n: int, edges: Iterable[Edge]) -> bool:
    adj = adjacency(n, edges)
    seen = {0}
    dq = deque([0])
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                dq.append(w)
    return len(seen) == n


def components(n: int, edges: Iterable[Edge]) -> List[List[int]]:
    adj = adjacency(n, edges)
    seen: Set[int] = set()
    out: List[List[int]] = []
    for s in range(n):
        if s in seen:
            continue
        comp = [s]
        seen.add(s)
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    comp.append(w)
                    dq.append(w)
        out.append(comp)
    return out


def bridges(n: int, edges: Sequence[Edge]) -> Set[Edge]:
    """Return the set of bridges (cut edges), each as an ordered tuple (min, max).

    Iterative Tarjan low-link, so deep graphs cannot blow the Python stack.
    """
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    disc = [-1] * n
    low = [0] * n
    timer = 0
    out: Set[Edge] = set()

    for root in range(n):
        if disc[root] != -1:
            continue
        stack: List[Tuple[int, int, int]] = [(root, -1, 0)]  # node, in-edge id, child ptr
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            u, pe, ptr = stack.pop()
            if ptr < len(adj[u]):
                stack.append((u, pe, ptr + 1))
                v, eid = adj[u][ptr]
                if eid == pe:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, eid, 0))
                else:
                    low[u] = min(low[u], disc[v])
            else:
                if pe != -1:
                    p = edges[pe][0] if edges[pe][1] == u else edges[pe][1]
                    low[p] = min(low[p], low[u])
                    if low[u] > disc[p]:
                        out.add((min(p, u), max(p, u)))
    return out


def dijkstra(n: int, wadj: List[List[Tuple[int, float]]], src: int) -> Tuple[List[float], List[int]]:
    """Single-source shortest path. Returns (distance, predecessor)."""
    dist = [math.inf] * n
    prev = [-1] * n
    dist[src] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u] + 1e-12:
            continue
        for v, w in wadj[u]:
            nd = d + w
            if nd < dist[v] - 1e-12:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def weighted_adjacency(n: int, edges: Sequence[Edge], weights: Sequence[float],
                       directed_extra: Dict[int, float] | None = None
                       ) -> List[List[Tuple[int, float]]]:
    """Build a weighted adjacency list.

    ``directed_extra`` maps a node id to an additional cost charged on every
    *outgoing* link of that node.  It is what lets the adversary of phase 5
    model the artificial delay its own spies inject.
    """
    extra = directed_extra or {}
    wadj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for (u, v), w in zip(edges, weights):
        wadj[u].append((v, w + extra.get(u, 0.0)))
        wadj[v].append((u, w + extra.get(v, 0.0)))
    return wadj


def all_pairs(n: int, wadj: List[List[Tuple[int, float]]]) -> Tuple[List[List[float]], List[List[int]]]:
    dist: List[List[float]] = []
    prev: List[List[int]] = []
    for s in range(n):
        d, p = dijkstra(n, wadj, s)
        dist.append(d)
        prev.append(p)
    return dist, prev


def hop_distance(n: int, edges: Sequence[Edge], src: int) -> List[int]:
    adj = adjacency(n, edges)
    dist = [-1] * n
    dist[src] = 0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist
