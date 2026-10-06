"""Offline analysis of a simulation run: metrics and adversary evaluation.

Important property exploited here: in phases 1-4 a spy behaves *exactly* like
an honest node (it may not drop, block or delay anything).  The set of bribed
nodes therefore has **no influence on the dynamics of the network**, and the
adversary's view for any candidate spy set is simply the subset of the
reference log observed at those nodes.  This lets us sweep the number of spies
and the placement strategy over a single recorded run - and it is exact, not an
approximation.  Phase 5 breaks that property (spies delay), so phase-5 runs are
simulated with their spy set fixed in advance.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adversary import (NetworkModel, Obs, dandelion_ml, first_spy, timing_ml)
from .topology import Topology


@dataclass
class Run:
    outdir: str
    meta: dict
    topo: Topology
    src: Dict[str, int]
    t_create: Dict[str, float]
    recv: Dict[str, List[Obs]]              # pid -> receptions (any node)
    first_recv: Dict[str, Dict[int, float]]
    counters: Dict[str, int]

    @property
    def n(self) -> int:
        return self.topo.n

    @property
    def pids(self) -> List[str]:
        return sorted(self.src)


def load_run(outdir: str) -> Run:
    meta = json.load(open(os.path.join(outdir, "meta.json")))
    topo = Topology.from_json(os.path.join(outdir, "topology.json"))
    src = {p["pid"]: p["src"] for p in meta["packets"]}
    t_create: Dict[str, float] = {}
    recv: Dict[str, List[Obs]] = {pid: [] for pid in src}
    first: Dict[str, Dict[int, float]] = {pid: {} for pid in src}
    counters: Dict[str, int] = {}
    with open(os.path.join(outdir, "events.jsonl")) as fh:
        for line in fh:
            e = json.loads(line)
            ev = e["ev"]
            counters[ev] = counters.get(ev, 0) + 1
            pid = e.get("pid")
            if pid is None or pid not in src:
                continue
            if ev == "create":
                t_create[pid] = e["t"]
                first[pid][e["node"]] = e["t"]
            elif ev == "recv":
                recv[pid].append(Obs(spy=e["node"], frm=e["frm"],
                                     state=e.get("state", "F"), t=e["t"]))
                if e["node"] not in first[pid]:
                    first[pid][e["node"]] = e["t"]
    return Run(outdir, meta, topo, src, t_create, recv, first, counters)


# --------------------------------------------------------------------------
# network-side metrics
# --------------------------------------------------------------------------
def coverage_time(run: Run, pid: str, frac: float = 0.8) -> Optional[float]:
    """Seconds until the packet has been received by ``frac`` of all nodes."""
    need = math.ceil(frac * run.n)
    ts = sorted(run.first_recv[pid].values())
    if len(ts) < need or pid not in run.t_create:
        return None
    return ts[need - 1] - run.t_create[pid]


def coverage_fraction(run: Run, pid: str) -> float:
    return len(run.first_recv[pid]) / run.n


def network_metrics(run: Run, frac: float = 0.8) -> Dict[str, float]:
    tv = [coverage_time(run, pid, frac) for pid in run.pids]
    tv = [x for x in tv if x is not None]
    cov = [coverage_fraction(run, pid) for pid in run.pids]
    sends = run.counters.get("send", 0)
    return {
        "T80_mean": float(np.mean(tv)) if tv else float("nan"),
        "T80_median": float(np.median(tv)) if tv else float("nan"),
        "T80_std": float(np.std(tv, ddof=1)) if len(tv) > 1 else 0.0,
        "T80_p95": float(np.percentile(tv, 95)) if tv else float("nan"),
        "coverage_mean": float(np.mean(cov)) if cov else 0.0,
        "full_coverage_ratio": float(np.mean([c >= 0.999 for c in cov])) if cov else 0.0,
        "packets_measured": len(tv),
        "messages_per_packet": sends / max(len(run.pids), 1),
    }


# --------------------------------------------------------------------------
# adversary-side evaluation
# --------------------------------------------------------------------------
def spy_view(run: Run, pid: str, spies: Sequence[int]) -> List[Obs]:
    S = set(spies)
    return sorted((o for o in run.recv[pid] if o.spy in S), key=lambda o: o.t)


def evaluate_attack(run: Run, spies: Sequence[int], estimator: str,
                    model: NetworkModel, p: float = 0.9) -> Dict[str, float]:
    S = set(spies)
    model.spies = list(spies)
    cand = [v for v in range(run.n) if v not in S]
    hits = 0
    total = 0
    blind = 0
    guesses: Dict[str, int] = {}
    for pid in run.pids:
        true_src = run.src[pid]
        if true_src in S:                 # sources are honest nodes by definition
            continue
        total += 1
        obs = spy_view(run, pid, spies)
        if not obs:
            blind += 1
            guesses[pid] = -1
            continue
        if estimator == "first_spy":
            g = first_spy(obs, spies, run.n)
        elif estimator == "timing_ml":
            g = timing_ml(obs, model, cand)
        elif estimator == "dandelion_ml":
            g = dandelion_ml(obs, model, p, cand)
        else:
            raise ValueError(estimator)
        guesses[pid] = g
        if g == true_src:
            hits += 1
    acc = hits / total if total else 0.0
    return {
        "estimator": estimator,
        "n_spies": len(spies),
        "packets": total,
        "hits": hits,
        "accuracy": acc,
        "blind_packets": blind,
        "score_adv": acc / len(spies) if spies else 0.0,
        "_guesses": guesses,
    }


def scores(run: Run, attack: Dict[str, float], frac: float = 0.8) -> Dict[str, float]:
    net = network_metrics(run, frac)
    t80 = net["T80_mean"]
    det = attack["accuracy"]
    return {
        **net,
        "accuracy": det,
        "n_spies": attack["n_spies"],
        "score_adv": attack["score_adv"],
        "score_honest": (1.0 / t80) * (1.0 - det) if t80 and not math.isnan(t80) else float("nan"),
    }


def greedy_spy_selection(runs: Sequence[Run], kmax: int, estimator: str,
                         model: NetworkModel, p: float = 0.9,
                         forbid: Sequence[int] = ()) -> List[Dict]:
    """Forward-greedy placement: repeatedly add the node that helps most.

    Accuracy is averaged over the calibration runs given in ``runs``.  Returns
    one record per k = 1..kmax with the chosen set, its accuracy and
    Score_adv = Accuracy / k.
    """
    n = runs[0].n
    chosen: List[int] = []
    out: List[Dict] = []
    banned = set(forbid)
    for k in range(1, kmax + 1):
        best, best_acc = None, -1.0
        for v in range(n):
            if v in chosen or v in banned:
                continue
            trial = sorted(chosen + [v])
            acc = float(np.mean([evaluate_attack(r, trial, estimator, model, p)["accuracy"]
                                 for r in runs]))
            if acc > best_acc:
                best, best_acc = v, acc
        chosen = sorted(chosen + [best])          # type: ignore[list-item]
        out.append({"k": k, "spies": list(chosen), "accuracy": best_acc,
                    "score_adv": best_acc / k})
    return out


def evaluate_spy_sets(runs: Sequence[Run], sets: Dict[str, List[int]], estimator: str,
                      model: NetworkModel, p: float = 0.9) -> List[Dict]:
    out = []
    for name, spies in sets.items():
        accs = [evaluate_attack(r, spies, estimator, model, p)["accuracy"] for r in runs]
        out.append({"name": name, "spies": spies, "k": len(spies),
                    "accuracy": float(np.mean(accs)),
                    "score_adv": float(np.mean(accs)) / max(len(spies), 1)})
    return out


def summarise(values: Sequence[float]) -> Dict[str, float]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "n": len(v),
    }
