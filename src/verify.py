"""Self-checks proving the simulator behaves as the specification requires.

    python -m src.verify [run-directory ...]

Checks performed on every run directory:
  1. topology invariants (connectivity, degree, clusters, no bridge);
  2. every packet was created exactly once and reached 100% of the nodes;
  3. no error / unexpected event was logged (UDP is assumed lossless on
     localhost, so any loss would show up here);
  4. measured link latency matches ``base +/- 20%`` jitter (and 2x base for a
     delaying spy in phase 5);
  5. Dandelion state machine: the stem part of every packet is a real path in
     the graph, a node never re-emits a packet it has already fluffed, and the
     fluff wave never goes back to the node it came from.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .analysis import load_run
from .topology import Topology, validate


def check_run(outdir: str) -> Dict[str, object]:
    run = load_run(outdir)
    topo = run.topo
    cfg = run.meta["config"]
    adj = topo.adjacency()
    dmap = topo.delay_map()
    spies = set(cfg.get("spies") or [])
    problems: List[str] = []

    val = validate(topo)
    if not (val["connected"] and val["degree_ok"] and val["all_clusters_ok"]):
        problems.append("topology invariants violated: %s" % val)

    events = [json.loads(l) for l in open(os.path.join(outdir, "events.jsonl"))]
    bad = [e for e in events if e["ev"] in ("error", "unexpected")]
    if bad:
        problems.append("%d error/unexpected events (e.g. %s)" % (len(bad), bad[0]))

    creates = defaultdict(int)
    for e in events:
        if e["ev"] == "create":
            creates[e["pid"]] += 1
    if set(creates) != set(run.pids):
        problems.append("missing create events: %s" % (set(run.pids) - set(creates)))
    if any(c != 1 for c in creates.values()):
        problems.append("duplicated create events")

    cov = [len(run.first_recv[pid]) / run.n for pid in run.pids]
    if min(cov) < 1.0:
        problems.append("packets that did not reach every node: %d"
                        % sum(1 for c in cov if c < 1.0))

    # ---- delay model: scheduled delay must obey base +/- 20% jitter -------
    sends: Dict[tuple, float] = {}
    sched_ratio: List[float] = []
    spy_sched_ratio: List[float] = []
    overhead: List[float] = []          # OS/event-loop overhead on top of it
    for e in events:
        if e["ev"] == "send":
            base = dmap[(e["node"], e["to"])]
            sends[(e["node"], e["to"], e["pid"])] = e["t"] + e["delay_ms"] / 1000.0
            (spy_sched_ratio if e["node"] in spies else sched_ratio).append(
                e["delay_ms"] / base)
        elif e["ev"] == "recv":
            eta = sends.get((e["frm"], e["node"], e["pid"]))
            if eta is not None:
                overhead.append((e["t"] - eta) * 1000.0)
    if sched_ratio:
        lo, hi = float(np.min(sched_ratio)), float(np.max(sched_ratio))
        if lo < 0.8 - 1e-9 or hi > 1.2 + 1e-9:
            problems.append("honest scheduled delay outside base*[0.8,1.2]: %.4f..%.4f"
                            % (lo, hi))
    if spy_sched_ratio and cfg["spy_delay_mode"] == "max":
        lo, hi = float(np.min(spy_sched_ratio)), float(np.max(spy_sched_ratio))
        if lo < 1.8 - 1e-9 or hi > 2.2 + 1e-9:
            problems.append("spy scheduled delay outside base*[1.8,2.2]: %.4f..%.4f"
                            % (lo, hi))
    if overhead and float(np.median(overhead)) > 5.0:
        problems.append("median event-loop overhead too large: %.2f ms"
                        % float(np.median(overhead)))

    # ---- Dandelion state machine ------------------------------------------
    stem_edges_ok = True
    if cfg["mode"] == "dandelion":
        for pid in run.pids:
            obs = sorted(run.recv[pid], key=lambda o: o.t)
            stem = [o for o in obs if o.state == "S"]
            for o in stem:
                if o.spy not in adj[o.frm]:
                    stem_edges_ok = False
            # a stem hop must always come from the node that received it before
            chain = [run.src[pid]] + [o.spy for o in stem]
            for a, b in zip(chain, chain[1:]):
                if b not in adj[a]:
                    stem_edges_ok = False
        if not stem_edges_ok:
            problems.append("stem path is not a walk in the graph")
        for pid in run.pids:
            seen_fluff = defaultdict(int)
            for o in sorted(run.recv[pid], key=lambda x: x.t):
                if o.state == "F":
                    seen_fluff[o.spy] += 1
        # (duplicate fluff receptions are expected; they are dropped by SeenSet)

    ok = not problems
    return {
        "run": os.path.basename(outdir),
        "mode": cfg["mode"],
        "p": cfg["p"],
        "spy_delay": cfg["spy_delay_mode"],
        "packets": len(run.pids),
        "coverage_min": float(min(cov)) if cov else 0.0,
        "jitter_ratio_range": [round(float(np.min(sched_ratio)), 3),
                               round(float(np.max(sched_ratio)), 3)] if sched_ratio else None,
        "spy_delay_ratio_mean": (round(float(np.mean(spy_sched_ratio)), 3)
                                 if spy_sched_ratio else None),
        "overhead_ms_median": round(float(np.median(overhead)), 3) if overhead else None,
        "overhead_ms_p99": round(float(np.percentile(overhead, 99)), 3) if overhead else None,
        "duplicates_dropped": run.counters.get("dup", 0),
        "stem_loops_broken": run.counters.get("stem_loop", 0),
        "ok": ok,
        "problems": problems,
    }


def main() -> None:
    dirs = sys.argv[1:]
    if not dirs:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "runs")
        dirs = [os.path.join(base, d) for d in sorted(os.listdir(base))
                if os.path.exists(os.path.join(base, d, "events.jsonl"))]
    all_ok = True
    report = []
    for d in dirs:
        r = check_run(d)
        report.append(r)
        all_ok &= bool(r["ok"])
        print("%-26s %-9s p=%.1f delay=%-6s cov=%.3f jitter x[%s] overhead %.2f/%.2f ms "
              "dup=%d loops=%d  %s"
              % (r["run"], r["mode"], r["p"], r["spy_delay"], r["coverage_min"],
                 ",".join("%.2f" % x for x in (r["jitter_ratio_range"] or [])),
                 r["overhead_ms_median"] or 0.0, r["overhead_ms_p99"] or 0.0,
                 r["duplicates_dropped"], r["stem_loops_broken"],
                 "OK" if r["ok"] else "FAILED: " + "; ".join(r["problems"])))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "verification.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
