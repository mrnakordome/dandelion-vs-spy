"""A small, verbose run - handy for the demonstration video.

    python -m src.demo --mode dandelion --p 0.5 --packets 20 [--spy-delay max]

It starts the real node processes, injects a handful of packets, then prints
for every packet: the true source, the stem path, the spies that saw it, and
what each estimator guessed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile

from .adversary import NetworkModel
from .analysis import (coverage_time, evaluate_attack, load_run, network_metrics,
                       spy_view)
from .controller import RunConfig, run_once
from .experiments import RESULTS, get_topology


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dandelion", choices=["flood", "dandelion"])
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--packets", type=int, default=20)
    ap.add_argument("--spy-delay", default="none", choices=["none", "max", "half", "uniform"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    topo = get_topology()
    calib_path = os.path.join(RESULTS, "calib", "calibration.json")
    spies = sorted(json.load(open(calib_path))["spy_order"]) if os.path.exists(calib_path) \
        else [0, 1, 2]
    print("topology : %d nodes, %d clusters, %d links, degree %d..%d, bridges %d"
          % (topo.n, topo.validation["n_clusters"], len(topo.edges),
             topo.validation["degree_min"], topo.validation["degree_max"],
             topo.validation["n_bridges"]))
    print("spies    : %s   (%.0f%% of the network)"
          % (spies, 100.0 * len(spies) / topo.n))
    print("protocol : %s%s, spy delay = %s\n"
          % (args.mode, "" if args.mode == "flood" else " (p=%.2f)" % args.p,
             args.spy_delay))

    out = tempfile.mkdtemp(prefix="dandelion-demo-")
    cfg = RunConfig(tag="demo", mode=args.mode, p=args.p, n_packets=args.packets,
                    inject_window_s=max(2.0, args.packets * 0.15), drain_s=6.0,
                    seed=args.seed, spies=spies, spy_delay_mode=args.spy_delay)
    run_once(topo, cfg, out)

    run = load_run(out)
    model = NetworkModel(topo, spies,
                         spy_delay_factor=2.0 if args.spy_delay == "max" else 1.0)
    cand = [v for v in range(run.n) if v not in set(spies)]
    est = "dandelion_ml" if args.mode == "dandelion" else "timing_ml"
    res_prop = evaluate_attack(run, spies, est, model, args.p)
    res_base = evaluate_attack(run, spies, "first_spy", model, args.p)

    print("%-7s %-6s %-28s %-22s %-8s %-8s" %
          ("packet", "source", "stem path", "spies that saw it", "baseline", "proposed"))
    for pid in run.pids:
        obs = sorted(run.recv[pid], key=lambda o: o.t)
        stem = [o for o in obs if o.state == "S"]
        path = " -> ".join(str(x) for x in [run.src[pid]] + [o.spy for o in stem])
        seen = ",".join("%d@%.0fms" % (o.spy, (o.t - run.t_create[pid]) * 1000)
                        for o in spy_view(run, pid, spies)[:3])
        g1 = res_base["_guesses"].get(pid, -1)
        g2 = res_prop["_guesses"].get(pid, -1)
        t80 = coverage_time(run, pid)
        print("%-7s %-6d %-28s %-22s %-8s %-8s  T80=%.0fms"
              % (pid, run.src[pid], path[:28], seen[:22],
                 "%d %s" % (g1, "OK" if g1 == run.src[pid] else "x"),
                 "%d %s" % (g2, "OK" if g2 == run.src[pid] else "x"),
                 (t80 or 0) * 1000))

    nm = network_metrics(run)
    print("\nT80%%  mean = %.3f s   coverage = %.0f%%   datagrams/packet = %.1f"
          % (nm["T80_mean"], 100 * nm["coverage_mean"], nm["messages_per_packet"]))
    print("accuracy: first-spy = %.3f | proposed (%s) = %.3f | Score_adv = %.4f"
          % (res_base["accuracy"], est, res_prop["accuracy"], res_prop["score_adv"]))
    print("Score_honest = %.4f" % ((1.0 / nm["T80_mean"]) * (1.0 - res_prop["accuracy"])))
    print("\nrun directory:", out)
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
