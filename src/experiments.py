"""Experiment driver: runs every phase of the project and stores the results.

    python -m src.experiments all            # topology + calibration + phases + analysis
    python -m src.experiments calibrate      # spy-count / spy-placement study only
    python -m src.experiments simulate       # the 5 phases (40 runs)
    python -m src.experiments analyse        # metrics, tables and figures

Every simulation writes its own directory under ``results/runs`` and is
skipped when it already exists, so the campaign can be interrupted and
resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np

from .adversary import NetworkModel, select_spies
from .analysis import (Run, evaluate_attack, evaluate_spy_sets,
                       greedy_spy_selection, load_run, network_metrics,
                       scores, summarise)
from .controller import RunConfig, run_once
from .topology import Topology, generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

TOPO_SEED = 7
SEEDS = [1, 2, 3, 4, 5]           # five independent scenario seeds
P_VALUES = [0.9, 0.5, 0.1]
N_PACKETS = 200
WINDOW = 12.0
DRAIN = 6.0
CAL_SEEDS = [101, 102]
MAX_BRIBE_FRACTION = 0.30


def topo_path() -> str:
    return os.path.join(RESULTS, "topology.json")


def get_topology() -> Topology:
    os.makedirs(RESULTS, exist_ok=True)
    if os.path.exists(topo_path()):
        return Topology.from_json(topo_path())
    topo = generate(TOPO_SEED)
    topo.to_json(topo_path())
    return topo


def _run(topo: Topology, cfg: RunConfig, outdir: str) -> str:
    if os.path.exists(os.path.join(outdir, "events.jsonl")):
        return outdir
    t = time.time()
    run_once(topo, cfg, outdir)
    print("   %-34s  %5.1fs" % (os.path.basename(outdir), time.time() - t), flush=True)
    return outdir


# --------------------------------------------------------------------------
# stage 1 - calibration: how many spies, and where?
# --------------------------------------------------------------------------
def calibrate(topo: Topology) -> Dict:
    caldir = os.path.join(RESULTS, "calib")
    os.makedirs(caldir, exist_ok=True)
    runs: List[Run] = []
    print("[calibration] flooding runs")
    for s in CAL_SEEDS:
        d = os.path.join(caldir, "flood_s%d" % s)
        _run(topo, RunConfig(tag="cal", mode="flood", n_packets=100,
                             inject_window_s=8.0, drain_s=5.0, seed=s,
                             sources_from="all"), d)
        runs.append(load_run(d))

    kmax = int(np.floor(MAX_BRIBE_FRACTION * topo.n))
    model = NetworkModel(topo)

    greedy_ml = greedy_spy_selection(runs, kmax, "timing_ml", model)
    greedy_fs = greedy_spy_selection(runs, kmax, "first_spy", model)

    # ordered list of spies: the order in which the greedy search added them
    order: List[int] = []
    for rec in greedy_ml:
        for v in rec["spies"]:
            if v not in order:
                order.append(v)

    heur = {}
    for st in ("kcenter", "degree", "betweenness", "hybrid", "random"):
        for k in range(1, kmax + 1):
            heur["%s-%d" % (st, k)] = select_spies(topo, k, st, seed=13)
    heur_eval = evaluate_spy_sets(runs, heur, "timing_ml", model)

    best = max(greedy_ml, key=lambda r: r["score_adv"])
    out = {
        "kmax": kmax,
        "greedy_timing_ml": greedy_ml,
        "greedy_first_spy": greedy_fs,
        "spy_order": order,
        "heuristics": sorted(heur_eval, key=lambda r: -r["score_adv"]),
        "optimal_k": best["k"],
        "optimal_spies": best["spies"],
        "optimal_score_adv": best["score_adv"],
        "calibration_seeds": CAL_SEEDS,
    }
    with open(os.path.join(caldir, "calibration.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("[calibration] optimal k = %d  (Score_adv = %.4f), spy order = %s"
          % (out["optimal_k"], out["optimal_score_adv"], order))
    return out


# --------------------------------------------------------------------------
# stage 2 - the five phases
# --------------------------------------------------------------------------
def simulate(topo: Topology, spies: Sequence[int]) -> Dict[str, str]:
    runs: Dict[str, str] = {}
    base = os.path.join(RESULTS, "runs")
    os.makedirs(base, exist_ok=True)
    spies = list(spies)

    print("[phase 1/2] base diffusion (flooding)")
    for s in SEEDS:
        tag = "flood_seed%d" % s
        runs[tag] = _run(topo, RunConfig(tag=tag, mode="flood", n_packets=N_PACKETS,
                                         inject_window_s=WINDOW, drain_s=DRAIN,
                                         seed=s, spies=spies), os.path.join(base, tag))

    print("[phase 3/4] Dandelion")
    for p in P_VALUES:
        for s in SEEDS:
            tag = "dand_p%.1f_seed%d" % (p, s)
            runs[tag] = _run(topo, RunConfig(tag=tag, mode="dandelion", p=p,
                                             n_packets=N_PACKETS, inject_window_s=WINDOW,
                                             drain_s=DRAIN, seed=s, spies=spies),
                             os.path.join(base, tag))

    print("[phase 5] spies delay their relays (max legal delay = 1 base delay)")
    for p in P_VALUES:
        for s in SEEDS:
            tag = "p5dand_p%.1f_seed%d" % (p, s)
            runs[tag] = _run(topo, RunConfig(tag=tag, mode="dandelion", p=p,
                                             n_packets=N_PACKETS, inject_window_s=WINDOW,
                                             drain_s=DRAIN + 3, seed=s, spies=spies,
                                             spy_delay_mode="max"),
                             os.path.join(base, tag))
    for s in SEEDS:
        tag = "p5flood_seed%d" % s
        runs[tag] = _run(topo, RunConfig(tag=tag, mode="flood", n_packets=N_PACKETS,
                                         inject_window_s=WINDOW, drain_s=DRAIN + 3,
                                         seed=s, spies=spies, spy_delay_mode="max"),
                         os.path.join(base, tag))
    return runs


# --------------------------------------------------------------------------
# stage 3 - analysis
# --------------------------------------------------------------------------
def _agg(records: Sequence[Dict], key: str) -> Dict[str, float]:
    return summarise([r[key] for r in records])


def analyse(topo: Topology, calib: Dict) -> Dict:
    base = os.path.join(RESULTS, "runs")
    order: List[int] = calib["spy_order"]
    kmax = calib["kmax"]
    model_plain = NetworkModel(topo)
    model_delay = NetworkModel(topo, order, spy_delay_factor=2.0)
    spies_full = sorted(order)

    res: Dict = {"topology": topo.validation, "n_nodes": topo.n,
                 "spy_order": order, "kmax": kmax, "seeds": SEEDS,
                 "p_values": P_VALUES, "n_packets": N_PACKETS}

    # ---------------- phase 1: pure diffusion --------------------------------
    flood_runs = [load_run(os.path.join(base, "flood_seed%d" % s)) for s in SEEDS]
    res["phase1"] = {
        "per_seed": [network_metrics(r) for r in flood_runs],
        "T80": _agg([network_metrics(r) for r in flood_runs], "T80_mean"),
        "messages_per_packet": _agg([network_metrics(r) for r in flood_runs],
                                    "messages_per_packet"),
        "coverage": _agg([network_metrics(r) for r in flood_runs], "coverage_mean"),
    }

    # ---------------- phase 2: attack on flooding, k sweep -------------------
    sweep = []
    for k in range(1, kmax + 1):
        sp = sorted(order[:k])
        rec = {"k": k, "spies": sp}
        for est in ("first_spy", "timing_ml"):
            accs = [evaluate_attack(r, sp, est, model_plain)["accuracy"] for r in flood_runs]
            rec[est] = summarise(accs)
            rec[est + "_score_adv"] = summarise([a / k for a in accs])
        sweep.append(rec)
    best_k = max(sweep, key=lambda r: r["timing_ml_score_adv"]["mean"])["k"]
    res["phase2"] = {"sweep": sweep, "optimal_k": best_k,
                     "optimal_spies": sorted(order[:best_k])}

    # ---------------- phase 3/4: Dandelion ----------------------------------
    res["phase3"] = {}
    res["phase4"] = {}
    for p in P_VALUES:
        rs = [load_run(os.path.join(base, "dand_p%.1f_seed%d" % (p, s))) for s in SEEDS]
        nm = [network_metrics(r) for r in rs]
        res["phase3"]["p=%.1f" % p] = {
            "per_seed": nm,
            "T80": _agg(nm, "T80_mean"),
            "T80_median": _agg(nm, "T80_median"),
            "messages_per_packet": _agg(nm, "messages_per_packet"),
            "coverage": _agg(nm, "coverage_mean"),
            "stem_hops": summarise([_stem_hops(r) for r in rs]),
        }
        block = []
        for k in range(1, kmax + 1):
            sp = sorted(order[:k])
            rec = {"k": k, "spies": sp}
            for est in ("first_spy", "timing_ml", "dandelion_ml"):
                accs = [evaluate_attack(r, sp, est, model_plain, p)["accuracy"] for r in rs]
                rec[est] = summarise(accs)
                rec[est + "_score_adv"] = summarise([a / k for a in accs])
            block.append(rec)
        res["phase4"]["p=%.1f" % p] = {"sweep": block}

    # ---- is the flooding-optimised placement still good against Dandelion? --
    cal = [load_run(os.path.join(base, "dand_p0.5_seed%d" % s)) for s in SEEDS[:2]]
    hold = [load_run(os.path.join(base, "dand_p0.5_seed%d" % s)) for s in SEEDS[2:]]
    greedy_d = greedy_spy_selection(cal, kmax, "dandelion_ml", model_plain, 0.5)

    def _holdout(sp):
        accs = [evaluate_attack(r, sp, "dandelion_ml", model_plain, 0.5)["accuracy"]
                for r in hold]
        return summarise(accs)

    rows = []
    for k in range(1, kmax + 1):
        sp_flood = sorted(order[:k])
        sp_dand = greedy_d[k - 1]["spies"]
        rows.append({"k": k,
                     "flood_spies": sp_flood, "flood_holdout": _holdout(sp_flood),
                     "dand_spies": sp_dand, "dand_holdout": _holdout(sp_dand)})
    res["phase4_placement"] = {
        "rows": rows,
        "selection_runs": ["dand_p0.5_seed%d" % s for s in SEEDS[:2]],
        "holdout_runs": ["dand_p0.5_seed%d" % s for s in SEEDS[2:]],
    }

    # ---------------- phase 5: spies delay ----------------------------------
    res["phase5"] = {}
    for p in P_VALUES:
        rs = [load_run(os.path.join(base, "p5dand_p%.1f_seed%d" % (p, s))) for s in SEEDS]
        nm = [network_metrics(r) for r in rs]
        entry = {"per_seed": nm, "T80": _agg(nm, "T80_mean"),
                 "messages_per_packet": _agg(nm, "messages_per_packet")}
        for name, mdl in (("naive", model_plain), ("compensated", model_delay)):
            for est in ("first_spy", "dandelion_ml"):
                accs = [evaluate_attack(r, spies_full, est, mdl, p)["accuracy"] for r in rs]
                entry["%s_%s" % (est, name)] = summarise(accs)
        res["phase5"]["p=%.1f" % p] = entry
    rs = [load_run(os.path.join(base, "p5flood_seed%d" % s)) for s in SEEDS]
    nm = [network_metrics(r) for r in rs]
    entry = {"per_seed": nm, "T80": _agg(nm, "T80_mean")}
    for name, mdl in (("naive", model_plain), ("compensated", model_delay)):
        for est in ("first_spy", "timing_ml"):
            accs = [evaluate_attack(r, spies_full, est, mdl)["accuracy"] for r in rs]
            entry["%s_%s" % (est, name)] = summarise(accs)
    res["phase5"]["flood"] = entry

    # ---------------- scores at the two reference operating points -----------
    res["scores"] = _score_table(res, best_k, kmax)
    with open(os.path.join(RESULTS, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def _stem_hops(run: Run) -> float:
    """Average number of stem receptions per packet (= length of the stem path)."""
    tot = 0
    for pid in run.pids:
        tot += sum(1 for o in run.recv[pid] if o.state == "S")
    return tot / max(len(run.pids), 1)


def _score_table(res: Dict, best_k: int, kmax: int) -> List[Dict]:
    rows = []
    for k in (best_k, kmax):
        row = {"k": k, "phase": "1+2 (flooding)",
               "accuracy": res["phase2"]["sweep"][k - 1]["timing_ml"]["mean"],
               "accuracy_baseline": res["phase2"]["sweep"][k - 1]["first_spy"]["mean"],
               "T80": res["phase1"]["T80"]["mean"]}
        row["score_adv"] = row["accuracy"] / k
        row["score_honest"] = (1.0 / row["T80"]) * (1.0 - row["accuracy"])
        rows.append(row)
        for p in P_VALUES:
            pk = "p=%.1f" % p
            acc = res["phase4"][pk]["sweep"][k - 1]["dandelion_ml"]["mean"]
            accb = res["phase4"][pk]["sweep"][k - 1]["first_spy"]["mean"]
            t80 = res["phase3"][pk]["T80"]["mean"]
            rows.append({"k": k, "phase": "3+4 (Dandelion p=%.1f)" % p,
                         "accuracy": acc, "accuracy_baseline": accb, "T80": t80,
                         "score_adv": acc / k,
                         "score_honest": (1.0 / t80) * (1.0 - acc)})
    for p in P_VALUES:
        pk = "p=%.1f" % p
        acc = res["phase5"][pk]["dandelion_ml_compensated"]["mean"]
        t80 = res["phase5"][pk]["T80"]["mean"]
        rows.append({"k": kmax, "phase": "5 (Dandelion p=%.1f + spy delay)" % p,
                     "accuracy": acc,
                     "accuracy_baseline": res["phase5"][pk]["first_spy_naive"]["mean"],
                     "T80": t80, "score_adv": acc / kmax,
                     "score_honest": (1.0 / t80) * (1.0 - acc)})
    return rows


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all", "topology", "calibrate", "simulate", "analyse"])
    args = ap.parse_args()

    topo = get_topology()
    print("topology: n=%d, clusters=%d, edges=%d, deg %d..%d, bridges=%d"
          % (topo.n, topo.validation["n_clusters"], topo.validation["n_edges"],
             topo.validation["degree_min"], topo.validation["degree_max"],
             topo.validation["n_bridges"]))
    if args.stage == "topology":
        return

    calib_path = os.path.join(RESULTS, "calib", "calibration.json")
    if args.stage in ("all", "calibrate") or not os.path.exists(calib_path):
        calib = calibrate(topo)
    else:
        calib = json.load(open(calib_path))
    if args.stage == "calibrate":
        return

    if args.stage in ("all", "simulate"):
        simulate(topo, sorted(calib["spy_order"]))
    if args.stage == "simulate":
        return

    res = analyse(topo, calib)
    print("\n== Score table ==")
    for row in res["scores"]:
        print("k=%d %-34s acc=%.3f (base %.3f)  T80=%.3fs  Score_adv=%.4f  Score_honest=%.4f"
              % (row["k"], row["phase"], row["accuracy"], row["accuracy_baseline"],
                 row["T80"], row["score_adv"], row["score_honest"]))


if __name__ == "__main__":
    main()
