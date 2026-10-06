"""Figures and tables for the report (``python -m src.plots``)."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import coverage_time, load_run
from .experiments import P_VALUES, RESULTS, SEEDS, get_topology

FIG = os.path.join(RESULTS, "figures")
TAB = os.path.join(RESULTS, "tables")
CMAP = plt.get_cmap("tab10")


def _save(fig, name: str) -> None:
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=160)
    plt.close(fig)
    print("  figure:", name)


# --------------------------------------------------------------------------
def fig_topology(topo, spies: Sequence[int]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7))
    pos = np.array(topo.pos)
    dmap = topo.delay_map()
    for (u, v) in topo.edges:
        d = dmap[(u, v)]
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]],
                color="0.75", lw=max(0.4, 2.2 - d / 90.0), zorder=1)
    for c in sorted(set(topo.cluster_of)):
        idx = [i for i, cc in enumerate(topo.cluster_of) if cc == c]
        ax.scatter(pos[idx, 0], pos[idx, 1], s=150, zorder=2,
                   color="0.45" if c < 0 else CMAP(c % 10),
                   marker="s" if c < 0 else "o",
                   label="scattered" if c < 0 else "cluster %d" % c)
    ax.scatter(pos[list(spies), 0], pos[list(spies), 1], s=330, facecolors="none",
               edgecolors="crimson", lw=2.4, zorder=3, label="spy (bribed)")
    for i, (x, y) in enumerate(pos):
        ax.annotate(str(i), (x, y), fontsize=7.5, ha="center", va="center",
                    color="white", zorder=4)
    ax.set_xlim(-30, 1030)
    ax.set_ylim(-30, 1030)
    ax.set_aspect("equal")
    ax.set_title("Network topology (%d nodes, %d clusters, %d links)\n"
                 "link width ~ 1/base-delay, base delay = 1 ms per unit"
                 % (topo.n, topo.validation["n_clusters"], len(topo.edges)))
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "fig1_topology.png")


def _errbar(ax, x, recs: List[Dict], key: str, label: str, color, marker="o"):
    m = [r[key]["mean"] for r in recs]
    s = [r[key]["std"] for r in recs]
    ax.errorbar(x, m, yerr=s, label=label, color=color, marker=marker,
                capsize=3, lw=1.8, ms=5)


def fig_phase2(res: Dict) -> None:
    sweep = res["phase2"]["sweep"]
    ks = [r["k"] for r in sweep]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _errbar(axes[0], ks, sweep, "first_spy", "baseline: first-spy", CMAP(1), "s")
    _errbar(axes[0], ks, sweep, "timing_ml", "proposed: timing ML", CMAP(0))
    axes[0].set_xlabel("number of bribed nodes k")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Phase 2 - source-identification accuracy (flooding)")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=.3)
    axes[0].legend()

    _errbar(axes[1], ks, sweep, "first_spy_score_adv", "baseline", CMAP(1), "s")
    _errbar(axes[1], ks, sweep, "timing_ml_score_adv", "proposed", CMAP(0))
    kopt = res["phase2"]["optimal_k"]
    axes[1].axvline(kopt, color="crimson", ls="--", lw=1,
                    label="optimum k*=%d" % kopt)
    axes[1].set_xlabel("number of bribed nodes k")
    axes[1].set_ylabel(r"$Score_{adv}=Accuracy/k$")
    axes[1].set_title("Phase 2 - adversary score")
    axes[1].grid(alpha=.3)
    axes[1].legend()
    _save(fig, "fig2_phase2_sweep.png")


def fig_phase4(res: Dict) -> None:
    fig, axes = plt.subplots(1, len(P_VALUES), figsize=(4.1 * len(P_VALUES), 4.2),
                             sharey=True)
    for ax, p in zip(np.atleast_1d(axes), P_VALUES):
        sweep = res["phase4"]["p=%.1f" % p]["sweep"]
        ks = [r["k"] for r in sweep]
        _errbar(ax, ks, sweep, "first_spy", "first-spy (baseline)", CMAP(1), "s")
        _errbar(ax, ks, sweep, "timing_ml", "phase-2 attack", CMAP(2), "^")
        _errbar(ax, ks, sweep, "dandelion_ml", "phase-4 attack", CMAP(0))
        ax.set_title("Dandelion, p = %.1f" % p)
        ax.set_xlabel("number of bribed nodes k")
        ax.grid(alpha=.3)
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("accuracy")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    _save(fig, "fig3_phase4_sweep.png")


def fig_attack_compare(res: Dict) -> None:
    kopt = res["phase2"]["optimal_k"]
    kmax = res["kmax"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, k in zip(axes, (kopt, kmax)):
        labels = ["flooding"] + ["p=%.1f" % p for p in P_VALUES]
        base = [res["phase2"]["sweep"][k - 1]["first_spy"]["mean"]] + \
               [res["phase4"]["p=%.1f" % p]["sweep"][k - 1]["first_spy"]["mean"]
                for p in P_VALUES]
        ph2 = [res["phase2"]["sweep"][k - 1]["timing_ml"]["mean"]] + \
              [res["phase4"]["p=%.1f" % p]["sweep"][k - 1]["timing_ml"]["mean"]
               for p in P_VALUES]
        ph4 = [res["phase2"]["sweep"][k - 1]["timing_ml"]["mean"]] + \
              [res["phase4"]["p=%.1f" % p]["sweep"][k - 1]["dandelion_ml"]["mean"]
               for p in P_VALUES]
        eb = [res["phase2"]["sweep"][k - 1]["first_spy"]["std"]] + \
             [res["phase4"]["p=%.1f" % p]["sweep"][k - 1]["first_spy"]["std"]
              for p in P_VALUES]
        e4 = [res["phase2"]["sweep"][k - 1]["timing_ml"]["std"]] + \
             [res["phase4"]["p=%.1f" % p]["sweep"][k - 1]["dandelion_ml"]["std"]
              for p in P_VALUES]
        x = np.arange(len(labels))
        ax.bar(x - 0.27, base, 0.27, yerr=eb, capsize=3, label="first-spy", color=CMAP(1))
        ax.bar(x, ph2, 0.27, label="phase-2 attack (timing ML)", color=CMAP(2))
        ax.bar(x + 0.27, ph4, 0.27, yerr=e4, capsize=3,
               label="phase-4 attack (Dandelion-aware)", color=CMAP(0))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("k = %d bribed nodes" % k)
        ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    _save(fig, "fig4_attack_comparison.png")


def fig_network(res: Dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["flooding"] + ["Dandelion\np=%.1f" % p for p in P_VALUES]
    t80 = [res["phase1"]["T80"]["mean"]] + \
          [res["phase3"]["p=%.1f" % p]["T80"]["mean"] for p in P_VALUES]
    t80e = [res["phase1"]["T80"]["std"]] + \
           [res["phase3"]["p=%.1f" % p]["T80"]["std"] for p in P_VALUES]
    msg = [res["phase1"]["messages_per_packet"]["mean"]] + \
          [res["phase3"]["p=%.1f" % p]["messages_per_packet"]["mean"] for p in P_VALUES]
    axes[0].bar(labels, t80, yerr=t80e, capsize=4, color=CMAP(0))
    axes[0].set_ylabel(r"$T_{80\%}$  (s)")
    axes[0].set_title("Time to reach 80% of the nodes")
    axes[0].grid(alpha=.3, axis="y")
    axes[1].bar(labels, msg, color=CMAP(3))
    axes[1].set_ylabel("UDP datagrams per packet")
    axes[1].set_title("Transmission cost")
    axes[1].grid(alpha=.3, axis="y")
    _save(fig, "fig5_network_cost.png")


def fig_phase5(res: Dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(P_VALUES))
    kmax = res["kmax"]
    no_delay = [res["phase4"]["p=%.1f" % p]["sweep"][kmax - 1]["dandelion_ml"]["mean"]
                for p in P_VALUES]
    naive = [res["phase5"]["p=%.1f" % p]["dandelion_ml_naive"]["mean"] for p in P_VALUES]
    comp = [res["phase5"]["p=%.1f" % p]["dandelion_ml_compensated"]["mean"]
            for p in P_VALUES]
    e0 = [res["phase4"]["p=%.1f" % p]["sweep"][kmax - 1]["dandelion_ml"]["std"]
          for p in P_VALUES]
    e2 = [res["phase5"]["p=%.1f" % p]["dandelion_ml_compensated"]["std"] for p in P_VALUES]
    axes[0].bar(x - 0.27, no_delay, 0.27, yerr=e0, capsize=3,
                label="phase 4 (no spy delay)", color=CMAP(0))
    axes[0].bar(x, naive, 0.27, label="phase 5, uncompensated model", color=CMAP(4))
    axes[0].bar(x + 0.27, comp, 0.27, yerr=e2, capsize=3,
                label="phase 5, compensated model", color=CMAP(3))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["p=%.1f" % p for p in P_VALUES])
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Phase 5 - effect of the spies' intentional delay (k=%d)" % kmax)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.3, axis="y")

    t0 = [res["phase3"]["p=%.1f" % p]["T80"]["mean"] for p in P_VALUES]
    t5 = [res["phase5"]["p=%.1f" % p]["T80"]["mean"] for p in P_VALUES]
    s0 = [res["phase3"]["p=%.1f" % p]["T80"]["std"] for p in P_VALUES]
    s5 = [res["phase5"]["p=%.1f" % p]["T80"]["std"] for p in P_VALUES]
    axes[1].bar(x - 0.18, t0, 0.36, yerr=s0, capsize=3, label="no spy delay", color=CMAP(0))
    axes[1].bar(x + 0.18, t5, 0.36, yerr=s5, capsize=3, label="spy delay = base delay",
                color=CMAP(3))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["p=%.1f" % p for p in P_VALUES])
    axes[1].set_ylabel(r"$T_{80\%}$ (s)")
    axes[1].set_title("Phase 5 - impact on the honest network")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.3, axis="y")
    _save(fig, "fig6_phase5.png")


def fig_scores(res: Dict) -> None:
    rows = res["scores"]
    kopt = res["phase2"]["optimal_k"]
    sel = [r for r in rows if r["k"] == kopt] or rows
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    names = [r["phase"] for r in sel]
    axes[0].barh(names, [r["score_adv"] for r in sel], color=CMAP(1))
    axes[0].set_xlabel(r"$Score_{adv} = Accuracy / k$")
    axes[0].set_title("Adversary score (k = %d)" % kopt)
    axes[0].grid(alpha=.3, axis="x")
    axes[1].barh(names, [r["score_honest"] for r in sel], color=CMAP(0))
    axes[1].set_xlabel(r"$Score_{honest} = (1/T_{80\%})(1-\mathrm{detection})$")
    axes[1].set_title("Honest-network score")
    axes[1].grid(alpha=.3, axis="x")
    axes[1].tick_params(labelleft=False)
    _save(fig, "fig7_scores.png")


def fig_coverage_curves(topo) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    runs = [("flooding", os.path.join(RESULTS, "runs", "flood_seed1"), CMAP(1))]
    for i, p in enumerate(P_VALUES):
        runs.append(("Dandelion p=%.1f" % p,
                     os.path.join(RESULTS, "runs", "dand_p%.1f_seed1" % p), CMAP(i + 2)))
    grid = np.linspace(0, 4.0, 400)
    for label, d, color in runs:
        if not os.path.exists(os.path.join(d, "events.jsonl")):
            continue
        run = load_run(d)
        curves = []
        for pid in run.pids:
            t0 = run.t_create.get(pid)
            if t0 is None:
                continue
            ts = np.sort(np.array(list(run.first_recv[pid].values())) - t0)
            curves.append(np.searchsorted(ts, grid, side="right") / run.n)
        if curves:
            ax.plot(grid, np.mean(curves, axis=0), label=label, color=color, lw=2)
    ax.axhline(0.8, color="crimson", ls="--", lw=1, label=r"80% threshold")
    ax.set_xlabel("time since packet creation (s)")
    ax.set_ylabel("fraction of nodes that received the packet")
    ax.set_title("Average diffusion curve (seed 1, 200 packets)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    _save(fig, "fig8_coverage.png")


# --------------------------------------------------------------------------
def tables(res: Dict) -> None:
    os.makedirs(TAB, exist_ok=True)
    lines: List[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Result tables\n")
    w("## Topology validation\n")
    for k, v in res["topology"].items():
        w("- `%s`: %s" % (k, v))
    w("\n## Phase 2 - k sweep (flooding), mean +/- std over %d seeds\n" % len(res["seeds"]))
    w("| k | spies | first-spy acc | timing-ML acc | Score_adv (first-spy) | Score_adv (ML) |")
    w("|---|-------|---------------|---------------|-----------------------|----------------|")
    for r in res["phase2"]["sweep"]:
        w("| %d | %s | %.3f ± %.3f | %.3f ± %.3f | %.4f | %.4f |"
          % (r["k"], r["spies"], r["first_spy"]["mean"], r["first_spy"]["std"],
             r["timing_ml"]["mean"], r["timing_ml"]["std"],
             r["first_spy_score_adv"]["mean"], r["timing_ml_score_adv"]["mean"]))
    w("\nOptimal number of spies: **k\\* = %d** (spies %s)\n"
      % (res["phase2"]["optimal_k"], res["phase2"]["optimal_spies"]))

    w("\n## Phase 3 - network behaviour\n")
    w("| protocol | T80 mean (s) | T80 median (s) | T80 std | datagrams/packet | stem hops |")
    w("|----------|--------------|----------------|---------|------------------|-----------|")
    w("| flooding | %.3f | %.3f | %.3f | %.1f | - |"
      % (res["phase1"]["T80"]["mean"], res["phase1"]["T80"]["median"],
         res["phase1"]["T80"]["std"], res["phase1"]["messages_per_packet"]["mean"]))
    for p in res["p_values"]:
        e = res["phase3"]["p=%.1f" % p]
        w("| Dandelion p=%.1f | %.3f | %.3f | %.3f | %.1f | %.2f |"
          % (p, e["T80"]["mean"], e["T80"]["median"], e["T80"]["std"],
             e["messages_per_packet"]["mean"], e["stem_hops"]["mean"]))

    w("\n## Phase 4 - attack accuracy against Dandelion\n")
    for p in res["p_values"]:
        w("\n**p = %.1f**\n" % p)
        w("| k | first-spy | phase-2 attack | phase-4 attack | Score_adv (phase 4) |")
        w("|---|-----------|----------------|----------------|---------------------|")
        for r in res["phase4"]["p=%.1f" % p]["sweep"]:
            w("| %d | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.4f |"
              % (r["k"], r["first_spy"]["mean"], r["first_spy"]["std"],
                 r["timing_ml"]["mean"], r["timing_ml"]["std"],
                 r["dandelion_ml"]["mean"], r["dandelion_ml"]["std"],
                 r["dandelion_ml_score_adv"]["mean"]))

    w("\n## Phase 5 - spies delaying their relays (k = %d)\n" % res["kmax"])
    w("| setting | accuracy (uncompensated) | accuracy (compensated) | accuracy without delay | T80 (s) |")
    w("|---------|--------------------------|------------------------|------------------------|---------|")
    for p in res["p_values"]:
        e = res["phase5"]["p=%.1f" % p]
        base = res["phase4"]["p=%.1f" % p]["sweep"][res["kmax"] - 1]["dandelion_ml"]["mean"]
        t0 = res["phase3"]["p=%.1f" % p]["T80"]["mean"]
        w("| Dandelion p=%.1f | %.3f | %.3f ± %.3f | %.3f | %.3f (was %.3f) |"
          % (p, e["dandelion_ml_naive"]["mean"], e["dandelion_ml_compensated"]["mean"],
             e["dandelion_ml_compensated"]["std"], base, e["T80"]["mean"], t0))
    f = res["phase5"]["flood"]
    w("| flooding | %.3f | %.3f | %.3f | %.3f (was %.3f) |"
      % (f["timing_ml_naive"]["mean"], f["timing_ml_compensated"]["mean"],
         res["phase2"]["sweep"][res["kmax"] - 1]["timing_ml"]["mean"],
         f["T80"]["mean"], res["phase1"]["T80"]["mean"]))

    w("\n## Final scores\n")
    w("| k | scenario | accuracy | baseline | T80 (s) | Score_adv | Score_honest |")
    w("|---|----------|----------|----------|---------|-----------|--------------|")
    for r in res["scores"]:
        w("| %d | %s | %.3f | %.3f | %.3f | %.4f | %.4f |"
          % (r["k"], r["phase"], r["accuracy"], r["accuracy_baseline"], r["T80"],
             r["score_adv"], r["score_honest"]))

    with open(os.path.join(TAB, "tables.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  tables: results/tables/tables.md")


def main() -> None:
    res = json.load(open(os.path.join(RESULTS, "results.json")))
    topo = get_topology()
    fig_topology(topo, sorted(res["spy_order"]))
    fig_phase2(res)
    fig_phase4(res)
    fig_attack_compare(res)
    fig_network(res)
    fig_phase5(res)
    fig_scores(res)
    fig_coverage_curves(topo)
    tables(res)


if __name__ == "__main__":
    main()
