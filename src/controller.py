"""Simulator / orchestrator.

The controller owns everything that a real network would *not* have: it
generates the topology, starts one OS process per node, injects the 200
packets of a scenario at random instants and, once the run is over, merges the
per-node logs into the "reference log" of the simulator (the only place where
the true originator of a packet is recorded).

It never participates in packet forwarding: nodes talk to each other over UDP
only.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

from .topology import Topology

HOST = "127.0.0.1"
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class RunConfig:
    tag: str
    mode: str = "flood"                 # "flood" | "dandelion"
    p: float = 0.9
    n_packets: int = 200
    inject_window_s: float = 15.0
    drain_s: float = 6.0
    seed: int = 1                       # scenario seed (sources + times + node RNG)
    spies: List[int] = field(default_factory=list)
    spy_delay_mode: str = "none"        # phase 5: "none"|"max"|"uniform"|"half"
    sources_from: str = "honest"        # "honest" (exclude spies) | "all"
    flood_exclude_sender: bool = False
    jitter: float = 0.2
    port_base: int = 47000


def _free_port_base(start: int, count: int) -> int:
    """Find an offset where 2*count consecutive ports are free."""
    base = start
    while base < start + 4000:
        ok = True
        for i in range(count):
            for port in (base + i, base + 1000 + i):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind((HOST, port))
                except OSError:
                    ok = False
                finally:
                    s.close()
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return base
        base += 100
    raise RuntimeError("no free port range")


class Controller:
    def __init__(self, topo: Topology, cfg: RunConfig, outdir: str):
        self.topo = topo
        self.cfg = cfg
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.procs: List[subprocess.Popen] = []
        self.socks: List[socket.socket] = []
        self.node_ids = [str(uuid.UUID(int=random.Random(f"{cfg.seed}:{i}").getrandbits(128),
                                       version=4)) for i in range(topo.n)]
        self.port_base = _free_port_base(cfg.port_base, topo.n)

    # ------------------------------------------------------------- scenario
    def scenario(self) -> List[dict]:
        """Packet schedule.

        Derived from the *scenario seed only*, so that the very same 200
        (source, time) pairs are replayed in every phase and for every value of
        p - which is what makes the comparison between phases fair.
        """
        rng = random.Random(self.cfg.seed * 977 + 11)
        spies = set(self.cfg.spies)
        pool = [i for i in range(self.topo.n)
                if self.cfg.sources_from == "all" or i not in spies]
        pkts = []
        times = sorted(rng.uniform(0.0, self.cfg.inject_window_s)
                       for _ in range(self.cfg.n_packets))
        for k, t in enumerate(times):
            pkts.append({"pid": "p%04d" % k, "src": rng.choice(pool), "t_rel": t})
        return pkts

    # --------------------------------------------------------------- launch
    def _node_config(self, i: int) -> dict:
        dmap = self.topo.delay_map()
        adj = self.topo.adjacency()
        peers = [{"index": j, "node_id": self.node_ids[j], "host": HOST,
                  "port": self.port_base + j, "base_delay_ms": dmap[(i, j)]}
                 for j in sorted(adj[i])]
        return {
            "index": i,
            "node_id": self.node_ids[i],
            "host": HOST,
            "port": self.port_base + i,
            "ctrl_port": self.port_base + 1000 + i,
            "peers": peers,
            "mode": self.cfg.mode,
            "p": self.cfg.p,
            "flood_exclude_sender": self.cfg.flood_exclude_sender,
            "jitter": self.cfg.jitter,
            "is_spy": i in set(self.cfg.spies),
            "spy_delay_mode": self.cfg.spy_delay_mode,
            "seed": self.cfg.seed * 100003 + i,
            "log_path": os.path.join(self.outdir, "node_%03d.jsonl" % i),
        }

    def start(self) -> None:
        cfgdir = os.path.join(self.outdir, "nodecfg")
        os.makedirs(cfgdir, exist_ok=True)
        for i in range(self.topo.n):
            path = os.path.join(cfgdir, "node_%03d.json" % i)
            with open(path, "w") as fh:
                json.dump(self._node_config(i), fh)
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "src.node", path],
                cwd=CODE_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.procs.append(proc)

        deadline = time.time() + 30
        for i in range(self.topo.n):
            s = None
            while time.time() < deadline:
                try:
                    s = socket.create_connection((HOST, self.port_base + 1000 + i), 1.0)
                    break
                except OSError:
                    if self.procs[i].poll() is not None:
                        err = self.procs[i].stderr.read().decode()  # type: ignore
                        raise RuntimeError("node %d died at startup:\n%s" % (i, err))
                    time.sleep(0.02)
            if s is None:
                raise RuntimeError("node %d never came up" % i)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socks.append(s)

    def _cmd(self, i: int, msg: dict, wait: bool = True) -> Optional[dict]:
        s = self.socks[i]
        s.sendall((json.dumps(msg) + "\n").encode())
        if not wait:
            return None
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf) if buf else None

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        pkts = self.scenario()
        self.start()
        for i in range(self.topo.n):
            self._cmd(i, {"cmd": "ping"})

        t0 = time.time() + 0.3
        for pk in pkts:
            target = t0 + pk["t_rel"]
            now = time.time()
            if target > now:
                time.sleep(target - now)
            pk["t_inject"] = time.time()
            self._cmd(pk["src"], {"cmd": "inject", "pid": pk["pid"]}, wait=True)

        time.sleep(self.cfg.drain_s)
        for i in range(self.topo.n):
            try:
                self._cmd(i, {"cmd": "shutdown"})
            except OSError:
                pass
        for i, proc in enumerate(self.procs):
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:      # pragma: no cover
                proc.send_signal(signal.SIGKILL)
        for s in self.socks:
            try:
                s.close()
            except OSError:
                pass

        meta = {
            "config": asdict(self.cfg),
            "t0": t0,
            "packets": [{"pid": p["pid"], "src": p["src"], "t_inject": p["t_inject"]}
                        for p in pkts],
            "node_ids": self.node_ids,
            "port_base": self.port_base,
            "topology_seed": self.topo.seed,
            "n_nodes": self.topo.n,
        }
        with open(os.path.join(self.outdir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        self.topo.to_json(os.path.join(self.outdir, "topology.json"))
        self._merge_logs()
        shutil.rmtree(os.path.join(self.outdir, "nodecfg"), ignore_errors=True)
        return meta

    def _merge_logs(self) -> None:
        events: List[dict] = []
        for i in range(self.topo.n):
            path = os.path.join(self.outdir, "node_%03d.jsonl" % i)
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                for line in fh:
                    events.append(json.loads(line))
            os.remove(path)
        events.sort(key=lambda r: r["t"])
        with open(os.path.join(self.outdir, "events.jsonl"), "w") as fh:
            for e in events:
                fh.write(json.dumps(e, separators=(",", ":")) + "\n")


def run_once(topo: Topology, cfg: RunConfig, outdir: str) -> str:
    ctl = Controller(topo, cfg, outdir)
    ctl.run()
    return outdir
