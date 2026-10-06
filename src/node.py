"""A single network node.

Every node of the simulated network is an **independent OS process** listening
on its **own UDP port** on localhost (started by ``controller.py`` as
``python -m src.node <config.json>``).  All I/O is asynchronous (asyncio):

* the *data plane* is pure UDP - the only bytes ever put on the wire are
  ``"<packet-id>|<S|F>"``, i.e. the unique packet id and its Dandelion state.
  Neither the originator of a packet nor its creation time is ever
  transmitted; the previous hop is known only because UDP tells the receiver
  the source address of the datagram (exactly as in a real network).
* the *control plane* is a small line-oriented TCP server used by the
  simulator to inject packets and to shut the node down.  It is a simulator
  scaffold, not part of the studied protocol.

State kept by a node (assignment, part 1.1): NodeID (UUID), SelfAddr,
PeerList (with per-neighbour bookkeeping), SeenSet.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Set

STEM = "S"
FLUFF = "F"


class Peer:
    __slots__ = ("index", "node_id", "addr", "base_delay_ms",
                 "last_seen", "last_recv_from", "sent", "recv")

    def __init__(self, d: dict):
        self.index: int = d["index"]
        self.node_id: str = d["node_id"]
        self.addr = (d["host"], d["port"])
        self.base_delay_ms: float = d["base_delay_ms"]
        self.last_seen: Optional[float] = None      # last time we heard from it
        self.last_recv_from: Optional[str] = None   # last packet id received from it
        self.sent = 0
        self.recv = 0


class Node:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.index: int = cfg["index"]
        self.node_id: str = cfg["node_id"]
        self.host: str = cfg["host"]
        self.port: int = cfg["port"]
        self.ctrl_port: int = cfg["ctrl_port"]

        self.peers: List[Peer] = [Peer(p) for p in cfg["peers"]]
        self.by_addr: Dict[tuple, Peer] = {p.addr: p for p in self.peers}

        self.mode: str = cfg.get("mode", "flood")           # "flood" | "dandelion"
        self.p_stem: float = float(cfg.get("p", 0.9))
        self.flood_exclude_sender: bool = bool(cfg.get("flood_exclude_sender", False))
        self.jitter: float = float(cfg.get("jitter", 0.2))

        self.is_spy: bool = bool(cfg.get("is_spy", False))
        self.spy_delay_mode: str = cfg.get("spy_delay_mode", "none")

        # --- protocol state ------------------------------------------------
        self.seen: Set[str] = set()        # packets already fluffed/flooded here
        self.stem_seen: Set[str] = set()   # packets already forwarded in stem state

        self.rng = random.Random(cfg["seed"])
        self.log: List[dict] = []          # kept in RAM, dumped at shutdown
        self.transport: Optional[asyncio.DatagramTransport] = None
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ log
    def _log(self, ev: str, **kw) -> None:
        rec = {"ev": ev, "t": time.time(), "node": self.index}
        rec.update(kw)
        self.log.append(rec)

    def dump_log(self) -> None:
        with open(self.cfg["log_path"], "w") as fh:
            for rec in self.log:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # ----------------------------------------------------------------- send
    def _link_delay_ms(self, peer: Peer) -> tuple:
        """Return (total_delay_ms, spy_extra_ms).

        Delay_total = Delay_base + U(-0.2*base, +0.2*base); a spy of phase 5
        may add on top of it an intentional delay of at most one base delay.
        """
        base = peer.base_delay_ms
        total = base + self.rng.uniform(-self.jitter * base, self.jitter * base)
        extra = 0.0
        if self.is_spy and self.spy_delay_mode != "none":
            if self.spy_delay_mode == "max":
                extra = base
            elif self.spy_delay_mode == "uniform":
                extra = self.rng.uniform(0.0, base)
            elif self.spy_delay_mode == "half":
                extra = 0.5 * base
        return total + extra, extra

    def send(self, peer: Peer, pid: str, state: str) -> None:
        delay_ms, extra = self._link_delay_ms(peer)
        payload = f"{pid}|{state}".encode()
        self._log("send", pid=pid, to=peer.index, state=state,
                  delay_ms=round(delay_ms, 4), spy_extra_ms=round(extra, 4))
        peer.sent += 1
        loop = asyncio.get_running_loop()
        loop.call_later(delay_ms / 1000.0, self._transmit, peer, payload)

    def _transmit(self, peer: Peer, payload: bytes) -> None:
        try:
            self.transport.sendto(payload, peer.addr)  # type: ignore[union-attr]
        except Exception as exc:                        # pragma: no cover
            self._log("error", what="sendto", detail=repr(exc), to=peer.index)

    def broadcast(self, pid: str, state: str, exclude: Optional[int] = None) -> None:
        for peer in self.peers:
            if exclude is not None and peer.index == exclude:
                continue
            self.send(peer, pid, state)

    # -------------------------------------------------------------- receive
    def on_datagram(self, data: bytes, addr) -> None:
        now = time.time()
        peer = self.by_addr.get(addr)
        if peer is None:
            self._log("unexpected", what="datagram from unknown address",
                      detail=str(addr))
            return
        try:
            pid, state = data.decode().split("|", 1)
        except Exception:
            self._log("unexpected", what="malformed datagram", detail=repr(data[:32]))
            return
        peer.last_seen = now
        peer.last_recv_from = pid
        peer.recv += 1
        self._log("recv", pid=pid, frm=peer.index, state=state)
        if self.mode == "flood":
            self._handle_flood(pid, peer)
        else:
            self._handle_dandelion(pid, state, peer)

    # ------------------------------------------------------------- protocol
    def _handle_flood(self, pid: str, frm: Peer) -> None:
        if pid in self.seen:
            self._log("dup", pid=pid, frm=frm.index)
            return
        self.seen.add(pid)
        excl = frm.index if self.flood_exclude_sender else None
        self.broadcast(pid, FLUFF, exclude=excl)

    def _go_fluff(self, pid: str, exclude: Optional[int]) -> None:
        self.seen.add(pid)
        self._log("fluff", pid=pid)
        self.broadcast(pid, FLUFF, exclude=exclude)

    def _handle_dandelion(self, pid: str, state: str, frm: Peer) -> None:
        if state == FLUFF:
            if pid in self.seen:
                self._log("dup", pid=pid, frm=frm.index)
                return
            self.seen.add(pid)
            self.broadcast(pid, FLUFF, exclude=frm.index)
            return

        # ---- stem ----
        if pid in self.seen:            # we already fluffed it: nothing to do
            self._log("dup", pid=pid, frm=frm.index)
            return
        if pid in self.stem_seen:
            # the stem random walk came back to us: break the loop by fluffing.
            self._log("stem_loop", pid=pid, frm=frm.index)
            self._go_fluff(pid, exclude=frm.index)
            return
        self.stem_seen.add(pid)

        if self.rng.random() < self.p_stem:
            cands = [q for q in self.peers if q.index != frm.index]
            if cands:
                nxt = self.rng.choice(cands)
                self.send(nxt, pid, STEM)
                return
            self._log("stem_dead_end", pid=pid)
        self._go_fluff(pid, exclude=frm.index)

    # ---------------------------------------------------------- origination
    def create_packet(self, pid: str) -> None:
        self._log("create", pid=pid)
        if self.mode == "flood":
            self.seen.add(pid)
            self.broadcast(pid, FLUFF, exclude=None)
        else:
            self.stem_seen.add(pid)
            nxt = self.rng.choice(self.peers)
            self.send(nxt, pid, STEM)

    # --------------------------------------------------------------- server
    async def _handle_ctrl(self, reader: asyncio.StreamReader,
                           writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                cmd = msg.get("cmd")
                if cmd == "ping":
                    resp = {"ok": True, "node": self.index, "node_id": self.node_id}
                elif cmd == "inject":
                    self.create_packet(msg["pid"])
                    resp = {"ok": True}
                elif cmd == "stats":
                    resp = {"ok": True, "records": len(self.log),
                            "seen": len(self.seen), "stem_seen": len(self.stem_seen)}
                elif cmd == "shutdown":
                    resp = {"ok": True}
                    writer.write((json.dumps(resp) + "\n").encode())
                    await writer.drain()
                    self._stop.set()
                    break
                else:
                    resp = {"ok": False, "error": "unknown command"}
                writer.write((json.dumps(resp) + "\n").encode())
                await writer.drain()
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        class Proto(asyncio.DatagramProtocol):
            def __init__(self, node: "Node"):
                self.node = node

            def datagram_received(self, data, addr):
                self.node.on_datagram(data, addr)

            def error_received(self, exc):                 # pragma: no cover
                self.node._log("unexpected", what="udp error", detail=repr(exc))

        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: Proto(self), local_addr=(self.host, self.port))
        server = await asyncio.start_server(self._handle_ctrl, self.host, self.ctrl_port)
        self._log("ready", port=self.port, ctrl_port=self.ctrl_port,
                  node_id=self.node_id, is_spy=self.is_spy, mode=self.mode,
                  peers=[q.index for q in self.peers], pid=None)
        sys.stdout.write("READY %d\n" % self.index)
        sys.stdout.flush()

        await self._stop.wait()
        await asyncio.sleep(0.05)          # let the last call_later sends fire
        server.close()
        await server.wait_closed()
        if self.transport:
            self.transport.close()
        self._log("bye", peers_stat=[{"peer": q.index, "sent": q.sent, "recv": q.recv,
                                      "last_seen": q.last_seen} for q in self.peers])
        self.dump_log()


def main() -> None:
    cfg_path = sys.argv[1]
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    try:
        os.nice(-5)          # best effort: keep timing jitter low
    except Exception:
        pass
    node = Node(cfg)
    asyncio.run(node.run())


if __name__ == "__main__":
    main()
